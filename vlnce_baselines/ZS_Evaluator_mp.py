import os
import pdb
import queue
import copy
import gzip
import json
import numpy as np
import time
from tqdm import tqdm
from PIL import Image
from fastdtw import fastdtw
from typing import List, Any, Dict
from collections import defaultdict
from skimage.morphology import binary_closing

import torch
from torch import Tensor
from torchvision import transforms

from habitat import Config, logger
from habitat_extensions.measures import NDTW
from habitat.core.simulator import Observations
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.environments import get_env_class
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.map.value_map import ValueMap
from vlnce_baselines.map.history_map import HistoryMap
from vlnce_baselines.map.direction_map import DirectionMap
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.Policy import FusionMapPolicy
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import gather_list_and_concat, get_device
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.common.constraints import ConstraintsMonitor
from vlnce_baselines.utils.constant import base_classes, map_channels

from pyinstrument import Profiler
import warnings
warnings.filterwarnings('ignore')


@baseline_registry.register_trainer(name="ZS-Evaluator-mp")
class ZeroShotVlnEvaluatorMP(BaseTrainer):
    def __init__(self, config: Config, segment_module=None, mapping_module=None) -> None:
        super().__init__()
        
        self.device = get_device(config.TORCH_GPU_ID)
        torch.cuda.set_device(self.device)
        self.config = config
        self.map_args = config.MAP
        self.visualize = config.MAP.VISUALIZE
        self.resolution = config.MAP.MAP_RESOLUTION #地图的分辨率（单位通常也是 cm/格，即每个栅格代表多少厘米）
        self.keyboard_control = config.KEYBOARD_CONTROL
        self.width = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        self.height = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
        self.max_step = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.map_shape = (config.MAP.MAP_SIZE_CM // self.resolution,
                          config.MAP.MAP_SIZE_CM // self.resolution) #地图在数组中的大小
        #把“实际世界尺寸（厘米）”转换成“地图栅格尺寸（cell 数）”，用于后面初始化一堆 2D 地图（floor / frontiers / traversible / visited ...）的数组大小
        
        self.trans = transforms.Compose([transforms.ToPILImage(),  #把任意输入图像 → 转成 PIL → 用最近邻 resize 到固定分辨率
                                         transforms.Resize(
                                             (self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH),  #160，120
                                             interpolation=Image.NEAREST)  #插值方式 Image.NEAREST（最近邻）
                                        ])
        
        self.classes = []
        self.current_episode_id = None
        self.current_detections = None
        self.map_channels = map_channels #4
        self.floor = np.zeros(self.map_shape)
        self.one_step_floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversible = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)
        self.min_constraint_steps = config.EVAL.MIN_CONSTRAINT_STEPS
        self.max_constraint_steps = config.EVAL.MAX_CONSTRAINT_STEPS
    
    def _set_eval_config(self) -> None:#把 eval runtime 信息写回 config（供子模块读取）
        print("set eval configs")
        self.config.defrost()
        self.config.MAP.DEVICE = self.config.TORCH_GPU_ID
        self.config.MAP.HFOV = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        self.config.MAP.AGENT_HEIGHT = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
        self.config.MAP.NUM_ENVIRONMENTS = self.config.NUM_ENVIRONMENTS
        self.config.MAP.RESULTS_DIR = self.config.RESULTS_DIR
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.config.freeze()
        
    def _init_envs(self) -> None:
        print("start to initialize environments")
        self.envs = construct_envs( # 构建 vector env   
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False, # done 后不自动 reset（由 rollout 控制）
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
        )
        print(f"local rank: {self.local_rank}, num of episodes: {self.envs.number_of_episodes}")
        self.detected_classes = OrderedSet() # 维护“已检测到的类别集合”（顺序稳定，用作 mask 通道索引）
        print("initializing environments finished!")
        
    def _collect_val_traj(self) -> None:  # 读取 GT 路径数据（用于 NDTW/SDTW）
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        with gzip.open(self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split)) as f:
            gt_data = json.load(f)

        self.gt_data = gt_data
        
    def _get_metrics_output_path(self) -> str:
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        return os.path.join(
            self.config.EVAL_CKPT_PATH_DIR,
            f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json",
        )

    def _atomic_json_dump(self, path: str, data: Any) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def _flush_metrics(self) -> None:
        # Incrementally persist metrics so a long eval run can be resumed/inspected.
        fname = self._get_metrics_output_path()
        self._atomic_json_dump(fname, self.state_eps)

    def _calculate_metric(self, infos: List):  # episode 结束时计算指标并写入 self.state_eps
        curr_eps = self.envs.current_episodes()
        info = infos[0]
        ep_id = curr_eps[0].episode_id
        gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float) # GT 路径点序列
        pred_path = np.array(info['position']['position'])# 预测轨迹点序列
        distances = np.array(info['position']['distance']) # 每一步到 goal 的距离序列
        gt_length = distances[0] # 用起点到 goal 的距离当作“最短路径长度”的近似
        dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]  # DTW 距离（用 NDTW 的欧氏距离）
        metric = {}
        metric['steps_taken'] = info['steps_taken']
        metric['distance_to_goal'] = distances[-1]
        metric['success'] = 1. if distances[-1] <= 3. else 0.
        metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
        metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
        # metric['collisions'] = info['collisions']['count'] / len(pred_path)
        metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
        metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
        metric['sdtw'] = metric['ndtw'] * metric['success']
        self.state_eps[ep_id] = metric
        print(self.state_eps[ep_id])
        self._flush_metrics()
        
    def _initialize_policy(self) -> None: # 初始化分割/建图/value_map/policy/约束模块
        print("start to initialize policy")
        # print(type(self.device))
        self.segment_module = GroundedSAM(self.config, self.device)
        #？？？？？？
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        
        self.value_map_module = ValueMap(self.config, self.mapping_module.map_shape, self.device) #价值地图模块（含 BLIP/VQA 等）
        self.history_module = HistoryMap(self.config, self.mapping_module.map_shape)# 历史惩罚地图
        self.direction_module = DirectionMap(self.config, self.mapping_module.map_shape)# 方向约束地图
        self.policy = FusionMapPolicy(self.config, self.mapping_module.map_shape[0])  # 策略：融合地图与价值输出动作
        self.policy.reset()
        
        self.constraints_monitor = ConstraintsMonitor(self.config, self.device) #ConstraintsMonitor(self.config, self.device)  # 约束检查模块（看 obs/detections/pose 判断是否满足）
        
    def _concat_obs(self, obs: Observations) -> np.ndarray:# 拼接 RGB 与 depth 成一个 state 张量
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1) # (h, w, c)->(c, h, w)
        
        return state
    
    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:  # 从 (RGB+depth) 生成 (RGB+depth+sem_mask) 的输入
        state = state.transpose(1, 2, 0) # (c,h,w)->(h,w,c)
        rgb = state[:, :, :3].astype(np.uint8) #[3, h, w]
        rgb = rgb[:,:,::-1] # RGB to BGR
        depth = state[:, :, 3:4] #[1, h, w]
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        env_frame_width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        #1111111
        sem_seg_pred = self._get_sem_pred(rgb) #[num_detected_classes, h, w] 语义 mask
        depth = self._preprocess_depth(depth, min_depth, max_depth) #[1, h, w] 度预处理到 cm/可用范围
        
        """
        ds: Downscaling factor
        args.env_frame_width = 640, args.frame_width = 160
        """
        #111111
        ds = env_frame_width // self.map_args.FRAME_WIDTH # ds = 4 # ds=4（把 640 缩到 160）
        if ds != 1:
            rgb = np.asarray(self.trans(rgb.astype(np.uint8))) # resize
            depth = depth[ds // 2::ds, ds // 2::ds] # down scaling start from 2, step=4
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2) # recover depth.shape to (height, width, 1)
        state = np.concatenate((rgb, depth, sem_seg_pred),axis=2).transpose(2, 0, 1) # (4+num_detected_classes, h, w)  # (C,H,W) 其中 C=3+1+num_classes
        
        return state
        
    def _get_sem_pred(self, rgb: np.ndarray) -> np.ndarray: # 调用 GroundedSAM 得到语义 mask
        """
        mask.shape=[num_detected_classes, h, w]
        labels looks like: ["kitchen counter 0.69", "floor 0.37"]
        """
        #111111
        masks, labels, annotated_images, self.current_detections = \
            self.segment_module.segment(rgb, classes=self.classes) # 分割：输入 BGR + 类别提示词
        self.mapping_module.rgb_vis = annotated_images # 保存可视化结果（可能用于 debug/视频）
        assert len(masks) == len(labels), f"The number of masks not equal to the number of labels!"
        print("current step detected classes: ", labels)
        class_names = self._process_labels(labels) # 去掉分数，更新 detected_classes
        masks = self._process_masks(masks, class_names) # 合并同类 mask，映射到稳定通道
        
        return masks.transpose(1, 2, 0)  # (C,H,W)->(H,W,C) 方便后续拼接
    
    def _process_labels(self, labels: List[str]) -> List:  # 从 "name score" 提取 name，并写入 detected_classes
        class_names = []
        for label in labels:
            class_name = " ".join(label.split(' ')[:-1])
            class_names.append(class_name)
            self.detected_classes.add(class_name)
        
        return class_names
        
    def _process_masks(self, masks: np.ndarray, labels: List[str]): # 合并同类实例 mask，并映射到 detected_classes 通道
        """Since we are now handling the open-vocabulary semantic mapping problem,
        we need to maintain a mask tensor with dynamic channels. The idea is to combine
        all same class tensors into one tensor, then let the "detected_classes" to 
        record all classes without duplication. Finally we can use each class's index
        in the detected_classes to determine as it's channel in the mask tensor.
        The organization of mask is similar to chaplot's Sem_Exp, please refer to this link:
        https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/agents/utils/semantic_prediction.py#L41
        
        Args:
            masks (np.ndarray): shape:(c,h,w), each instance(even the same class) has one channel
            labels (List[str]): masks' corresponding labels. len(masks) = len(labels)

        Returns:
            final_masks (np.ndarray): each mask will find their channel in self.detected_classes.
            len(final_masks) = len(self.detected_classes)
        """
        #111111
        if masks.shape != (0,):
            same_label_indexs = defaultdict(list)
            for idx, item in enumerate(labels):
                same_label_indexs[item].append(idx) #dict {class name: [idx]}
            combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
            for i, indexs in enumerate(same_label_indexs.values()):
                combined_mask[i] = np.sum(masks[indexs, ...], axis=0)
            
            idx = [self.detected_classes.index(label) for label in same_label_indexs.keys()]
            
            """
            max_idx = max(idx) + 1, attention: remember to add one becaure index start from 0
            init final masks as [max_idx + 1, h, w]; add not_a_category channel at last
            """
            final_masks = np.zeros((len(self.detected_classes), *masks.shape[1:]))
            final_masks[idx, ...] = combined_mask
        else:
            final_masks = np.zeros((len(self.detected_classes), self.height, self.width))
        
        return final_masks
    
    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:   # 深度预处理
        # Preprocesses a depth map by handling missing values, removing outliers, and scaling the depth values.
        depth = depth[:, :, 0] * 1   #(H,W,1)->(H,W)

        for i in range(depth.shape[1]): # 逐列处理
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()  # 把 0 替换成该列最大值（避免空洞）

        mask2 = depth > 0.99 # turn too far pixels to invalid  # 过远像素置为 invalid
        depth[mask2] = 0. # 置零作为 invalid

        mask1 = depth == 0 # invalid 像素
        depth[mask1] = 100.0 # then turn all invalid pixels to vision_range(100)  # invalid 设为 vision_range(100)（单位后续变成 cm）
        depth = min_depth * 100.0 + depth * max_depth * 100.0  # 把归一化深度映射到 [min,max] 并转 cm
        
        return depth # 返回 (H,W) 深度（cm）
    
    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:  # obs -> state（含语义
        concated_obs = self._concat_obs(obs)  # 拼接 RGB+depth
        state = self._preprocess_state(concated_obs)  # 加上语义 mask，resize/downsample
        
        return state # state.shape=(c,h,w) # 输出给 mapping_module
    
    #1111111
    def _batch_obs(self, n_obs: List[Observations]) -> Tensor:# 多环境 obs -> batch tensor（并对齐动态语义通道）
        n_states = [self._preprocess_obs(obs) for obs in n_obs] # 每个 env 预处理成 (C,H,W)
        max_channels = max([len(state) for state in n_states]) # 动态通道：取本 batch 最大通道数
        batch = np.stack([np.pad(state,  # 用 0 padding 把每个 state pad 到相同通道数
                [(0, max_channels - state.shape[0]),  # pad channel 维
                 (0, 0),  # H 不 pad
                 (0, 0)], # W 不 pad
                mode='constant') 
         for state in n_states], axis=0)  # (B,C,H,W)
        
        
        return torch.from_numpy(batch).to(self.device)
    
    def _random_policy(self):# 随机动作策略（调试/备用）
        action = np.random.choice([
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ])
        
        return {"action": action}

    def _process_classes(self, base_class: List, target_class: List) -> List: # 把目标类别放到 classes 列表末尾（并去重）
        for item in target_class:
            if item in base_class:
                base_class.remove(item) # 已存在则先移除（避免重复）
        base_class.extend(target_class) # 再追加到末尾（让目标更“显著”）
        
        return base_class
    
    def _check_destination(self, current_idx: int, sub_constraints: dict, llm_destination: str, decisions: dict) -> str:# 在约束+decision 中选择更合适的 destination
        for idx in range(current_idx, len(sub_constraints)):
                constraints = sub_constraints[str(idx)] # 当前子指令约束列表
                landmarks = decisions[str(idx)]["landmarks"]# decision 给出的 landmark 选择
                for constraint in constraints:
                    if constraint[0] == "direction constraint": # 方向约束不提供 destination
                        continue
    
                    else:
                        landmark = constraint[1]  # 约束里提到的 landmark/目标
                        for item in landmarks:
                            print(landmark, item)
                            if landmark in item:
                                choice = item[1] # decision 对该 landmark 的选择（如 move away/approach）
                            else:
                                continue
                            print(choice, choice != "move away")
                            if choice != "move away":
                                return constraint[1]
                            else:
                                break
        else:
            return llm_destination # 如果没找到合适的，则回退到 LLM 总 destination 
    
    def _process_llm_reply(self, obs: Observations):# 从 obs 中解析 llm_reply，并初始化子指令/约束/目标类别
        def _get_first_destination(sub_constraints: dict, llm_destination: str) -> str:  # 选第一个“非方向约束”的目标
            for constraints in sub_constraints.values():
                for constraint in constraints:
                    if constraint[0] != "direction constraint":
                        return constraint[1]
            else:
                return llm_destination
        
        self.llm_reply = obs['llm_reply'] # LLM 回复结构（由 LLM_SENSOR 注入到 obs）
        self.instruction = obs['instruction']['text'] # 原始自然语言指令
        self.sub_instructions = self.llm_reply['sub-instructions']  # 子指令序列
        self.sub_constraints = self.llm_reply['state-constraints'] # 每个子指令对应的约束（状态/方向等）
        self.decisions = self.llm_reply['decisions']  # LLM 的 decision 结构（含 landmarks 等）
        self.destination = _get_first_destination(self.sub_constraints, self.llm_reply['destination'])  #最近子指令目标
        print("!!!!!!!!!!!!!!! first destination: ", self.destination)
        # self.destination = self.sub_instructions[0]
        self.last_destination = self.destination    #上一步子指令目标
        first_landmarks = self.decisions['0']['landmarks']  #TODO 第一个decision没有landmark怎么办？例如turn around
        self.destination_class = [item[0] for item in first_landmarks]  # 抽取 landmark 类别名列表
        self.classes = self._process_classes(self.base_classes, self.destination_class)  # 更新 GroundedSAM 的类别提示
        self.constraints_check = [False] * len(self.sub_constraints) # 每个子指令是否完成约束检查
    
    #111111
    def _process_one_step_floor(self, one_step_full_map: np.ndarray, kernel_size: int=3) -> np.ndarray:# 从单步地图提取 floor（可走区域）
        #111111
        navigable_index = process_navigable_classes(self.detected_classes)  # 可走类别索引
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index] # 不可走类别索引
        one_step_full_map = remove_small_objects(one_step_full_map.astype(bool), min_size=64)  # 去小噪声
        
        obstacles = one_step_full_map[0, ...].astype(bool)  # 基础通道：障碍
        explored_area = one_step_full_map[1, ...].astype(bool) # 基础通道：已探索区域
        objects = np.sum(one_step_full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)  # 不可走语义对象合并
        navigable = np.logical_or.reduce(one_step_full_map[map_channels:, ...][navigable_index]) # 可走语义对象合并
        navigable = np.logical_and(navigable, np.logical_not(objects)) # 可走中剔除 objects
        
        free_mask = 1 - np.logical_or(obstacles, objects) # 自由空间=非障碍且非对象
        free_mask = np.logical_or(free_mask, navigable) # 把显式“可走语义”也算作 free
        floor = explored_area * free_mask # floor=探索过且自由
        floor = remove_small_objects(floor, min_size=400).astype(bool) # 去掉小块
        floor = binary_closing(floor, selem=disk(kernel_size))   # 闭运算平滑
        
        return floor # 返回 floor mask
        
       #111111
    def _process_map(self, step: int, full_map: np.ndarray, kernel_size: int=3) -> tuple: # 从全局地图生成 traversible/floor/frontiers
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        full_map = remove_small_objects(full_map.astype(bool), min_size=64)
        
        obstacles = full_map[0, ...].astype(bool)# 障碍通道
        explored_area = full_map[1, ...].astype(bool)# 探索通道
        objects = np.sum(full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)# 不可走对象合并
        
        selem = disk(kernel_size) # 结构元素（圆形） 
        obstacles_closed = binary_closing(obstacles, selem=selem) # 闭运算平滑障碍
        objects_closed = binary_closing(objects, selem=selem)# 闭运算平滑对象
        navigable = np.logical_or.reduce(full_map[map_channels:, ...][navigable_index])# 可走语义合并
        navigable = np.logical_and(navigable, np.logical_not(objects))# 可走中剔除对象
        navigable_closed = binary_closing(navigable, selem=selem)# 平滑可走区域
        
        untraversible = np.logical_or(objects_closed, obstacles_closed)# 不可通行=障碍或对象
        untraversible[navigable_closed == 1] = 0# 如果语义判断可走，则覆盖掉不可走
        untraversible = remove_small_objects(untraversible, min_size=64)
        untraversible = binary_closing(untraversible, selem=disk(3))
        traversible = np.logical_not(untraversible) # 可通行=非不可通行

        free_mask = 1 - np.logical_or(obstacles, objects)# 自由空间
        free_mask = np.logical_or(free_mask, navigable) # 加入可走语义
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, selem=selem)
        traversible = np.logical_or(floor, traversible)# floor 也强制算 traversible 
        
        explored_area = binary_closing(explored_area, selem=selem)# 平滑探索区域
        contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# 找探索边界轮廓
        image = np.zeros(full_map.shape[-2:], dtype=np.uint8) # 空白图像用于绘制轮廓
        image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)# 画轮廓线
        frontiers = np.logical_and(floor, image) # frontier=地面且位于探索边界附近
        frontiers = remove_small_objects(frontiers.astype(bool), min_size=64)

        return traversible, floor, frontiers.astype(np.uint8)# 返回可通行/地面/探索前沿
    #1111111
    def _maps_initialization(self): # episode 开始：reset env + 解析 llm + 初始化地图并融合首帧
        obs = self.envs.reset() #type(obs): list
        self._process_llm_reply(obs[0]) # 解析 LLM reply，初始化 destination/classes/constraints
        self.current_episode_id = self.envs.current_episodes()[0].episode_id
        print("current episode id: ", self.current_episode_id)
        
        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))# 初始化地图/位姿（语义通道数动态）
        batch_obs = self._batch_obs(obs)# obs -> batch tensor
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device) # 提取每个 env 的位姿
        self.mapping_module(batch_obs, poses)# 前向：把当前观测写入 mapping buffer
        full_map, full_pose, _ = self.mapping_module.update_map(0, self.detected_classes, self.current_episode_id)# 更新全局地图
        self.mapping_module.one_step_full_map.fill_(0.)# 清空单步 map buffer
        self.mapping_module.one_step_local_map.fill_(0.) #清空单步 local map buffer
    
    def _look_around(self):# episode 初始环视：原地转圈建图+估计 value map，然后选第一次 action
        print("\n========== LOOK AROUND ==========\n")
        full_pose, obs, dones, infos = None, None, None, None
        for step in range(0, 12):
            actions = []
            for _ in range(self.config.NUM_ENVIRONMENTS):
                actions.append({"action": HabitatSimActions.TURN_LEFT})
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)] # 解包 vector env 输出
            if dones[0]:
                return full_pose, obs, dones, infos
            batch_obs = self._batch_obs(obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses)  # 融合观测到 map
            full_map, full_pose, one_step_full_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)# 更新地图并取出 full/pose/one_step
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            self.traversible, self.floor, self.frontiers = self._process_map(step, full_map[0]) # 从 full_map 得到可通行/地面/前沿
            self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])# 单步 floor（用于近距离可走估计）
              #1111111          
            blip_value = self.value_map_module.get_blip_value(Image.fromarray(obs[0]['rgb']), self.destination) # 计算图像与 destination 的相关度/价值
            blip_value = blip_value.detach().cpu().numpy()
            value_map = self.value_map_module(step, full_map[0], self.floor, self.one_step_floor,  # 更新 value map（内部写 self.value_map）
                                              self.collision_map, blip_value, full_pose[0], 
                                  self.detected_classes, self.current_episode_id) #传入类别与 episode id 便于索引/记录
        self._action = self.policy(self.value_map_module.value_map[1], self.collision_map,# 环视完成后，用 value_map 选择第一步动作
                                    full_map[0], self.floor, self.traversible, 
                                    full_pose[0], self.frontiers, self.detected_classes,
                                    self.destination_class, self.classes, False, one_step_full_map[0], 
                                    self.current_detections, self.current_episode_id, False, step)# search_destination/replan 等 flag
        
        return full_pose, obs, dones, infos# 返回当前状态
    
    def _use_keyboard_control(self):
        a = input("action:")
        if a == 'w':
           return {"action": 1}
        elif a == 'a':
            return {"action": 2}
        elif a == 'd':
            return {"action": 3}
        else:
            return {"action": 0}
    
    def reset(self) -> None:
        self.classes = []
        self.current_detections = None
        self.detected_classes = OrderedSet()
        self.floor = np.zeros(self.map_shape)
        self.one_step_floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversible = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)
        
        self.policy.reset()
        self.mapping_module.reset()
        self.value_map_module.reset()
        self.history_module.reset()
    
    def rollout(self):# 执行单个 episode rollout（核心闭环：obs->map->value->policy->action）
        """
        execute a whole episode which consists of a sequence of sub-steps
        """
        self._maps_initialization()# reset env + 初始化地图 + 解析 llm
        full_pose, obs, dones, infos = self._look_around() # 环视建图 + 初始化 action
        print("\n ========== START TO NAVIGATE ==========\n")
        
        trajectory_points = []
        direction_points = []
        constraint_steps = 0 # 当前 constraint 已执行步数
        collided = 0  # 连续“未移动”的计数（用于判断卡住）
        empty_value_map = 0 # value map 过空的计数（用于触发重新环视）
        direction_map = np.ones(self.map_shape)  # 默认方向图（全 1 表示不施加方向约束）
        direction_map_exist = False # 是否已经生成 direction map
        replan = False # 是否触发重规划（卡住时 True）
        start_to_wait = False # 是否开始等待切换 constraint
        search_destination = False # 是否进入“寻找最终 destination”阶段（最后子指令之后）
        last_action, current_action = None, None
        last_pose, start_check_pose = None, None
        current_pose = full_pose[0] # 当前位姿（单 env 取第 0 个）
        self._action2 = None
        current_idx = self.constraints_check.index(False)  # 当前子指令索引（第一个未完成约束）
        landmarks = self.decisions[str(current_idx)]['landmarks'] # 当前子指令的 landmarks
        self.destination_class = [item[0] for item in landmarks] # 当前目标类别（供分割提示）
        self.classes = self._process_classes(self.base_classes, self.destination_class)# 更新分割类别提示列表
        current_constraint = self.sub_constraints[str(current_idx)] # 当前子指令的约束列表
        all_constraint_types = [item[0] for item in current_constraint] # 当前约束类型列表
        
        for step in range(12, self.max_step): # 从 12 开始（前 12 步用于环视）
            print(f"\nepisode:{self.current_episode_id}, step:{step}")
            print(f"instr: {self.instruction}")
            print(f"sub_instr_{current_idx}: {self.sub_instructions[current_idx]}") # 打印当前子指令
            constraint_steps += 1
            position = full_pose[0][:2] * 100 / self.resolution# 把米->cm->格（依 pose 定义）
            heading = full_pose[0][-1]  # 航向角
            print("full pose: ", full_pose[0])
            y, x = min(int(position[0]), self.map_shape[0] - 1), min(int(position[1]), self.map_shape[1] - 1)# 限幅到 map 内
            self.visited[x, y] = 1 # 标记访问（注意 x/y 顺序这里做了交换）
            trajectory_points.append((y, x)) # 轨迹点（用于 history）
            direction_points.append(np.array([x, y])) # 方向点（用于 direction constraint）
            if len(trajectory_points) > 2:
                del trajectory_points[0] # 只保留最近 2 个点（控制 history 计算窗口）
            if len(direction_points) > 5:
                del direction_points[0] # 只保留最近 5 个点（用于估计方向）
            #1111111
            history_map = self.history_module(trajectory_points, step, self.current_episode_id) # 生成 history 惩罚/权重图

            if "direction constraint" in all_constraint_types and start_check_pose is None:
                start_check_pose = full_pose[0] # 方向约束第一次出现时记录起点
            
            if int(current_idx) >= len(self.sub_instructions) - 1:
                search_destination = True # 最后子指令：开始面向最终 destination 搜索
                print("start to search destination")
                
            if sum(self.constraints_check) < len(self.sub_instructions): # 仍有未完成的子指令
                if (len(current_constraint) > 0 
                    and current_constraint[0][0] == "direction constraint" 
                    and not direction_map_exist): # 如果当前 constraint 是方向约束且还没生成 direction_map
                    direction = current_constraint[0][1] # 方向文本/类型（如 left/right/forward/back 等）
                    if len(direction_points) < 5:
                        current_position = direction_points[-1]# 当前点
                        last_five_position = direction_points[-1]  # 不足 5 点则用当前点代替
                    else:
                        current_position = direction_points[-1]
                        last_five_position = direction_points[0]
                    direction_map = self.direction_module(current_position, last_five_position, heading,# 生成方向权重图
                                                          direction, step, self.current_episode_id)
                    direction_map_exist = True  # 标记已生成
                else: 
                    direction_map = np.ones(self.map_shape) # 非方向约束则不施加方向限制
                
                check = self.constraints_monitor(current_constraint, obs[0],  # 检查当前约束是否满足
                                                self.current_detections, self.classes, 
                                                current_pose, start_check_pose) 
                print(current_constraint, check)
                if (len(current_constraint) > 0 
                    and current_constraint[0][0] == "direction constraint" 
                    and check[0] == True):
                    direction_map = np.ones(self.map_shape) # 方向约束满足后，取消方向限制
                
                if len(check) == 0:
                    print("empty constraint")
                elif sum(check) < len(check):
                    """update current_constraint, keep only items that don't meet constraints"""
                    current_constraint = [current_constraint[i] 
                                          for i in range(len(current_constraint)) 
                                          if not check[i]]# 删除已满足的 constraint
                    all_constraint_types = [item[0] for item in current_constraint] # 更新 constraint 类型列表
                if (sum(check) == len(check) or constraint_steps >= self.max_constraint_steps): # 全满足或超时
                    if not start_to_wait:
                        start_to_wait = True# 开始等待切换到下一个子指令
                        self.constraints_check[current_idx] = True    # 标记当前子指令完成
                if start_to_wait and (constraint_steps >= self.min_constraint_steps): # 至少执行了 min 步才切换
                    if False in self.constraints_check:
                        current_idx = self.constraints_check.index(False) # 取下一个未完成子指令
                        print(f"sub_instr_{current_idx}: {self.sub_instructions[current_idx]}")
                        landmarks = self.decisions[str(current_idx)]['landmarks']
                        if len(landmarks) > 0:
                            self.destination_class = [item[0] for item in landmarks]
                            self.classes = self._process_classes(self.base_classes, self.destination_class)
                        current_constraint = self.sub_constraints[str(current_idx)]
                        all_constraint_types = [item[0] for item in current_constraint]
                        current_pose, start_check_pose = None, None
                    else:
                        current_constraint, all_constraint_types = [], [] # 所有约束完成
                        print("all constraints are done")
                    constraint_steps = 0#  重置 constraint 步计数
                    start_to_wait = False  # 结束等待
                    
            print("current constraint: ", current_constraint)
            print("constraint_steps: ", constraint_steps)
                
            # process empty constraint and landmark
            if len(current_constraint) > 0 and current_constraint[0][0] != "direction constraint":
                new_destination = current_constraint[0][1] # 非方向约束：约束的实体目标
                if current_idx >= len(self.sub_instructions) - 1:
                    self.destination = self.llm_reply['destination'] # 最后阶段：用最终 destination
                else:
                    self.destination = new_destination# 否则用当前约束目标
            if len(current_constraint) == 0 and current_idx >=len(self.sub_constraints) - 1:
                self.destination = self.llm_reply['destination']
                
            if self.destination != self.last_destination:
                self.value_map_module.value_map[...] *= 0.5 # 目标切换时衰减旧 value_map（避免旧热点强影响）
                self.last_destination = self.destination# 更新 last_destination
                
            if np.sum(self.value_map_module.value_map[1].astype(bool)) <= 24**2:
                empty_value_map += 1 # value_map 太稀疏：计数+1
                constraint_steps = 0
            else:
                empty_value_map = 0 
            if empty_value_map >= 5:
                full_pose, obs, dones, infos = self._look_around()# 连续多次空 value_map：重新环视建图
                if dones[0]:
                    self._calculate_metric(infos)
                    break
                empty_value_map = 0
                constraint_steps = 0
            
            actions = []
            for _ in range(self.config.NUM_ENVIRONMENTS):
                if self.keyboard_control:
                    self._action2 =self._use_keyboard_control() 
                    actions.append(self._action2)
                else:
                    actions.append(self._action)# 使用 policy 给出的动作
            outputs = self.envs.step(actions)# 环境执行
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]# 解包返回
            
            if dones[0]:
                self._calculate_metric(infos)# episode 结束：计算指标
                break
            batch_obs = self._batch_obs(obs)# 预处理 obs（含动态语义通道
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses)# 融合到地图
            full_map, full_pose, one_step_full_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            
            self.traversible, self.floor, self.frontiers = self._process_map(step, full_map[0])
            self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])
            
            last_pose = current_pose# 记录上一 pose
            current_pose = full_pose[0]# 更新当前 pose
            if last_pose is not None and current_pose is not None:
                displacement = calculate_displacement(last_pose, current_pose, self.resolution) # 计算位移（格/米）
                if displacement < 0.2 * 100 / self.resolution: # 位移过小：认为发生碰撞/卡住（连续计数）
                    collided += 1
                else:
                    collided = 0
                    replan = False
                if collided >= 30:
                    replan = True# 连续卡住达到阈值：触发 replan
                    print(f"{self.current_episode_id}: {collided}\n")
                    fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR, 
                                        f"r{self.local_rank}_w{self.world_size}_collision_stuck.txt")
                    with open(fname, "a") as f:
                        f.writelines(f"id: {str(self.current_episode_id)}; step: {str(step)}; collided: {str(collided)}\n")
                
            last_action = current_action# 记录上一 action
            current_action = self._action# 当前 action（policy 输出）
            if last_pose is not None and current_action["action"] == 1:
                collision_map = collision_check_fmm(last_pose, current_pose, self.resolution, 
                                                self.mapping_module.map_shape) # 根据前进但没动推断碰撞区域
                self.collision_map = np.logical_or(self.collision_map, collision_map)# 累积碰撞区域（用于避开）
            
            blip_value = self.value_map_module.get_blip_value(Image.fromarray(obs[0]['rgb']), self.destination)# 图像与 destination 相似度
            blip_value = blip_value.detach().cpu().numpy()
            value_map = self.value_map_module(step, full_map[0], self.floor, self.one_step_floor, self.collision_map, 
                                  blip_value, full_pose[0], self.detected_classes, self.current_episode_id) # 更新 value map
            self._action = self.policy(self.value_map_module.value_map[1] * history_map, self.collision_map, # 用 value_map*history_map 做决策
                                    full_map[0], self.floor, self.traversible, 
                                    full_pose[0], self.frontiers, self.detected_classes,
                                    self.destination_class, self.classes, search_destination, 
                                    one_step_full_map[0], self.current_detections, 
                                    self.current_episode_id, replan, step)# 输出下一步动作
    
    def eval(self):# evaluator 入口：遍历本进程负责的 episodes，并写指标到 json
        self._set_eval_config()
        self._init_envs()
        self._collect_val_traj()
        self._initialize_policy()
        
        episodes_allowed = self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED
        if episodes_allowed is not None:
            eps_to_eval = len(episodes_allowed)
        elif self.config.EVAL.EPISODE_COUNT == -1: # -1 表示跑完全部 allowed episodes
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))# 否则取 min
            
        self.state_eps = {}# {episode_id: metric_dict}
        print(
            f"[DEBUG] EPISODES_ALLOWED={episodes_allowed} "
            f"type={type(episodes_allowed)} eps_to_eval={eps_to_eval} "
            f"envs.number_of_episodes={self.envs.number_of_episodes}"
        )
        t1 = time.time()
        for i in tqdm(range(eps_to_eval)):
            self.rollout()# 跑一个 episode
            self.reset()# 重置内部状态（不重建模块）
                    
        self.envs.close()
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR, 
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             ) # 输出文件名（每个 rank 一个文件，run_mp.py 会汇总）
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)
        t2 = time.time()
        logger.info(f"time: {t2 - t1}s")
        print("test time: ", t2 - t1)