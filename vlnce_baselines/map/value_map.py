"""
Value map moudle aims to calcluate cosine similarity
between current observation and destination description
"""

import os
import cv2
import torch
import torch.nn as nn

import numpy as np
from PIL import Image
from habitat import Config
from collections import Sequence
from typing import Union, Tuple, List
from lavis.models import load_model_and_preprocess
from transformers import AutoTokenizer, BertTokenizer
from skimage.morphology import remove_small_objects

from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.visualization import *


class ValueMap(nn.Module): # 价值地图：融合“观测得分 + 视野置信度 + floor 掩码”到全局地图
    def __init__(self, 
                 config: Config, 
                 full_value_map_shape: Union[Tuple, List, np.ndarray],
                 device: torch.device) -> None:
        super(ValueMap, self).__init__()
        self.config = config
        self.shape = full_value_map_shape
        self.visualize = config.MAP.VISUALIZE
        self.print_images = config.MAP.PRINT_IMAGES
        
        # two channels for value map: 
        # channel 0 is confidence map;
        # channel 1 is blip value map;
        self.value_map = np.zeros((2, *self.shape))
        self.accumulated_mask = np.zeros(self.shape)
        self.resolution = config.MAP.MAP_RESOLUTION
        self.hfov = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV
        self.radius = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        self.device = device
        # self.device = (torch.device("cuda", self.config.TORCH_GPU_ID) if 
        #                torch.cuda.is_available() else torch.device("cpu"))
        self.vis_image = np.ones((580, 480 * 3 + 20 * 4, 3)).astype(np.uint8) * 255 # 可视化画布（白底）
        self.previous_floor = np.zeros(self.shape)# 上一帧 floor
        self._load_model_from_disk()  # 从磁盘反序列化 BLIP2 模型与预处理器（离线）
        self.model.eval()
    
    def _create_model(self): # 在线创建 BLIP2（当前主流程不用，而是用 _load_model_from_disk）
        self.model, vis_processors, text_processors = \
            load_model_and_preprocess(
                "blip2_image_text_matching", 
                "pretrain", 
                device=self.device,
                is_eval=True)
        self.vis_processors = vis_processors["eval"]
        self.text_processors = text_processors["eval"]
        
    def _load_model_from_disk(self): # 离线加载：直接 torch.load 反序列化模型与 processor
        self.model = torch.load(self.config.BLIP2_MODEL_DIR, map_location="cpu").to(self.device)
        self._ensure_gelu_has_approximate(self.model) # 修复旧 torch 版本 GELU 没有 approximate 属性的问题

        # Re-create tokenizer to avoid compatibility issues from deserializing old tokenizer objects
        tokenizer = None
        try:
            tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", local_files_only=True
            )
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "bert-base-uncased", use_fast=False, local_files_only=True
                )
            except Exception:
                tokenizer = None

        if tokenizer is not None:
            if getattr(tokenizer, "pad_token", None) is None:
                try:
                    tokenizer.pad_token = "[PAD]"
                except Exception:
                    pass
            self._replace_tokenizers(self.model, tokenizer)

        self._ensure_tokenizer_attrs(self.model)

        self.vis_processors = torch.load(self.config.BLIP2_VIS_PROCESSORS_DIR)["eval"]
        self.text_processors = torch.load(self.config.BLIP2_TEXT_PROCESSORS_DIR)["eval"]

    @staticmethod
    def _ensure_gelu_has_approximate(model: nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, nn.GELU) and not hasattr(m, "approximate"):
                m.approximate = "none"

    @staticmethod
    def _replace_tokenizers(model: nn.Module, tokenizer) -> None:
        visited = set()

        def _walk(obj) -> None:
            oid = id(obj)
            if oid in visited:
                return
            visited.add(oid)

            if obj is None:
                return

            if hasattr(obj, "tokenize") and hasattr(obj, "convert_tokens_to_ids"):
                return

            if isinstance(obj, (str, bytes, int, float, bool)):
                return

            if isinstance(obj, (list, tuple, set)):
                for item in obj:
                    _walk(item)
                return

            if isinstance(obj, dict):
                for v in obj.values():
                    _walk(v)
                return

            if isinstance(obj, nn.Module):
                if hasattr(obj, "tokenizer"):
                    try:
                        setattr(obj, "tokenizer", tokenizer)
                    except Exception:
                        pass
                for child in obj.children():
                    _walk(child)

            try:
                for v in vars(obj).values():
                    _walk(v)
            except Exception:
                pass

        _walk(model)

    @staticmethod
    def _ensure_tokenizer_attrs(model: nn.Module) -> None:
        def _patch_tokenizer(tokenizer) -> None:
            if tokenizer is None:
                return
            if not hasattr(tokenizer, "unique_no_split_tokens"):
                tokenizer.unique_no_split_tokens = []
            if not hasattr(tokenizer, "added_tokens_encoder"):
                tokenizer.added_tokens_encoder = {}
            if not hasattr(tokenizer, "added_tokens_decoder"):
                tokenizer.added_tokens_decoder = {}

        _patch_tokenizer(getattr(model, "tokenizer", None))

        for obj in vars(model).values():
            if obj is None:
                continue
            if obj is model:
                continue
            if hasattr(obj, "tokenize") and hasattr(obj, "convert_tokens_to_ids"):
                _patch_tokenizer(obj)
            tokenizer = getattr(obj, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "tokenize") and hasattr(tokenizer, "convert_tokens_to_ids"):
                _patch_tokenizer(tokenizer)
    
    def _calculate_confidence(self, theta: np.ndarray) -> np.float64:
        return (np.cos(0.5 * np.pi * theta / (self.hfov / 2)))**2

    def _angle_to_vector(self, angle: np.ndarray) -> np.ndarray:  # 根据夹角 theta 计算“视野置信度”衰减
        angle_rad = np.radians(angle)
        x = np.cos(angle_rad)
        y = np.sin(angle_rad)
        
        return np.array([x, y])

    def _angle_between_vectors(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray: # 计算两向量夹角（度）
        # return [0, pi]
        dot_product = np.sum(vector1 * vector2, axis=0)
        vector1_length = np.linalg.norm(vector1, axis=0)
        vector2_length = np.linalg.norm(vector2, axis=0)
        angle = np.arccos(dot_product / (vector1_length * vector2_length))
        
        return np.degrees(angle)

    def _create_sector_mask(self, position: Sequence, heading: float):  # 用位姿创建扇形 mask（可视范围）与置信度 mask
        """ 
        arg "position" came from full pose, full pose use standard Cartesian coordinate.
        """
        mask = np.zeros(self.shape)
        confidence_mask = np.zeros(self.shape) # 置信度 mask：扇形内为衰减值，扇形外为 0
        heading = (360 - heading) % 360  # 坐标系转换：把 simulator heading 映射到地图角度约定
        angle_high = (heading + self.hfov / 2) % 360 # 扇形上边界角
        angle_low = (heading - self.hfov / 2) % 360   # 扇形下边界角
        heading = np.ones(self.shape) * heading   # broadcast：每个像素共享同一 heading
        heading_vector = self._angle_to_vector(heading)  # heading 单位向量（shape: (2,H,W)）

        y, x = np.meshgrid(np.arange(self.shape[0]) - position[0], np.arange(self.shape[1]) - position[1]) # 构建从 position 指向每个栅格的向量
        # x = np.flipud(x)
        distance = np.sqrt(x**2 + y**2) # 每个格子到 position 的距离（格）
        angle = np.arctan2(x, y) * 180 / np.pi
        angle = (360 - angle) % 360

        angle_vector = self._angle_to_vector(angle) # (2, 480, 480) # 每个格子方向向量（2,H,W）
        theta = self._angle_between_vectors(heading_vector, angle_vector)  # heading 与格子方向夹角（度）

        confidence = self._calculate_confidence(theta)  # 根据夹角得到衰减置信度（H,W）

        valid_distance = distance <= self.radius * 100 / self.resolution  # 半径限制：max_depth(m)->cm->格
        if angle_high > angle_low:
            valid_angle = (angle_low <= angle) & (angle <= angle_high)
        else:
            valid_angle = (angle_low <= angle) | (angle <= angle_high)
        mask[valid_distance & valid_angle] = 1
        confidence_mask[valid_distance & valid_angle] = confidence[valid_distance & valid_angle] # 置信度仅在扇形内赋值

        return mask, confidence_mask

    def _update_value_map(self,   # 用当前帧的 value 与 confidence 融合更新全局 value_map
                          prev_value: np.ndarray, 
                          curr_value: np.ndarray, 
                          prev_confidence: np.ndarray, 
                          curr_confidence: np.ndarray,
                          one_step_floor: np.ndarray,
                          mask: np.ndarray) -> np.ndarray:
        new_map_mask = np.logical_and(curr_confidence < 0.35, curr_confidence < prev_confidence) # 低置信度且不如历史的点
        curr_confidence[new_map_mask] = 0.0  # 对这些点把当前贡献置 0（避免噪声覆盖）
        new_value = curr_confidence * curr_value * self.current_floor + prev_confidence * prev_value # 置信度加权融合（只在 floor 上）
        new_confidence = (self.current_floor * curr_confidence)**2 + prev_confidence**2  # 置信度融合（平方强调高置信度）
        partion = curr_confidence * self.current_floor + prev_confidence # 分母：总权重（floor 外只剩 prev_confidence）
        partion[partion == 0] = np.inf  # 避免除 0（无权重的位置保持 0）
        new_value /= partion  # 归一化得到融合后的 value
        new_confidence /= partion # 归一化得到融合后的 confidence
        self.value_map[0][one_step_floor == 1] = new_confidence[one_step_floor == 1]
        self.value_map[1][one_step_floor == 1] = new_value[one_step_floor == 1]
        self.value_map *= self.current_floor
        
    def reset(self) -> None:
        self.value_map = np.zeros((2, *self.shape))
        self.vis_image = np.ones((580, 480 * 3 + 20 * 4, 3)).astype(np.uint8) * 255
    
    @torch.no_grad()
    def get_blip_value(self, image: Image, caption: str) -> torch.Tensor: # 用 BLIP2 计算图像-文本匹配得分（ITC）
        tokenizer = getattr(self.model, "tokenizer", None)

        # If the model swaps in an old deserialized tokenizer at runtime, force-patch it here.
        if tokenizer is not None:
            if not hasattr(tokenizer, "unique_no_split_tokens"):
                tokenizer.unique_no_split_tokens = []
            if not hasattr(tokenizer, "added_tokens_encoder"):
                tokenizer.added_tokens_encoder = {}
            if not hasattr(tokenizer, "added_tokens_decoder"):
                tokenizer.added_tokens_decoder = {}
            # Some versions store these as private fields
            if not hasattr(tokenizer, "_added_tokens_encoder"):
                tokenizer._added_tokens_encoder = getattr(tokenizer, "added_tokens_encoder", {})
            if not hasattr(tokenizer, "_added_tokens_decoder"):
                tokenizer._added_tokens_decoder = getattr(tokenizer, "added_tokens_decoder", {})

        if tokenizer is None:
            try:
                self.model.tokenizer = BertTokenizer.from_pretrained(
                    "bert-base-uncased", local_files_only=True
                )
            except Exception:
                try:
                    self.model.tokenizer = AutoTokenizer.from_pretrained(
                        "bert-base-uncased", use_fast=False, local_files_only=True
                    )
                except Exception:
                    pass

        self._ensure_tokenizer_attrs(self.model)
        img = self.vis_processors(image).unsqueeze(0).to(self.device) # 图像预处理 -> [1,3,H,W]
        txt = self.text_processors(caption)  # 文本预处理（可能返回字符串或 tokenized 文本，取决于 lavis）
        itc_score = self.model({"image": img, "text_input": txt}, match_head='itc') # BLIP2 ITC 得分（越大越匹配）
        
        return itc_score
    
    def forward(self,  
                step: int,
                full_map: np.ndarray, 
                floor: np.ndarray,
                one_step_floor: np.ndarray,
                collision_map: np.ndarray,
                blip_value: np.ndarray,
                full_pose: Sequence,
                classes: List,
                current_episode_id: int):    # 主入口：把 blip_value 投影并融合到 value_map
        """project cosine similarity to floor

        Args:
            local_map (np.array): one step local map, current observation's 
                                  2D Top-down semantic map. shape: [c,h,w] 
                                  no batch dimension
            value (torch.Tensor): torch.size([1,1]) on device
        """
        self.current_floor = floor  # 当前 floor mask（H,W），通常 1  表示可行走地面
        self.current_floor[collision_map == 1] = 0 # 碰撞区域从 floor 排除（避免把价值投到不可走区域）
        position = full_pose[:2] * (100 / self.resolution)
        heading = full_pose[-1]
        mask, confidence_mask = self._create_sector_mask(position, heading)  # 视野扇形 mask + 置信度 mask
        current_confidence = confidence_mask
        previous_confidence = self.value_map[0]
        current_value = blip_value
        previous_value = self.value_map[1]
        self._update_value_map(previous_value, current_value, previous_confidence, current_confidence, one_step_floor, mask)
        if self.visualize:
            self._visualize(step, current_episode_id)
        
        return self.value_map[1]
        
    def _visualize(self, step: int, current_episode_id: int):  # 显示 floor/confidence/value 三图
        confidence_mask_vis = cv2.convertScaleAbs(self.value_map[0] * 255)
        confidence_mask_vis = np.stack((confidence_mask_vis,) * 3, axis=-1)
        value_map_vis = self.value_map[1]
        
        min_val = np.min(value_map_vis)
        max_val = np.max(value_map_vis)
        normalized_values = (value_map_vis - min_val) / (max_val - min_val + 1e-8)
        normalized_values[value_map_vis == 0] = 1
        value_map_vis = cv2.applyColorMap((normalized_values* 255).astype(np.uint8), cv2.COLORMAP_HOT)
        floor_vis = cv2.convertScaleAbs(self.current_floor * 255)
        floor_vis = np.stack((floor_vis,) * 3, axis=-1)
        self.vis_image[80 : 560, 20 : 500] = np.flipud(floor_vis)
        self.vis_image[80: 560, 520 : 1000] = np.flipud(confidence_mask_vis)
        self.vis_image[80: 560, 1020: 1500] = np.flipud(value_map_vis)
        
        self.vis_image = add_text(self.vis_image, "Floor", (20, 50))
        self.vis_image = add_text(self.vis_image, "Confidence Mask", (520, 50))
        self.vis_image = add_text(self.vis_image, "Value Map", (1020, 50))
        
        cv2.imshow("info", self.vis_image)
        cv2.waitKey(1)
        
        if self.print_images:
            save_dir = os.path.join(self.config.RESULTS_DIR, "floor_confidence_value/eps_%d"%current_episode_id)
            os.makedirs(save_dir, exist_ok=True)
            fn = "{}/step-{}.png".format(save_dir, step)
            cv2.imwrite(fn, self.vis_image)