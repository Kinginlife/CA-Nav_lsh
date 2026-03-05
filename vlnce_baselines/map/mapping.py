r""" 
We followed the process of constructing semantic map provided by chaplot.
However, their work doesn't support to build open-vocabulary semantic map.
We improved this by using dynamic feature map.

REFERENCE:
https://github.com/devendrachaplot/Object-Goal-Navigation/tree/master
"""

import os
import cv2
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import habitat_extensions.pose_utils as pu

from vlnce_baselines.utils.constant import *
from vlnce_baselines.utils.map_utils import *
import vlnce_baselines.utils.depth_utils as du
import vlnce_baselines.utils.visualization as vu
from vlnce_baselines.utils.data_utils import OrderedSet


class Semantic_Mapping(nn.Module): # 语义建图模块：将观测投影到局部/全局语义地图
    r"""
    Semantic_Mapping moudle: initialize map and do the projection.
    Projection procedure: 
    1. use depth observation to comupte a point cloud
    2. associate predicted semantic categories with each point in the point cloud
    3. project point cloud into 3D space to get voxel representation
    4. summing over height dimension
    """
    
    """
    Initialize map variables:
    Full map consists of multiple channels containing the following:
    1. Obstacle Map
    2. Explored Area
    3. Current Agent Location
    4. Past Agent Locations
    5,6,7,.. : Semantic Categories
    """
    MAP_CHANNELS = map_channels # map_channels is defined in constant.py

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()
        self.args = args
        self.dropout = 0.5 #no use
        self.n_channels = 3 #no use
        self.goal = None # 目标点（可视化用）
        self.curr_loc = None  # 当前位姿（全局坐标/或局部坐标，后续赋值）
        self.last_loc = None # 上一步位姿（用于轨迹绘制）
        self.vis_classes = []  # 可视化 legend 中已经添加的类别名列表（避免重复添加）
        
        self.fov = args.HFOV
        self.min_z = args.MIN_Z # a lager min_z could lost some information on the floor, 2cm is ok
        self.device = args.DEVICE
        self.du_scale = args.DU_SCALE # depth unit
        self.visualize = args.VISUALIZE # 是否实时显示窗口
        self.screen_w = args.FRAME_WIDTH 
        self.screen_h = args.FRAME_HEIGHT
        self.vision_range = args.VISION_RANGE # args.vision_range=100(cm) # 局部投影视野范围（cm，对应局部栅格边长）
        self.resolution = args.MAP_RESOLUTION
        self.print_images = args.PRINT_IMAGES  # 是否保存可视化图片到磁盘
        self.z_resolution = args.MAP_RESOLUTION
        self.num_environments = args.NUM_ENVIRONMENTS
        self.global_downscaling = args.GLOBAL_DOWNSCALING  # 全图到 局部图缩放比
        self.cat_pred_threshold = args.CAT_PRED_THRESHOLD  # 语义通道阈值（归一化用）
        self.exp_pred_threshold = args.EXP_PRED_THRESHOLD  # explored 阈值（归一化用）
        self.map_pred_threshold = args.MAP_PRED_THRESHOLD  # obstacle 阈值（归一化用）
        self.map_shape = (self.args.MAP_SIZE_CM // self.resolution,
                          self.args.MAP_SIZE_CM // self.resolution)
        self.map_size_cm = args.MAP_SIZE_CM // args.GLOBAL_DOWNSCALING
        
        if self.visualize or self.print_images: # 需要可视化才初始化画布
            self.vis_image = vu.init_vis_image() # 初始化大画布（拼接 rgb 与 map）
            self.rgb_vis = None  # 外部赋值：带分割标注的 rgb 图（BGR）

        # 72; 3.6m is about the height of one floor
        self.max_height = int(360 / self.z_resolution) # 体素高度上界（cm->格；360cm≈一层楼高度）
        
        # -8; we can use negative height to ensure information on the floor is contained
        self.min_height = int(-40 / self.z_resolution) # 体素高度下界（允许负值确保地面信息覆盖）
        self.agent_height = args.AGENT_HEIGHT * 100. # 0.88 * 100 = 88cm
        self.shift_loc = [# 将 agent 坐标系点云移到局部栅格中心，并设置初始朝向
            self.vision_range * self.resolution // 2, # x 平移：让 agent 处于局部窗口中
              0, 
              np.pi / 2.0]  # heading：让朝向对齐到地图坐标（这里约定为 90 度）       # [250, 0, pi/2] 
        self.camera_matrix = du.get_camera_matrix( # 相机内参矩阵（用于 depth->点云）
            self.screen_w, self.screen_h, self.fov)

        # feat's first channel is prepared for obstacles;
        self.feat = torch.ones(  # feat：用于 splat_feat_nd 的“每个点的通道特征”（动态扩展）
            args.NUM_ENVIRONMENTS, 
            1,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale  # 下采样后的像素点数（h/du * w/du）
        ).float().to(self.device)
    
    def reset(self) -> None:
        self.curr_loc = None
        self.last_loc = None
        self.vis_classes = []
        self.feat = torch.ones(
            self.args.NUM_ENVIRONMENTS, 1, self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)
        
        if self.visualize or self.print_images:
            self.vis_image = vu.init_vis_image()
            self.rgb_vis = None
    
    def _dynamic_process(self, num_detected_classes: int) -> None: # 动态扩展：根据当前检测类别数扩展网格/地图通道
        vr = self.vision_range  # 局部视野边长（格，或者以 cm 为单位但这里当格用）
        self.init_grid = torch.zeros(
            self.args.NUM_ENVIRONMENTS,
              1 + num_detected_classes,  # 1 个占位/障碍 + N 个语义类别（这里是“投影用通道数”，不是最终 map_channels）
              vr,  # x
              vr, # y
            self.max_height - self.min_height  # z（高度层数）
        ).float().to(self.device)
        
        if num_detected_classes > (self.feat.shape[1] - 1):  # 如果 feat 通道不够（除去第 0 通道占位）
            pad_num = 1 + num_detected_classes - self.feat.shape[1]  # 需要补多少通道
            feat_pad = torch.ones(  # 新增通道初始化为 1（与原实现一致，用作默认/占位）
                self.num_environments, 
                pad_num, 
                self.screen_h // self.du_scale * self.screen_w // self.du_scale
                ).float().to(self.device)
            self.feat = torch.cat([self.feat, feat_pad], axis=1) # 在通道维拼接扩展 feat
        
        new_nc = num_detected_classes + self.MAP_CHANNELS # 最终 map 需要的通道数 = 基础通道 + 语义通道
        if new_nc > self.local_map.shape[1]: # 如果 local_map/full_map 通道不足则补零
            pad_num = new_nc - self.local_map.shape[1]
            local_map_pad = torch.zeros(self.num_environments, 
                                        pad_num, 
                                        self.local_w, 
                                        self.local_h).float().to(self.device)
            full_map_pad = torch.zeros(self.num_environments, 
                                       pad_num, 
                                       self.full_w, 
                                       self.full_h).float().to(self.device)
            self.local_map = torch.cat([self.local_map, local_map_pad], axis=1)
            self.one_step_local_map = torch.cat([self.one_step_local_map, local_map_pad], axis=1)
            self.full_map = torch.cat([self.full_map, full_map_pad], axis=1)
            self.one_step_full_map = torch.cat([self.one_step_full_map, full_map_pad], axis=1)
            
    def _prepare(self, nc: int) -> None:  # 初始化 full/local map、pose、边界、state 等基础张量
        r"""Create empty full_map, local_map, full_pose, local_pose, origins, local map boundries
        Args:
        nc: num channels
        """
        
        r"""
        Calculating full and local map sizes
        args.global_downscaling = 2
        full_w = full_h = 480
        local_w, local_h = 240
        """ 
        self.full_w, self.full_h = self.map_shape  # 全局地图大小（格）
        self.local_w = int(self.full_w / self.global_downscaling) # 局部地图宽（格）
        self.local_h = int(self.full_h / self.global_downscaling)
        self.visited_vis = np.zeros(self.map_shape)  # 可视化用轨迹画布（numpy）
        
        r"""
        map_size_cm is the real world word map size(cm), i.e. (2400cm, 2400cm) <=> (24m, 24m)
        each element in the spatial map(full_map) corresponds to a cell of size (5cm, 5cm) in the physical world
        map_resolution = 5 so, the full_map should be (2400 / 5, 2400 / 5) = (480, 480)
        local_map is half of full_map, i.e. (240, 240)
        """
        self.full_map = torch.zeros(self.num_environments,   # full_map: [B, C, full_w, full_h]
                                    nc, 
                                    self.full_w, 
                                    self.full_h).float().to(self.device)
        self.one_step_full_map = torch.zeros(self.num_environments,   # one_step_full_map：一步内的 full_map（用于某些策略/可视化）
                                             nc, 
                                             self.full_w, 
                                             self.full_h).float().to(self.device)
        self.local_map = torch.zeros(self.num_environments,  # local_map: [B, C, local_w, local_h]
                                     nc,  
                                     self.local_w, 
                                     self.local_h).float().to(self.device)
        self.one_step_local_map = torch.zeros(self.num_environments, 
                                              nc, 
                                              self.local_w, 
                                              self.local_h).float().to(self.device)
        
        r"""
        pose.shape=(3,): [x, y, orientation]
        full pose: the agent always starts at the center of the map facing east
        """
        self.full_pose = torch.zeros(self.num_environments, 3).float().to(self.device) 
        self.local_pose = torch.zeros(self.num_environments, 3).float().to(self.device)
        self.curr_loc = torch.zeros(self.num_environments, 3).float().to(self.device)
        
        # Origin of local map
        self.origins = np.zeros((self.num_environments, 3)) # 每个环境 local_map 在全局坐标中的原点（m）

        # Local Map Boundaries
        self.lmb = np.zeros((self.num_environments, 4)).astype(int)   # local map boundaries：在 full_map 上的裁剪边界 [gx1,gx2,gy1,gy2]
        
        # state has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.state = np.zeros((self.num_environments, 7))  # state: [x,y,theta,gx1,gx2,gy1,gy2]（混合连续位姿+边界）
        
    def _get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes): # 根据 agent 在 full_map 的格子坐标，计算 local_map 在 full_map 的裁剪边界
        loc_r, loc_c = agent_loc # represent agent's position
        local_w, local_h = local_sizes # (240, 240)
        full_w, full_h = full_sizes # (480, 480)

        if self.global_downscaling > 1: # True, since args.global_downscaling = 2
            # calculate local map boundaries in full_map: width: (gx1, gx2); height: (gy1, gy2)
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2  # 窗口左上角（row/col）
            gx2, gy2 = gx1 + local_w, gy1 + local_h  # 窗口右下角（row/col）
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else: # 不裁剪：local==full
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]
        
    def init_map_and_pose(self, num_detected_classes: int): # episode 开始：初始化地图与位姿（并切出 local_map）
        r"""
        1. Initialize full_map as all zeros
        2. Initialize agent at the middle of the map
        3. extract the local map from the full map
        """
        
        nc = num_detected_classes + self.MAP_CHANNELS   # 总通道数 = 基础通道 + 语义通道数
        self._prepare(nc)  # 分配 full/local map 与 pose/state
        
        self.full_map.fill_(0.)  # 清空 full_map
        self.one_step_full_map.fill_(0.)
        self.full_pose.fill_(0.) # [bs, 3]
        
        # map_size_cm = 2400
        # full_pos[0]: [x=12m, y=12m, ori=0], agent always start at the center of the map
        self.full_pose[:, :2] = self.args.MAP_SIZE_CM / 100.0 / 2.0 # 初始位置放在全局地图中心（m）

        locs = self.full_pose.cpu().numpy()  
        self.state[:, :3] = locs # state: [x,y,z,gx1,gx2,gy1,gy2]  # state 前三维写入初始 pose
        for e in range(self.num_environments):
            r, c = locs[e, 1], locs[e, 0] # r,c = 12; r is x direction, c is y direction # 注意这里把 y 当 row，把 x 当 col（与地图数组索引对应）
            loc_r, loc_c = [int(r * 100.0 / self.resolution), # loc_r, loc_c = 12 * 100 / 5 = 240  # 将米转换为格子坐标：m->cm->cell
                            int(c * 100.0 / self.resolution)]
            
            # current and past agent location: agent takes a (3,3) square in the middle of the (480, 480) map. 
            # (3, 3) in spatial map <=> (15cm, 15cm) in physical world
            self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0 # 通道2/3：当前与历史位置（3x3 方块）
            self.one_step_full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0 # one_step 同样标记

            # lmb: [gx1, gx2, gy1, gy2]
            self.lmb[e] = self._get_local_map_boundaries((loc_r, loc_c), # 计算 local_map 在 full_map 上的裁剪边界
                                                (self.local_w, self.local_h),
                                                (self.full_w, self.full_h))
            self.state[e, 3:] = self.lmb[e]  # state 后四维写入边界
            
            # the origin of the local map is the top-lef corner of local map [6,6,0] meter
            self.origins[e] = [self.lmb[e][2] * self.resolution / 100.0, # local_map 原点（m）：全局坐标中 local 窗口左上角对应的位置
                            self.lmb[e][0] * self.resolution / 100.0, 0.]

        for e in range(self.num_environments):  # 从 full_map 裁剪 local_map，并设置 local_pose
            # extract the local map
            self.local_map[e] = self.full_map[e, :, self.lmb[e, 0] : self.lmb[e, 1], self.lmb[e, 2] : self.lmb[e, 3]]
            self.one_step_local_map[e] = self.one_step_full_map[e, :, self.lmb[e, 0] : self.lmb[e, 1], 
                                                                self.lmb[e, 2] : self.lmb[e, 3]]
            
            # local_pose initialized as (6,6,0) meter
            self.local_pose[e] = self.full_pose[e] - \
                torch.from_numpy(self.origins[e]).to(self.device).float() # local_pose = full_pose - origin
                
            self.curr_loc[e] = self.full_pose[e] - \
                torch.from_numpy(self.origins[e]).to(self.device).float()# curr_loc 同步为 local_pose
                                
    def update_map(self, step: int, detected_classes: OrderedSet, current_episode_id: int) -> None:  # 更新位姿相关通道 + 同步 local/full（不做新投影）
        if step == 0:
            self.last_loc = self.state[:, :3]
        else:
            self.last_loc = self.curr_loc
            
        # if step == 12:
        #     self.feat[:, 0, :] = 1.0
        #     for e in range(self.num_environments):
        #         self.local_map[e, 0, ...] = 0.0
                
        locs = self.local_pose.cpu().numpy()
        self.state[:, :3] = locs + self.origins # state 存全局 pose = local_pose + origin
        self.curr_loc = self.state[:, :3] # curr_loc 更新为全局 pose
        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        self.one_step_local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(self.num_environments):
            r, c = locs[e, 1], locs[e, 0]  # local 坐标（m）
            loc_r, loc_c = [int(r * 100.0 / self.resolution),
                        int(c * 100.0 / self.resolution)]
            self.local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.  # 标记当前+历史位置（3x3）
            self.one_step_local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
            
            self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
                    self.local_map[e]  # 把 local_map 写回 full_map 对应窗口
            self.one_step_full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
                    self.one_step_local_map[e]
            
            self.full_pose[e] = self.local_pose[e] + \
                    torch.from_numpy(self.origins[e]).to(self.device).float()  # full_pose = local_pose + origin（torch）
            
        if ((step + 1) % self.args.CENTER_RESET_STEPS) == 0:  # 定期重置 local 窗口中心（防止 agent 走到 local 边缘）
            for e in range(self.num_environments):
                self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
                    self.local_map[e]
                self.one_step_full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
                    self.one_step_local_map[e]
                
                # full_pose is actually global agent position.
                self.full_pose[e] = self.local_pose[e] + \
                    torch.from_numpy(self.origins[e]).to(self.device).float()
                locs = self.full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / self.resolution),
                                int(c * 100.0 / self.resolution)]
                self.lmb[e] = self._get_local_map_boundaries((loc_r, loc_c),
                                                  (self.local_w, self.local_h),
                                                  (self.full_w, self.full_h))
                self.state[e, 3:] = self.lmb[e]
                self.origins[e] = [self.lmb[e][2] * self.resolution / 100.0,
                              self.lmb[e][0] * self.resolution / 100.0, 0.]
                self.local_map[e] = self.full_map[e, :,
                                        self.lmb[e, 0]:self.lmb[e, 1],
                                        self.lmb[e, 2]:self.lmb[e, 3]]
                self.one_step_local_map[e] = self.one_step_full_map[e, :,
                                        self.lmb[e, 0]:self.lmb[e, 1],
                                        self.lmb[e, 2]:self.lmb[e, 3]]
                self.local_pose[e] = self.full_pose[e] - \
                    torch.from_numpy(self.origins[e]).to(self.device).float()
        # frontiers = find_frontiers(self.full_map[0].cpu().numpy(), detected_classes)
        # if self.print_images:
        #     plt.imshow(np.flipud(frontiers))
        #     save_dir = os.path.join(self.args.RESULTS_DIR, "frontiers/eps_%d"%current_episode_id)
        #     os.makedirs(save_dir, exist_ok=True)
        #     fn = "{}/step-{}.png".format(save_dir, step)
        #     plt.savefig(fn)
                
        if self.visualize:
            self._visualize(current_episode_id,   # 只可视化 id=0 的环境，节省资源
                            id=0,
                            goal=self.goal, 
                            detected_classes=detected_classes,
                            step=step)
        # torch.save(self.full_map, "/data/ckh/Zero-Shot-VLN-FusionMap/tests/full_maps/full_map%d.pt"%step)
        # torch.save(self.one_step_full_map, "/data/ckh/Zero-Shot-VLN-FusionMap/tests/one_step_full_maps/full_map%d.pt"%step)
        
        return (self.full_map.cpu().numpy(), 
                self.full_pose.cpu().numpy(), 
                # frontiers, 
                self.one_step_full_map.cpu().numpy())
    
    def _visualize(self,  # 可视化 rgb + 语义地图（用于 debug/论文展示）
                   current_episode_id: int, 
                   id: int=0,
                   goal: Tensor=None, 
                   detected_classes: OrderedSet=None,
                   step: int=None) -> None:
        """Try to visualize RGB images with segmentation and semantic map

        Args:
            id (int): since we are running a batch of environments, 
            it's resource consuming to render all environments together,
            so please only choose one environmet to visualize.
        """
        
        # the last item of detected_class is always "not_a_cat"
        if len(detected_classes[:-1]) > len(self.vis_classes): # 新增类别时更新 legend（排除最后一个 not_a_cat）
            vis_classes = copy.deepcopy(self.vis_classes) # 拷贝已可视化类列表
            for i in range(len(detected_classes[:-1]) - len(vis_classes)): # 逐个把新增类别加入 legend
                self.vis_image = vu.add_class(  # 在画布上添加类别条目
                    self.vis_image, 
                    5 + len(vis_classes) + i, 
                    detected_classes[i + len(vis_classes)], 
                    legend_color_palette) # legend 调色板
                self.vis_classes.append(detected_classes[i])
        
        local_maps = self.local_map.clone()
        local_maps[:, -1, ...] = 1e-5
        obstacle_map = local_maps[id, 0, ...].cpu().numpy()  # 障碍通道（numpy）
        explored_map = local_maps[id, 1, ...].cpu().numpy()  # explored 通道
        semantic_map = local_maps[id, 4:, ...].argmax(0).cpu().numpy() # 语义通道 argmax -> 类别索引（从 0 开始）
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = self.state[id]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        r, c = start_y, start_x
        start = [int(r * 100.0 / self.resolution - gx1),
                 int(c * 100.0 / self.resolution - gy1)] # get agent's location in local map
        start = pu.threshold_poses(start, obstacle_map.shape)
        
        last_start_x, last_start_y = self.last_loc[id][0], self.last_loc[id][1]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / self.resolution - gx1),
                        int(c * 100.0 / self.resolution - gy1)]
        last_start = pu.threshold_poses(last_start, obstacle_map.shape)
        self.visited_vis[gx1:gx2, gy1:gy2] = vu.draw_line(last_start, start, self.visited_vis[gx1:gx2, gy1:gy2]) # 在 visited_vis 上画轨迹线
        
        """
        color palette:
        0: out of map
        1: obstacles
        2: agent trajectory
        3: goal
        4 ~ num_detected_class: detected objects
        """
        semantic_map += 5  # palette 偏移：0-4 给特殊颜色，语义类别从 5 开始
        not_cat_id = local_maps.shape[1]
        not_cat_mask = (semantic_map == not_cat_id)
        obstacle_map_mask = np.rint(obstacle_map) == 1
        explored_map_mask = np.rint(explored_map) == 1
        
        semantic_map[not_cat_mask] = 0 # 默认：未分类区域设置为 0（out of map / 背景）
        
        m_free = np.logical_and(not_cat_mask, explored_map_mask)
        semantic_map[m_free] = 2 # palette id 2：轨迹/自由区域（与注释的 palette 定义对应）
        
        m_obstacle = np.logical_and(not_cat_mask, obstacle_map_mask)
        semantic_map[m_obstacle] = 1  # palette id 1：障碍
        
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        semantic_map[vis_mask] = 3
        color_pal = [int(x * 255.) for x in color_palette]
        
        # create a new image using palette mode
        # (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes)
        # in this mode, we can map colors to picture use a color palette
        sem_map_vis = Image.new("P", (semantic_map.shape[1], semantic_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        
        # put the flattened data, so that each instance will be mapped a color according to color palette
        sem_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        
        # flip image up and down, so that agnet's turn in simulator 
        # is the same as its turn in semantic map visualization
        sem_map_vis = np.flipud(sem_map_vis) # 上下翻转：让地图朝向与仿真器转向一致
        # sem_map_vis = np.array(sem_map_vis)
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]] # turn to bgr for opencv
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:655] = self.rgb_vis # 480, 640
        self.vis_image[50:530, 670:1150] = sem_map_vis # 480, 480
        
        pos = ( # 将 agent 的位置/朝向映射到可视化画布坐标系
            (start_x * 100. / self.resolution - gy1) * 480 / obstacle_map.shape[0],
            (obstacle_map.shape[1] - start_y * 100. / self.resolution + gx1) * 480 / obstacle_map.shape[1],
            np.deg2rad(-start_o)
        )
        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))  # 生成 agent 箭头轮廓点（右图起点偏移）
        cv2.waitKey(1)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1) # draw agent arrow
        
        if self.visualize:
            # cv2.imwrite('img_debug/ref.png', self.vis_image)
            cv2.imshow("Thread 1", self.vis_image) # 显示
            cv2.waitKey(1)
            
        if self.print_images:
            result_dir = self.args.RESULTS_DIR
            save_dir = "{}/visualization/eps_{}".format(result_dir, current_episode_id)
            os.makedirs(save_dir, exist_ok=True)
            fn = "{}/step-{}.png".format(save_dir, step)
            cv2.imwrite(fn, self.vis_image)

    def forward(self, obs: torch.Tensor, pose_obs: torch.Tensor):# 核心：将观测投影融合进 local_map
        """
        Args:
            obs: (b, c, h, w), b = batch size, c = 3(RGB) + 1(Depth) + num_detected_categories
        """
        # if use CoCo the number of categories is 16(i.e. c=16), but now open-vocabulary; 
        bs, c, h, w = obs.size() # batch 与维度
        depth = obs[:, 3, :, :] # depth.shape = (bs, H, W)  # 取 depth 通道（shape: [B,H,W]）
        
        # cut out the needed tensor from presupposed categories dimension
        num_detected_categories = c - 4 # 4=3(RGB) + 1(Depth)   # 动态语义通道数（除去 RGBD）
        self._dynamic_process(num_detected_categories) # 根据语义类别数动态扩展 init_grid/feat/map 通道

        # shape: [bs, h, w, 3] 3 is (x, y, z) for each point in (h, w)  # shape: [B,H,W,3]（每个像素一个 3D 点）
        point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)  # depth->点云（相机坐标系）
        
        agent_view_t = du.transform_camera_view_t(point_cloud_t, self.agent_height, 0, self.device)  # 相机坐标->agent 坐标（加上相机高度等）
        
        # point cloud in world axis
        # self.shift_loc=[250, 0, pi/2] => heading is always 90(degree), change with turn left
        # shape: [bs, h, w, 3] => (bs, 120, 160, 3)
        agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device)    # 将点云平移/旋转到局部网格中心坐标系

        max_h = self.max_height # 72  # z 上界（格）
        min_h = self.min_height # -8  # z 下界（格）
        xy_resolution = self.resolution  # xy 分辨率（cm/格）
        z_resolution = self.z_resolution # z 分辨率（cm/格）
        
        # vision_range = 100(cm)
        # in sem_exp.py _preprocess_depth(), all invalid depth values are set as 100 
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float() # (bs, x, y, 3) => (bs, 120, 160, 3)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] - vision_range // 2.) / vision_range * 2. # normalize to (-1, 1)
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] - (max_h + min_h) // 2.) / (max_h - min_h) * 2. # normalize
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3]) # [bs, 3, x*y]
        
        # obs: [b, c, h*w] => [b, 17, 19200], feat is a tensor contains all predicted semantic features
        pool = nn.AvgPool2d(self.du_scale) # 对语义 mask/features 做下采样平均池化
        # obs[:, 4, ...] = 0.
        self.min_z = int(25 / z_resolution - min_h) # 25 / 5 - (-8) = 13  # 设置 floor/障碍高度下界（经验值：25cm 上方开始）
        # self.min_z = 2 # use grounded-sam to detect floor
        self.feat[:, 1:, :] = pool(obs[:, 4:, :, :]).view(bs, c - 4, h // self.du_scale * w // self.du_scale)  # 将语义通道下采样并 flatten，写入 feat 的 1: 通道

        # self.init_grid: [bs, categories + 1, x=vr, y=vr, z=(max_height - min_height)] => [bs, 17, 100, 100, 80]
        # feat: average of all categories's predicted semantic features, [bs, 17, 19200]
        # XYZ_cm_std: point cloud in physical world, [bs, 3, 19200]
        # splat_feat_nd:
        assert self.init_grid.shape[1] == self.feat.shape[1], "init_grid and feat should have same number of channels!"
        
        # shape: [bs, num_detected_classes + 1, 100, 100, 80]
        voxels = du.splat_feat_nd(self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)  # 将点特征“splat”到 3D 体素网格
        max_z = int((self.agent_height + 1) / z_resolution - min_h) # int((88 + 1) / 5 - (-8))= 25
        
        agent_height_proj = voxels[..., self.min_z:max_z].sum(4) # shape: [bs, num_detected_classes + 1, 100, 100]
        all_height_proj = voxels.sum(4) # shape: [bs, num_detected_classes + 1, 100, 100]

        fp_map_pred = agent_height_proj[:, :1, :, :] # obstacle map
        fp_exp_pred = all_height_proj[:, :1, :, :] # explored map
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = self.local_pose

        agent_view = torch.zeros(bs, self.local_map.shape[1],
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device) # (bs, c, 480, 480) => full_map

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred # obstacle map
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred # explored area
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0) # semantic categories

        corrected_pose = pose_obs # sensor pose

        def get_new_pose_batch(pose, rel_pose_change):
            # pose: (bs, 3) -> x, y, ori(degree)
            # 57.29577951308232 = 180 / pi
            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose
        
        current_poses = get_new_pose_batch(self.local_pose, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        # get rotation matrix and translation matrix according to new pose (x, y, theta(degree))
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True) # shape: [bs, c, 240, 240]
        maps2 = torch.cat((self.local_map.unsqueeze(1), translated.unsqueeze(1)), 1)
        one_step_maps2 = torch.cat((self.one_step_local_map.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)
        one_step_map_pred, _ = torch.max(one_step_maps2, 1)
        self.local_map = map_pred
        self.one_step_local_map = one_step_map_pred
        self.local_pose = current_poses

        # return fp_map_pred, map_pred, pose_pred, current_poses