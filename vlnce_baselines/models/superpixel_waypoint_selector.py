import numpy as np
import torch.nn as nn
from scipy.spatial.distance import cdist
from vlnce_baselines.utils.acyclic_enforcer import AcyclicEnforcer
from vlnce_baselines.utils.map_utils import get_nearest_nonzero_waypoint


class WaypointSelector(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.resolution = config.MAP.MAP_RESOLUTION
        self.distance_threshold = 0.25 * 100 / self.resolution
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_waypoint = np.zeros(2)
        self._stick_current_waypoint = False
        self.change_threshold = self.config.EVAL.CHANGE_THRESHOLD
        # ================= NEW: 动量与距离惩罚参数 =================
        self._last_agent_pos = None  # 用于记录上一步位置以计算动量
        self.alpha_dist = 0.1       # 距离惩罚系数（根据你的地图尺度微调）
        self.beta_mom = 0.05          # 动量（惯性）奖励系数
        # ========================================================
        
    def reset(self) -> None:
        self._last_value = float("-inf")
        self._last_waypoint = np.zeros(2)
        self._stick_current_waypoint = False
        # ======== 记得补上这行 ========
        self._last_agent_pos = None 
        # ============================
        self._acyclic_enforcer.reset()
        
    def closest_point(self, points: np.ndarray, target_point: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(points - target_point, axis=1)
        
        return points[np.argmin(distances)]
    
    def _get_value(self, position: np.ndarray, value_map: np.ndarray) -> float:
        x, y = position
        value = value_map[x - 5 : x + 6, y - 5: y + 6]
        value = np.mean(value[value != 0])
        
        return value
        
    def forward(self, sorted_waypoints: np.ndarray, position: np.ndarray, collision_map: np.ndarray, 
                value_map: np.ndarray, fmm_dist: np.ndarray, traversible: np.ndarray, replan: bool):
        # ================= NEW: 计算智能体当前动量方向 =================
        agent_momentum = None
        if self._last_agent_pos is not None:
            movement = position - self._last_agent_pos
            dist_moved = np.linalg.norm(movement)
            if dist_moved > 1e-3:  # 只有发生了实际位移才计算动量
                agent_momentum = movement / dist_moved
        self._last_agent_pos = position.copy()
        # ==========================================================
        best_waypoint, best_value = None, None
        invalid_waypoint = False
        if not np.array_equal(self._last_waypoint, np.zeros(2)):
            if replan:
                invalid_waypoint = True

            if np.sum(collision_map) > 0:
                """ 
                check if the last_waypoint is too close to the current collision area
                """
                nonzero_indices = np.argwhere(collision_map != 0)
                distances = cdist([self._last_waypoint], nonzero_indices)
                if np.min(distances) <= 5:
                    invalid_waypoint = True
                    print("################################################ close to collision")
                    
            if np.linalg.norm(self._last_waypoint - position) < self.distance_threshold:
                invalid_waypoint = True
                print("################################################ achieve")
            
            x, y = int(position[0]), int(position[1])
            if fmm_dist is not None:
                print("fmm dist: ", np.mean(fmm_dist[x-10:x+11, y-10:y+11]), np.max(fmm_dist))
            if fmm_dist is not None and abs(np.mean(fmm_dist[x-10:x+11, y-10:y+11]) - np.max(fmm_dist)) <= 5.0:
                invalid_waypoint = True
                print("################################################ created an enclosed area!")
        
            if invalid_waypoint:
                idx = 0
                new_waypoint = sorted_waypoints[idx]
                distance_flag = np.linalg.norm(new_waypoint - position) < self.distance_threshold
                last_waypoint_flag = np.linalg.norm(new_waypoint - self._last_waypoint) < self.distance_threshold
                flag = distance_flag or last_waypoint_flag
                while ( flag and idx + 1 < len(sorted_waypoints)):
                    idx += 1
                    new_waypoint = sorted_waypoints[idx]
                self._last_waypoint = new_waypoint
                
            """ 
            if last_waypoint's current value doesn't get worse too much 
            then we stick to it.
            """
            curr_value = self._get_value(self._last_waypoint, value_map)
                
            if ((np.linalg.norm(self._last_waypoint - position) > self.distance_threshold) and 
                (curr_value - self._last_value > self.change_threshold)):
                best_waypoint = self._last_waypoint
        
        # if best_waypoint is None:
        #     for waypoint in sorted_waypoints:
        #         cyclic = self._acyclic_enforcer.check_cyclic(position, waypoint, 
        #                                                      threshold=0.5*100/self.resolution)
        #         if cyclic or np.linalg.norm(waypoint - position) < self.distance_threshold:
        #             continue
                
        #         best_waypoint= waypoint
        #         break
        
        # if best_waypoint is None:
        #     print("All waypoints are cyclic! Choosing the closest one.")
        #     best_waypoint = self.closest_point(sorted_waypoints, position)

        # ================= MODIFIED: 基于综合得分选择新 Waypoint =================
        if best_waypoint is None:
            best_score = float('-inf')
            valid_candidates = []
            
            # 先筛选出非循环且距离足够的候选点
            for waypoint in sorted_waypoints:
                cyclic = self._acyclic_enforcer.check_cyclic(position, waypoint, 
                                                             threshold=0.5*100/self.resolution)
                if cyclic or np.linalg.norm(waypoint - position) < self.distance_threshold:
                    continue
                valid_candidates.append(waypoint)
            
            # 综合打分：原始Value - 距离惩罚 + 动量奖励
            if len(valid_candidates) > 0:
                for wp in valid_candidates:
                    v_raw = self._get_value(wp, value_map)
                    dist = np.linalg.norm(wp - position)
                    
                    # 1. 距离惩罚 (越远扣分越多)
                    penalty = self.alpha_dist * dist
                    
                    # 2. 动量奖励 (顺路加分，逆路不扣分以免死胡同卡死)
                    bonus = 0.0
                    if agent_momentum is not None:
                        wp_dir = wp - position
                        wp_dist = np.linalg.norm(wp_dir)
                        if wp_dist > 1e-3:
                            wp_dir = wp_dir / wp_dist
                            cos_sim = np.dot(agent_momentum, wp_dir)
                            if cos_sim > 0: 
                                bonus = self.beta_mom * cos_sim
                                
                    final_score = v_raw - penalty + bonus
                    if final_score > best_score:
                        best_score = final_score
                        best_waypoint = wp
            else:
                print("All waypoints are cyclic or too close! Choosing the closest one.")
                best_waypoint = self.closest_point(sorted_waypoints, position)
        # =========================================================================
        
        if traversible[best_waypoint[0], best_waypoint[1]] == 0:
            best_waypoint = get_nearest_nonzero_waypoint(traversible, best_waypoint)
            
        best_value = self._get_value(best_waypoint, value_map)
        
        self._acyclic_enforcer.add_state_action(position, best_waypoint)
        self._last_value = best_value
        self._last_waypoint = best_waypoint
        
        return best_waypoint, best_value, sorted_waypoints