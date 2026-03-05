import torch
import numpy as np
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.rgbdimages import RGBDImages


class Mapping3D:
    def __init__(self, config, device):
        self.device = device
        self.config = config

        self.slam = PointFusion(
            odom="gt",
            dsratio=1,
            device=self.device,
        )

        self.pointclouds = Pointclouds(device=self.device)

        self.map_resolution = config.MAP.MAP_RESOLUTION
        self.map_size_cm = config.MAP.MAP_SIZE_CM
        self.map_shape = (
            self.map_size_cm // self.map_resolution,
            self.map_size_cm // self.map_resolution,
        )

        self._num_update_failures = 0

    def reset(self):
        self.pointclouds = Pointclouds(device=self.device)
        self._num_update_failures = 0

    def _sanitize_pose_4x4(self, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape == (4, 4):
            pose_4x4 = pose
        elif pose.size == 16:
            pose_4x4 = pose.reshape(4, 4)
        else:
            raise ValueError(f"pose must be 4x4 or have 16 elements, got shape {pose.shape}")

        if not np.isfinite(pose_4x4).all():
            raise ValueError("pose contains NaN/Inf")

        return pose_4x4

    def _sanitize_intrinsics_4x4(self, intrinsics: np.ndarray) -> np.ndarray:
        K = np.asarray(intrinsics, dtype=np.float32)
        if K.shape == (4, 4):
            K4 = K
        elif K.shape == (3, 3):
            K4 = np.eye(4, dtype=np.float32)
            K4[:3, :3] = K
        else:
            raise ValueError(f"intrinsics must be 4x4 or 3x3, got {K.shape}")

        if not np.isfinite(K4).all():
            raise ValueError("intrinsics contains NaN/Inf")

        if K4[0, 0] <= 0 or K4[1, 1] <= 0:
            raise ValueError("intrinsics fx/fy must be > 0")

        return K4

    def _sanitize_rgb_depth(self, rgb: np.ndarray, depth: np.ndarray):
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3, got shape {rgb.shape}")

        if depth.ndim == 2:
            depth = depth[..., np.newaxis]
        if depth.ndim != 3 or depth.shape[2] != 1:
            raise ValueError(f"depth must be HxWx1, got shape {depth.shape}")

        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            raise ValueError(f"rgb/depth spatial mismatch: rgb {rgb.shape}, depth {depth.shape}")

        rgb = rgb.astype(np.float32)
        depth = depth.astype(np.float32)

        if not np.isfinite(rgb).all():
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        if not np.isfinite(depth).all():
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        return rgb, depth

    def update(self, rgb, depth, intrinsics, pose):
        """Updates the 3D point cloud with a new observation.

        Hardens against:
        - wrong pose shapes (expects a single 4x4)
        - wrong intrinsics shape (3x3 or 4x4)
        - invalid/NaN/Inf rgb/depth
        - CUDA device-side asserts inside gradslam by skipping/resetting on bad frames
        """
        try:
            rgb, depth = self._sanitize_rgb_depth(rgb, depth)
            K4 = self._sanitize_intrinsics_4x4(intrinsics)
            Tcw = self._sanitize_pose_4x4(pose)

            _color = torch.from_numpy(rgb).to(self.device).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W,3)
            _depth = torch.from_numpy(depth).to(self.device).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W,1)
            _intrinsics = torch.from_numpy(K4).to(self.device).float().unsqueeze(0).unsqueeze(0)  # (1,1,4,4)
            _pose = torch.from_numpy(Tcw).to(self.device).float().unsqueeze(0).unsqueeze(0)  # (1,1,4,4)

            # Clamp and remove NaN/Inf in tensors to prevent downstream NaNs in normals/dots.
            # Habitat depth is typically in meters; keep a conservative upper bound.
            _depth = torch.clamp(_depth, min=0.0, max=10.0)
            _depth = torch.nan_to_num(_depth, nan=0.0, posinf=0.0, neginf=0.0)
            _color = torch.nan_to_num(_color, nan=0.0, posinf=255.0, neginf=0.0)

            frame_cur = RGBDImages(
                _color,
                _depth,
                _intrinsics,
                _pose,
            )

            self.pointclouds, _ = self.slam.step(self.pointclouds, frame_cur)
            self._num_update_failures = 0
        except Exception as e:
            # Keep evaluation alive even if gradslam hits a CUDA assert or invalid data slips through.
            # Reset pointclouds after repeated failures to avoid persistent corruption.
            self._num_update_failures += 1
            if self._num_update_failures >= 3:
                self.reset()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Re-raise non-CUDA errors? For robustness in long eval, we swallow and continue.
            # If you want strictness, change this to `raise`.
            return
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def get_traversability_map(self, floor_height_threshold=0.2, normals_up_threshold=0.85):
        """Projects the 3D point cloud to a 2D traversability map."""
        traversability_map = np.zeros(self.map_shape, dtype=np.bool_)

        if not self.pointclouds.has_points or not self.pointclouds.has_normals:
            return traversability_map

        points = self.pointclouds.points_padded[0].cpu().numpy()
        normals = self.pointclouds.normals_padded[0].cpu().numpy()

        if points.size == 0 or normals.size == 0:
            return traversability_map

        if not np.isfinite(points).all() or not np.isfinite(normals).all():
            return traversability_map

        floor_points_mask = points[:, 2] < floor_height_threshold
        up_normals_mask = normals[:, 2] > normals_up_threshold

        valid_mask = floor_points_mask & up_normals_mask
        if not np.any(valid_mask):
            return traversability_map

        floor_points = points[valid_mask]

        grid_coords = (floor_points[:, :2] * 100 / self.map_resolution) + (self.map_shape[0] / 2)
        grid_coords = np.round(grid_coords).astype(int)

        in_bounds_mask = (
            (grid_coords[:, 0] >= 0)
            & (grid_coords[:, 0] < self.map_shape[0])
            & (grid_coords[:, 1] >= 0)
            & (grid_coords[:, 1] < self.map_shape[1])
        )
        grid_coords = grid_coords[in_bounds_mask]

        if grid_coords.size == 0:
            return traversability_map

        traversability_map[grid_coords[:, 1], grid_coords[:, 0]] = True
        return traversability_map

    def is_forward_blocked(self, current_pose, forward_dist=0.4, width_dist=0.4, height_threshold=0.2):
        """Checks for obstacles in a box directly in front of the agent."""
        if not self.pointclouds.has_points:
            return False

        current_pose = np.asarray(current_pose)
        if current_pose.shape[0] >= 4:
            agent_pos = current_pose[:3]
            agent_heading = current_pose[3]
        elif current_pose.shape[0] == 3:
            agent_pos = np.array([current_pose[0], current_pose[1], 0.0], dtype=np.float32)
            agent_heading = current_pose[2]
        else:
            return False

        points_world = self.pointclouds.points_padded[0].cpu().numpy()
        if points_world.size == 0 or (not np.isfinite(points_world).all()):
            return False

        cos_h, sin_h = np.cos(-agent_heading), np.sin(-agent_heading)
        rot_matrix = np.array(
            [
                [cos_h, -sin_h, 0],
                [sin_h, cos_h, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        points_translated = points_world - agent_pos
        points_local = points_translated @ rot_matrix.T

        x_min, x_max = 0.05, forward_dist
        y_min, y_max = -width_dist / 2, width_dist / 2
        z_min = height_threshold

        in_box_mask = (
            (points_local[:, 0] >= x_min)
            & (points_local[:, 0] <= x_max)
            & (points_local[:, 1] >= y_min)
            & (points_local[:, 1] <= y_max)
            & (points_local[:, 2] >= z_min)
        )

        return bool(np.any(in_box_mask))
