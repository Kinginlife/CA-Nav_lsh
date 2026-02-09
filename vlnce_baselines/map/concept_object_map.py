import torch
import numpy as np
import open_clip
from PIL import Image
from typing import List, Tuple, Dict, Any

from supervision import Detections


def depth_to_pointcloud(depth: np.ndarray, intrinsics: Dict[str, float], pose: np.ndarray) -> np.ndarray:
    """
    Convert depth (meters) to point cloud in world coordinates.
    
    Args:
        depth: (H, W) depth in meters
        intrinsics: dict with fx, fy, cx, cy
        pose: (4,) [x, y, z, yaw] in meters and radians
    
    Returns:
        pcd_world: (N, 3) points in world frame
    """
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    
    # Pixel coordinates grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # Skip invalid depth (0 or >max_range)
    valid = (depth > 0) & (depth < 10.0)  # 10m max range
    u, v, d = u[valid], v[valid], depth[valid]
    
    # Camera coordinates
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d
    pcd_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N,3)
    
    # Transform to world using pose [x, y, z, yaw]
    x, y, z, yaw = pose
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos_yaw, -sin_yaw, 0],
                  [sin_yaw,  cos_yaw, 0],
                  [0,        0,       1]])
    t = np.array([x, y, z])
    pcd_world = (R @ pcd_cam.T).T + t  # (N,3)
    return pcd_world


def mask_to_pointcloud(mask: np.ndarray, depth: np.ndarray, intrinsics: Dict[str, float], pose: np.ndarray, voxel_size: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract point cloud within a binary mask.

    Notes:
        `depth_to_pointcloud` returns a point for each *valid depth pixel* only, so the pointcloud length
        is smaller than H*W. Therefore we must index using the same valid-depth mask, not `mask.ravel()`.

    Returns:
        pcd_world: (M,3) points
        center: (3,) mean point
    """
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return np.empty((0, 3)), np.empty(3)

    valid_depth = (depth > 0) & (depth < 10.0)
    joint = mask_bool & valid_depth
    if not joint.any():
        return np.empty((0, 3)), np.empty(3)

    pcd = depth_to_pointcloud(depth, intrinsics, pose)
    pcd_masked = pcd[joint[valid_depth]]

    # Voxel downsample
    if len(pcd_masked) > 0:
        voxel_grid = (pcd_masked / voxel_size).astype(int)
        _, uniq_idx = np.unique(voxel_grid, axis=0, return_index=True)
        pcd_masked = pcd_masked[uniq_idx]
    center = pcd_masked.mean(axis=0) if len(pcd_masked) > 0 else np.empty(3)
    return pcd_masked, center


class LocalCLIP:
    def __init__(self, model_name: str = "ViT-B-32-quickgelu", pretrained: str = "openai", device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: (H,W,3) uint8 RGB
        Returns:
            feat: (D,) np.float32
        """
        pil = Image.fromarray(image)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        feat = self.model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()


class ConceptObjectMap:
    """
    Lightweight object-centric map for online VLN.
    Each object stores:
        - class_name
        - center_3d (world coords, meters)
        - clip_ft (normalized)
        - last_seen_step
        - n_observations (for EMA)
    """
    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device
        self.objects = []  # List[Dict]
        self.step = 0
        # Fusion thresholds (tuneable)
        self.spatial_thresh = getattr(config, "OBJECT_SPATIAL_THRESH", 1.2)  # meters
        self.visual_thresh = getattr(config, "OBJECT_VISUAL_THRESH", 0.30)
        self.ema_alpha = getattr(config, "OBJECT_EMA_ALPHA", 0.7)

    def reset(self):
        self.objects.clear()
        self.step = 0

    def _match_to_existing(self, center: np.ndarray, clip_feat: np.ndarray) -> int:
        """
        Greedy nearest-neighbor matching using spatial + visual similarity.
        Returns index of matched object or -1 if none.
        """
        if len(self.objects) == 0:
            return -1
        centers = np.stack([obj["center_3d"] for obj in self.objects], axis=0)  # (N,3)
        feats = np.stack([obj["clip_ft"] for obj in self.objects], axis=0)      # (N,D)

        # Spatial gating
        dists = np.linalg.norm(centers - center[None, :], axis=1)  # (N,)
        spatial_mask = dists <= self.spatial_thresh
        if not spatial_mask.any():
            return -1

        # Visual similarity
        sims = (feats @ clip_feat)  # (N,)
        visual_mask = sims >= self.visual_thresh
        if not visual_mask.any():
            return -1

        # Combine: require both
        candidate_mask = spatial_mask & visual_mask
        if not candidate_mask.any():
            return -1
        candidate_idxs = np.where(candidate_mask)[0]
        # Pick highest visual similarity
        best_idx = candidate_idxs[np.argmax(sims[candidate_idxs])]
        return int(best_idx)

    def update(self,
               detections: Detections,
               rgb: np.ndarray,
               depth: np.ndarray,
               intrinsics: Dict[str, float],
               pose: np.ndarray,
               clip_encoder: LocalCLIP,
               class_names: List[str]):
        """
        detections: supervision.Detections with xyxy, mask, confidence, class_id
        rgb: (H,W,3) uint8 BGR (as used in GroundedSAM)
        depth: (H,W) in meters
        pose: (4,) [x,y,z,yaw]
        clip_encoder: LocalCLIP instance
        class_names: list of class strings aligned with detections.class_id
        """
        self.step += 1
        if len(detections) == 0:
            return

        # Preprocess depth to meters (if your depth is normalized, convert here)
        # Assuming depth already in meters; if not, adapt:
        # depth = depth * (max_depth - min_depth) + min_depth

        for i in range(len(detections)):
            xyxy = detections.xyxy[i].astype(int)
            mask = detections.mask[i]  # (H,W) bool
            conf = detections.confidence[i]
            cid = detections.class_id[i]
            class_name = class_names[cid] if cid is not None and cid < len(class_names) else "unknown"

            # Crop and encode image
            x1, y1, x2, y2 = xyxy
            crop = rgb[y1:y2+1, x1:x2+1][:, :, ::-1]  # BGR->RGB for CLIP
            if crop.size == 0:
                continue
            clip_feat = clip_encoder.encode_image(crop)

            # Extract point cloud and center
            pcd, center = mask_to_pointcloud(mask, depth, intrinsics, pose)
            if len(pcd) == 0:
                continue

            # Match to existing objects
            match_idx = self._match_to_existing(center, clip_feat)
            if match_idx >= 0:
                # EMA update
                obj = self.objects[match_idx]
                obj["center_3d"] = self.ema_alpha * center + (1 - self.ema_alpha) * obj["center_3d"]
                obj["clip_ft"] = self.ema_alpha * clip_feat + (1 - self.ema_alpha) * obj["clip_ft"]
                obj["last_seen_step"] = self.step
                obj["n_observations"] += 1
            else:
                # New object
                self.objects.append({
                    "class_name": class_name,
                    "center_3d": center.copy(),
                    "clip_ft": clip_feat.copy(),
                    "last_seen_step": self.step,
                    "n_observations": 1,
                })

    def query(self, text_ft: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Return top-K objects most similar to the text embedding.
        Each dict includes: idx, class_name, center_3d, similarity
        """
        if len(self.objects) == 0:
            return []
        feats = np.stack([obj["clip_ft"] for obj in self.objects], axis=0)
        sims = feats @ text_ft
        top_idxs = np.argsort(-sims)[:top_k]
        return [
            {
                "idx": int(idx),
                "class_name": self.objects[idx]["class_name"],
                "center_3d": self.objects[idx]["center_3d"].copy(),
                "similarity": float(sims[idx]),
            }
            for idx in top_idxs
        ]


class GoalPriorProjector:
    """
    Project 3D object centers to 2D map grid and generate a goal heatmap.
    """
    def __init__(self, map_shape: Tuple[int, int], resolution: float):
        """
        map_shape: (H, W) in grid cells
        resolution: meters per cell (e.g., 0.05)
        """
        self.map_shape = map_shape
        self.resolution = resolution
        self.sigma_cells = 3.0  # Gaussian spread in cells

    def world_to_map(self, point: np.ndarray) -> Tuple[int, int]:
        """
        point: (3,) world coords in meters
        Returns (row, col) in map grid; clamped to bounds.
        
        Logic aligned with ZS_Evaluator_mp.py rollout:
        position = full_pose[0][:2] * 100 / self.resolution
        y, x = min(int(position[0]), map_shape[0] - 1), min(int(position[1]), map_shape[1] - 1)
        """
        x_world, y_world, z_world = point
        # ZS_Evaluator_mp.py uses pos[0] for row (y) and pos[1] for col (x)
        # after converting meters to resolution-units (centimeters / resolution)
        row = int(np.floor(x_world * 100.0 / self.resolution))
        col = int(np.floor(y_world * 100.0 / self.resolution))
        
        row = np.clip(row, 0, self.map_shape[0] - 1)
        col = np.clip(col, 0, self.map_shape[1] - 1)
        return row, col

    def __call__(self, goal_objects: List[Dict], traversible: np.ndarray) -> np.ndarray:
        """
        goal_objects: list from ConceptObjectMap.query()
        traversible: (H,W) bool map (optional, used to mask out unreachable)
        Returns:
            heatmap: (H,W) float32, sum of Gaussians at object centers
        """
        heatmap = np.zeros(self.map_shape, dtype=np.float32)
        for obj in goal_objects:
            r, c = self.world_to_map(obj["center_3d"])
            # Only place if traversible (optional)
            if traversible is not None and not traversible[r, c]:
                continue
            # Add Gaussian blob
            H, W = self.map_shape
            y, x = np.ogrid[:H, :W]
            dist2 = (x - c) ** 2 + (y - r) ** 2
            blob = np.exp(-dist2 / (2 * self.sigma_cells ** 2))
            heatmap += blob * obj["similarity"]
        # Normalize to [0,1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap


def make_intrinsics_from_hfov(hfov: float, width: int, height: int) -> Dict[str, float]:
    """
    Build camera intrinsics dict from horizontal FoV and image size.
    Returns dict with fx, fy, cx, cy (pixels).
    """
    fx = width / (2 * np.tan(np.deg2rad(hfov) / 2))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}