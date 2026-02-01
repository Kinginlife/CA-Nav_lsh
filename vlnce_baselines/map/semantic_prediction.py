import attr
import time
from typing import Any, Union, List, Tuple
from abc import ABCMeta, abstractmethod

import cv2
import torch
import numpy as np

from habitat import Config

import supervision as sv
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

from vlnce_baselines.map.RepViTSAM.setup_repvit_sam import build_sam_repvit
from vlnce_baselines.common.utils import get_device


VisualObservation = Union[torch.Tensor, np.ndarray]


@attr.s(auto_attribs=True)
class Segment(metaclass=ABCMeta):
    config: Config
    device: torch.device
    
    def __attrs_post_init__(self):
        self._create_model(self.config, self.device)
    
    @abstractmethod
    def _create_model(self, config: Config, device: torch.device) -> None:
        pass
    
    @abstractmethod
    def segment(self, image: VisualObservation, **kwargs) -> Any:
        pass
    

@attr.s(auto_attribs=True)
class GroundedSAM(Segment):
    height: float = 480.
    width: float = 640.
    
    def _create_model(self, config: Config, device: torch.device) -> Any: # GroundingDINO（检测框）+ SAM（mask）的组合
        GROUNDING_DINO_CONFIG_PATH = config.MAP.GROUNDING_DINO_CONFIG_PATH # GroundingDINO 配置文件路径
        GROUNDING_DINO_CHECKPOINT_PATH = config.MAP.GROUNDING_DINO_CHECKPOINT_PATH # GroundingDINO 权重路径
        SAM_CHECKPOINT_PATH = config.MAP.SAM_CHECKPOINT_PATH  # SAM 权重路径（ViT-H/B/L 等）
        SAM_ENCODER_VERSION = config.MAP.SAM_ENCODER_VERSION  # sam_model_registry key（例如 "vit_h"）
        RepViTSAM_CHECKPOINT_PATH = config.MAP.RepViTSAM_CHECKPOINT_PATH # RepViT-SAM 权重路径（如果启用）
        # device = torch.device("cuda", config.TORCH_GPU_ID if torch.cuda.is_available() else "cpu")
        
        self.grounding_dino_model = Model( # 构建 GroundingDINO 推理模型（内部会 load 权重）
            model_config_path=GROUNDING_DINO_CONFIG_PATH, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=device
            )
        if config.MAP.REPVITSAM:  # 是否使用 RepViT-SAM（更轻量的 SAM 变体）
            sam = build_sam_repvit(checkpoint=RepViTSAM_CHECKPOINT_PATH)
            sam.to(device=device)
        else:
            sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=device)
        self.sam_predictor = SamPredictor(sam)  # SAM 推理 wrapper：输入图像后可对 box 点/框预测 mask
        self.box_threshold = config.MAP.BOX_THRESHOLD # GroundingDINO box 置信度阈值
        self.text_threshold = config.MAP.TEXT_THRESHOLD  # GroundingDINO text 匹配阈值
        self.grounding_dino_model.model.eval()  # 推理模式：关闭 dropout 等
        
    def _segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)  # 给 SAM 设置当前图像（需要 RGB，且形状 HWC）
        result_masks = []  # 每个检测框对应一个最终 mask（选得分最高的那张）
        for box in xyxy:  # 遍历每个检测框（xyxy 格式）
            masks, scores, logits = sam_predictor.predict(  # SAM 对单个 box 预测多个候选 mask
                box=box,  # numpy: [x1,y1,x2,y2]
                multimask_output=True  # 输出多个候选 mask（通常 3 个）
            )
            index = np.argmax(scores) # 选择得分最高的候选 mask
            result_masks.append(masks[index]) # masks[index] shape: (H, W) bool
        return np.array(result_masks) 
    
    def _process_detections(self, detections: sv.Detections) -> sv.Detections: # 过滤“几乎覆盖整张图”的异常框
        box_areas = detections.box_area  # 每个框面积（像素级）
        i = len(detections) - 1 # 从后往前删（避免 index 变化影响遍历）
        while i >= 0:
            if box_areas[i] / (self.width * self.height) < 0.95: # 框占整图比例 < 0.95：保留（认为不是“整屏误检”）
                i -= 1
                continue
            else: # 否则删除该检测（通常是把整张图都框住的异常框）
                detections.xyxy = np.delete(detections.xyxy, i, axis=0)
                if detections.mask is not None:
                    detections.mask = np.delete(detections.mask, i, axis=0)
                if detections.confidence is not None:
                    detections.confidence = np.delete(detections.confidence, i)
                if detections.class_id is not None:
                    detections.class_id = np.delete(detections.class_id, i)
                if detections.tracker_id is not None:
                    detections.tracker_id = np.delete(detections.tracker_id, i)
            i -= 1
            
        return detections
    
    @torch.no_grad()
    def segment(self, image: VisualObservation, **kwargs) -> Tuple[np.ndarray, List[str], np.ndarray]:
        classes = kwargs.get("classes", [])  # 文本提示类别列表，例如 ["chair","table",...]
        box_annotator = sv.BoxAnnotator() # 画框用
        mask_annotator = sv.MaskAnnotator() # 画 mask 用
        labels = []  # 每个检测对应的可视化 label（含置信度）
        # t1 = time.time()
        detections = self.grounding_dino_model.predict_with_classes(  # GroundingDINO：用文本类别做检测
            image=image,
            classes=classes,  # 类别文本列表（作为 prompt）
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        # t2 = time.time()
        detections = self._process_detections(detections)
        for _, _, confidence, class_id, _ in detections:   # supervision.Detections 的迭代会 yield (xyxy, mask, conf, class_id, tracker_id)
            if class_id is not None:
                labels.append(f"{classes[class_id]} {confidence:0.2f}")   # 用 class_id 索引回类别名
            else:
                labels.append(f"unknow {confidence:0.2f}")  # 没有 class_id 的兜底显示
        # t3 = time.time()
        detections.mask = self._segment(  # SAM：对每个检测框生成实例 mask
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy  # 检测框数组 shape: (N, 4)
        )
        # t4 = time.time()
        # print("grounding dino: ", t2 - t1)
        # print("process detections: ", t3 - t2)
        # print("sam: ", t4 - t3)
        # annotated_image.shape=(h,w,3)
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections) # 先画 mask（覆盖在原图上）
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)  # 再画框+label
        
        # detectins.mask.shape=[num_detected_classes, h, w]
        # attention: sometimes the model can't detect all classes, so num_detected_classes <= len(classes)
        return (detections.mask.astype(np.float32), labels, annotated_image, detections)
    

class BatchWrapper:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of input images.
    """
    def __init__(self, model) -> None:
        self.model = model
    
    def __call__(self, images: List[VisualObservation]) -> List:
        return [self.model(image) for image in images]