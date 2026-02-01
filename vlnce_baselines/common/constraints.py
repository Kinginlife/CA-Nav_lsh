import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from typing import List
import supervision as sv
from habitat import Config
from collections import Sequence
from vlnce_baselines.utils.map_utils import *
from lavis.models import load_model_and_preprocess
from transformers import AutoTokenizer, BertTokenizer
from vlnce_baselines.utils.constant import direction_mapping


class ConstraintsMonitor(nn.Module):
    def __init__(self, config: Config, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.resolution = config.MAP.MAP_RESOLUTION
        self.turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE
        self.device = device
        self._load_from_disk()
        
    def _create_model(self):
        self.model, vis_processors, text_processors = \
            load_model_and_preprocess("blip_vqa", model_type="vqav2", device=self.device, is_eval=True)
        self.vis_processors = vis_processors["eval"]
        self.text_processors = text_processors["eval"]
    
    def _load_from_disk(self):
        self.model = torch.load(self.config.VQA_MODEL_DIR, map_location='cpu').to(self.device)
        self._ensure_gelu_has_approximate(self.model)
        self._ensure_vqa_tokenizer(self.model)
        self.vis_processors = torch.load(self.config.VQA_VIS_PROCESSORS_DIR)["eval"]
        self.text_processors = torch.load(self.config.VQA_TEXT_PROCESSORS_DIR)["eval"]

    @staticmethod
    def _ensure_gelu_has_approximate(model: nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, nn.GELU) and not hasattr(m, "approximate"):
                m.approximate = "none"

    @staticmethod
    def _ensure_vqa_tokenizer(model: nn.Module) -> None:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "added_tokens_encoder"):
            tok = None
            try:
                tok = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
            except Exception:
                try:
                    tok = AutoTokenizer.from_pretrained(
                        "bert-base-uncased", use_fast=False, local_files_only=True
                    )
                except Exception:
                    tok = None
            if tok is not None:
                if getattr(tok, "pad_token", None) is None:
                    try:
                        tok.pad_token = "[PAD]"
                    except Exception:
                        pass
                if getattr(tok, "bos_token", None) is None:
                    try:
                        tok.bos_token = tok.cls_token
                    except Exception:
                        pass
                if getattr(tok, "bos_token_id", None) is None:
                    try:
                        tok.bos_token_id = tok.cls_token_id
                    except Exception:
                        pass
                try:
                    model.tokenizer = tok
                except Exception:
                    pass
                tokenizer = tok

        if tokenizer is None:
            return
        if not hasattr(tokenizer, "unique_no_split_tokens"):
            tokenizer.unique_no_split_tokens = []
        if not hasattr(tokenizer, "added_tokens_encoder"):
            tokenizer.added_tokens_encoder = {}
        if not hasattr(tokenizer, "added_tokens_decoder"):
            tokenizer.added_tokens_decoder = {}
        if not hasattr(tokenizer, "_added_tokens_encoder"):
            tokenizer._added_tokens_encoder = getattr(tokenizer, "added_tokens_encoder", {})
        if not hasattr(tokenizer, "_added_tokens_decoder"):
            tokenizer._added_tokens_decoder = getattr(tokenizer, "added_tokens_decoder", {})
        if not hasattr(tokenizer, "enc_token_id"):
            tokenizer.enc_token_id = getattr(tokenizer, "bos_token_id", None)
            if tokenizer.enc_token_id is None:
                tokenizer.enc_token_id = getattr(tokenizer, "cls_token_id", None)
        
    def location_constraint(self, obs: np.ndarray, scene: str):
        """ 
        use VQA to check scene type
        """
        image = Image.fromarray(obs['rgb'].astype(np.uint8))
        question = f"Are you in the {scene}"
        image = self.vis_processors(image).unsqueeze(0).to(self.device)
        question = self.text_processors(question)
        samples = {"image": image, "text_input": question}
        answer_candidates = ["yes", "no"]
        answer = self.model.predict_answers(samples, answer_list=answer_candidates, inference_method="rank")[0]
        if answer == "yes":
            return True
        else:
            return False
    
    def object_constraint(self, current_detection: sv.Detections, object: str, classes: List):
        """ 
        use grounded-sam's detections to check object
        """
        class_ids = current_detection.class_id
        class_names = [classes[i] for i in class_ids]
        if object in class_names:
            return True
        else:
            return False
    
    def direction_constraint(self, current_pose: Sequence, last_pose: Sequence, object):
        """ 
        check by geometric relation
        """
        heading = -1 * last_pose[-1]
        current_position, _ = get_agent_position(current_pose, self.resolution)
        last_position, _ = get_agent_position(last_pose, self.resolution)
        position_vector = current_position - last_position
        displacement = np.linalg.norm(position_vector)
        heading_vector = angle_to_vector(heading)
        rotation_matrix = np.array([[0, -1], 
                                [1, 0]])
        heading_vector = np.dot(rotation_matrix, heading_vector)
        if np.array_equal(position_vector, np.array([0., 0.])):
            return False
        degrees, direction = angle_and_direction(heading_vector, position_vector, self.turn_angle + 1)
        if degrees >= 120:
            movement = "backward"
        elif degrees == 0 or degrees == 180 or direction == 1:
            movement = "forward"
        else:
            if direction == 2:
                movement = "left"
            elif direction == 3:
                movement = "right"
        object_direction = direction_mapping.get(object, "ambiguous direction")
        if object_direction == "ambiguous direction":
            print("!Won't check ambiguous direction!")
            return True
        elif movement == object_direction and displacement >= 0.5 * 100 / self.resolution:
            return True
        else:
            return False
    
    def forward(self, 
                constraints: List, obs: np.ndarray, 
                detection: sv.Detections, classes: List,
                current_pose: Sequence, last_pose: Sequence):
        res  = []
        for item in constraints:
            constraint_type, constraint_object = item[:2]
            constraint_type = constraint_type.lower().strip()
            if constraint_type == "location constraint":
                check = self.location_constraint(obs, constraint_object)
                res.append(check)
            elif constraint_type == "object constraint":
                check = self.object_constraint(detection, constraint_object, classes)
                res.append(check)
            elif constraint_type == "direction constraint":
                check = self.direction_constraint(current_pose, last_pose, constraint_object)
                res.append(check)
        
        return res