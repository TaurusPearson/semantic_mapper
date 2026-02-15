#!/usr/bin/env python3

from typing import Dict, Any, Tuple, Generator, List
import numpy as np
import torch
import tqdm
import os
import yaml
import traceback
# 1. Import your existing utils
import segment_utils
from replica_text_encoder import load_replica_text_embeddings
# 2. Strict Hugging Face Imports (for SAM 3)
try:
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
except ImportError:
    print("Error: 'transformers' not installed or outdated.")
    raise

# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

class Log:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def track(msg): print(f"{Log.CYAN}[TRACK]{Log.END} {msg}", flush=True)
    @staticmethod
    def fuse(msg):  print(f"{Log.YELLOW}[FUSION]{Log.END} {msg}", flush=True)
    @staticmethod
    def cls(msg):   print(f"{Log.GREEN}[CLASS]{Log.END} {msg}", flush=True)

# -------------------------------------------------------------------------
# FUSION & SIMILARITY UTILS (From HOV-SG / ConceptFusion)
# -------------------------------------------------------------------------

def siglip_cosine_similarity(txt_embeds: torch.Tensor, img_embed: torch.Tensor, logit_scale: torch.Tensor, logit_bias: float) -> torch.Tensor:
    """
    Computes SigLIP-specific similarity probabilities.
    Crucial for SigLIP models to avoid "drift" caused by raw cosine values.
    """
    # Normalize features if not already done
    img_embed = img_embed / (img_embed.norm(dim=-1, keepdim=True) + 1e-6)
    txt_embeds = txt_embeds / (txt_embeds.norm(dim=-1, keepdim=True) + 1e-6)
    
    p = txt_embeds.to(img_embed.dtype) 
    # SigLIP Formula: sigmoid( (img @ txt.T) * exp(scale) + bias )
    logits = torch.mm(img_embed, p.T) * logit_scale.exp() + logit_bias 
    output = torch.sigmoid(logits)
    return output

def clip_cosine_similarity(txt_embeds: torch.Tensor, img_embed: torch.Tensor) -> torch.Tensor:
    """Standard CLIP similarity (for OpenAI CLIP)"""
    img_embed = img_embed / (img_embed.norm(dim=-1, keepdim=True) + 1e-6)
    txt_embeds = txt_embeds / (txt_embeds.norm(dim=-1, keepdim=True) + 1e-6)
    
    p = txt_embeds.to(img_embed.dtype)
    output = torch.mm(img_embed, p.T)
    return output

import os
import torch
import numpy as np
import tqdm
from typing import Dict, Any, Tuple, List

# Assuming segment_utils and other imports exist in your environment
# import segment_utils 

class MaskGenerator:
    def __init__(self, config: Dict[str, Any], scene_name: str = None, device = "cuda", class_names: List[str] = None) -> None:
        self.precomputed = config["precomputed"]
        self.config = config
        self.model_type = config.get("model_type", "sam3") # Options: "sam3", "mobile_sam", "sam3_semantic"
        self.class_names = class_names  # For text-prompted segmentation

        if scene_name:
            self.masks_path = os.path.join(config["masks_base_path"], scene_name)
        else:
            self.masks_path = ""

        # Configs for segment_utils (NMS)
        self.nms_iou_th = config.get("nms_iou_th", 0.8)
        self.nms_score_th = config.get("nms_score_th", 0.7)
        self.nms_inner_th = config.get("nms_inner_th", 0.5)

        self.device = device
        
        # We only load the model if we are NOT using precomputed masks
        if not self.precomputed or not os.path.exists(self.masks_path):
            self.load_mask_generator()

    def load_mask_generator(self) -> None:
        if self.model_type == "mobile_sam":
            self._load_mobile_sam()
        elif self.model_type == "sam3_semantic":
            self._load_sam3_semantic()
        else:
            self._load_sam3_hf()

    def _load_mobile_sam(self) -> None:
        print("Loading Mobile SAM (ViT-Tiny)...")
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        
        # Ensure you have the path in your config, or hardcode default
        checkpoint_path = self.config.get("mobile_sam_weights", "weights/mobile_sam.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"MobileSAM weights not found at {checkpoint_path}")

        mobile_sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        mobile_sam.to(device=self.device)
        mobile_sam.eval()

        # We configure the automatic generator to match your grid logic roughly
        # Adjust points_per_side to match the density of your 'step=64' logic
        self.mobile_generator = SamAutomaticMaskGenerator(
            model=mobile_sam,
            points_per_side=24,  # 24x24=576 points (balanced)
            pred_iou_thresh=0.80,
            stability_score_thresh=0.85,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )
        print("Mobile SAM Loaded.")

    def _load_sam3_hf(self) -> None:
        # Your existing Loading Logic
        print("Loading SAM 3 Tracker (Hugging Face) in Single-Frame Mode...")
        # Assuming imports exist for Sam3TrackerVideoModel/Processor
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor 
        
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        self.model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", torch_dtype=self.dtype).to(self.device)
        self.processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
        print("SAM 3 Loaded.")

    def _load_sam3_semantic(self) -> None:
        """Load SAM3 with text-prompt capability via Ultralytics."""
        print("Loading SAM 3 Semantic Predictor (Ultralytics) with text prompts...")
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError:
            raise ImportError("Install ultralytics>=8.3.237: pip install ultralytics --upgrade")
        
        weights_path = self.config.get("sam3_weights", "weights/sam3.pt")
        if not os.path.exists(weights_path):
            print(f"[Warning] SAM3 weights not found at {weights_path}. Will attempt download.")
        
        overrides = {
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "model": weights_path,
            "half": True,  # FP16 for speed
            "verbose": False,
        }
        self.semantic_predictor = SAM3SemanticPredictor(overrides=overrides)
        print(f"SAM 3 Semantic Loaded. Will query {len(self.class_names) if self.class_names else 0} classes.")

    def _generate_grid_points(self, h, w, step=64):
        xs = np.arange(step // 2, w, step)
        ys = np.arange(step // 2, h, step)
        grid_x, grid_y = np.meshgrid(xs, ys)
        return np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)

    def get_masks(self, image: np.ndarray, frame_id: int = None):
        """
        SOTA-Style API: Random access mask generation.
        """
        if self.precomputed and frame_id is not None:
            seg_map, binary_maps = self._load_masks(frame_id)
            if seg_map.size == 0:
                 seg_map, binary_maps = self.segment(image)
        else:
            seg_map, binary_maps = self.segment(image)

        return torch.from_numpy(seg_map).to(self.device), torch.from_numpy(binary_maps).to(self.device)

    def segment(self, image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates masks for a SINGLE frame.
        """
        if self.model_type == "mobile_sam":
            formatted_masks = self._segment_mobile(image_np)
        elif self.model_type == "sam3_semantic":
            return self._segment_sam3_semantic(image_np)  # Returns directly with class labels
        else:
            formatted_masks = self._segment_sam3(image_np)

        # Shared Post-Processing (NMS and Mapping)
        # ----------------------------------------
        if not formatted_masks:
            return np.array([]), np.array([])

        # Use custom segment_utils for consistent NMS behavior
        masks_filtered, = segment_utils.masks_update(
            formatted_masks,
            iou_thr=self.nms_iou_th,
            score_thr=self.nms_score_th,
            inner_thr=self.nms_inner_th
        )
        
        masks_filtered = sorted(masks_filtered, key=lambda x: x['area'], reverse=True)
        
        h, w = image_np.shape[:2]
        seg_map = np.full((h, w), -1, dtype=np.int32)
        binary_stack = []
        
        # Store SAM confidences for each mask (predicted_iou * stability_score)
        self.sam_confidences = []

        for i, m_dict in enumerate(masks_filtered):
            mask = m_dict['segmentation']
            seg_map[mask] = i
            binary_stack.append(mask)
            
            # Compute and store SAM confidence
            pred_iou = m_dict.get('predicted_iou', 1.0)
            stability = m_dict.get('stability_score', 1.0)
            self.sam_confidences.append(pred_iou * stability)

        binary_maps = np.array(binary_stack) if binary_stack else np.array([])
        
        return seg_map, binary_maps

    def _segment_mobile(self, image_np: np.ndarray) -> List[Dict]:
        """
        Mobile SAM specific segmentation logic
        """
        # MobileSAM generator expects uint8 0-255 image
        raw_masks = self.mobile_generator.generate(image_np)
        
        formatted_masks = []
        for ann in raw_masks:
            formatted_masks.append({
                'segmentation': ann['segmentation'], # Already boolean
                'area': ann['area'],
                'bbox': ann['bbox'], 
                'predicted_iou': ann.get('predicted_iou', 1.0), 
                'stability_score': ann.get('stability_score', 1.0),
                'point_coords': ann.get('point_coords', [[0,0]]),
                'original_id': 0 
            })
        return formatted_masks

    def _segment_sam3(self, image_np: np.ndarray) -> List[Dict]:
        """
        Your existing SAM 3 Video Logic
        """
        h, w = image_np.shape[:2]

        # 1. Init Session
        inference_session = self.processor.init_video_session(
            video=[image_np],
            inference_device=self.device,
            dtype=self.dtype,
        )

        # 2. Generate Grid Points
        grid_points = self._generate_grid_points(h, w, step=64) 
        
        input_points = grid_points.reshape(1, -1, 1, 2).tolist()
        input_labels = np.ones((1, len(grid_points), 1), dtype=int).tolist()
        obj_ids = list(range(len(grid_points)))

        # 3. Prompt
        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=0,
            obj_ids=obj_ids,
            input_points=input_points,
            input_labels=input_labels,
        )

        # 4. Inference
        formatted_masks = []
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session,
            start_frame_idx=0
        ):
            processed_masks = self.processor.post_process_masks(
                [model_outputs.pred_masks], original_sizes=[[h, w]], binarize=True
            )[0]
            masks_np = processed_masks.squeeze(1).cpu().numpy() # [N, H, W]

            for idx, mask in enumerate(masks_np):
                m_bool = mask.astype(bool)
                if m_bool.sum() > 0:
                    formatted_masks.append({
                        'segmentation': m_bool,
                        'area': m_bool.sum(),
                        'bbox': [0,0,0,0], 
                        'predicted_iou': 1.0, 
                        'stability_score': 1.0,
                        'point_coords': [[0,0]],
                        'original_id': idx 
                    })
            break

        del inference_session
        return formatted_masks

    def _segment_sam3_semantic(self, image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        SAM3 Semantic: Text-prompted segmentation.
        Returns seg_map where pixel values = class indices (not instance IDs).
        """
        if self.class_names is None or len(self.class_names) == 0:
            raise ValueError("class_names must be provided for sam3_semantic mode")
        
        h, w = image_np.shape[:2]
        
        # Set image once
        self.semantic_predictor.set_image(image_np)
        
        # Query with all class names
        results = self.semantic_predictor(text=self.class_names)
        
        # Build seg_map: each pixel gets the class index
        seg_map = np.full((h, w), -1, dtype=np.int32)
        binary_stack = []
        class_indices = []
        
        if results and len(results) > 0:
            result = results[0]  # Single image result
            
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()  # [N, H, W]
                
                # Each mask corresponds to a detected instance
                # We need to map back to class indices
                if hasattr(result, 'boxes') and result.boxes is not None:
                    # boxes.cls contains class indices
                    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                else:
                    cls_ids = np.zeros(len(masks), dtype=int)
                
                # Sort by area (largest first) so smaller objects overlay larger ones
                areas = [m.sum() for m in masks]
                sorted_indices = np.argsort(areas)[::-1]
                
                for idx in sorted_indices:
                    mask = masks[idx].astype(bool)
                    class_idx = cls_ids[idx] if idx < len(cls_ids) else 0
                    
                    if mask.sum() > 0:
                        seg_map[mask] = len(binary_stack)
                        binary_stack.append(mask)
                        class_indices.append(class_idx)
        
        binary_maps = np.array(binary_stack) if binary_stack else np.array([])
        
        # Store class indices for later use (semantic labels)
        self._last_class_indices = class_indices
        
        # DEBUG: Log what SAM3 detected
        if len(class_indices) > 0:
            unique_classes = set(class_indices)
            class_names_detected = [self.class_names[i] if i < len(self.class_names) else f"idx{i}" for i in unique_classes]
            print(f"[SAM3-Sem] Frame detected {len(binary_stack)} masks, classes: {class_names_detected}")
        
        return seg_map, binary_maps

    def get_semantic_class_indices(self) -> List[int]:
        """Get class indices from last semantic segmentation."""
        return getattr(self, '_last_class_indices', [])

    def get_sam_confidences(self) -> List[float]:
        """Get SAM confidence (predicted_iou * stability_score) per mask from last segmentation."""
        return getattr(self, 'sam_confidences', [])

    def _save_masks(self, seg_map: np.ndarray, binary_maps: np.ndarray, frame_id: int) -> None:
        map_path = os.path.join(self.masks_path, f"{frame_id:04d}_seg_map_default")
        np.save(map_path, seg_map)
        bmap_path = os.path.join(self.masks_path, f"{frame_id:04d}_bmap_default")
        np.save(bmap_path, binary_maps)

    def _load_masks(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        map_path = os.path.join(self.masks_path, f"{frame_id:04d}_seg_map_default.npy")
        if os.path.exists(map_path):
            seg_map = np.load(map_path)
            binary_path = os.path.join(self.masks_path, f"{frame_id:04d}_bmap_default.npy")
            if os.path.exists(binary_path):
                binary_maps = np.load(binary_path)
            else:
                max_id = seg_map.max()
                if max_id >= 0:
                    binary_maps = np.zeros((max_id + 1, *seg_map.shape), dtype=bool)
                    for i in np.unique(seg_map):
                        if i >= 0: binary_maps[i] = (seg_map == i)
                else:
                    binary_maps = np.array([])
            return seg_map, binary_maps
        return np.array([]), np.array([])

    def to(self, device: str) -> None:
        self.device = device
        # Handle moving appropriate model
        if hasattr(self, 'model'): 
            if isinstance(self.model, torch.nn.Module):
                self.model.to(device)
        if hasattr(self, 'mobile_generator'):
             self.mobile_generator.predictor.model.to(device)

    def precompute(self, dataset: torch.utils.data.Dataset, segment_every: int = 1) -> None:
        print(f"Precomputing segmentation masks (Mode: {self.model_type}).")
        os.makedirs(self.masks_path, exist_ok=True)
        
        for i in tqdm.tqdm(range(len(dataset))):
            if i % segment_every == 0:
                image = dataset[i][1] 
                if os.path.exists(os.path.join(self.masks_path, f"{i:04d}_seg_map_default.npy")):
                    continue
                
                seg_map, binary_maps = self.segment(image)
                self._save_masks(seg_map, binary_maps, i)
        self.precomputed = True
"""
main.py

Unified 3D Semantic Tracking on Replica Dataset.
Refactored to integrate SLAMMapper backbone with robust geometric memory synchronization.
Includes 3D Proximity Voting to fix "Object Amnesia".
"""

import os
import cv2
import time
import json
import gc
import numpy as np
import torch
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple
from pathlib import Path
from instance3d import Instance3D
import geometry_utils

# Hugging Face
from transformers import pipeline

# --- Custom Modules ---
try:
    from unified_semantic_backbone import load_semantic_backbone
    from replica_text_encoder import load_replica_text_embeddings
except ImportError:
    print("[Warning] Custom modules 'unified_semantic_backbone' or 'replica_text_encoder' not found.")
    def load_semantic_backbone(*args, **kwargs): return None
    def load_replica_text_embeddings(*args, **kwargs): return None

from mapper import SLAMMapper
import eval_utils
import instance_utils
from segment_utils import CLIPGenerator
from async_components import AsyncCLIPProcessor, AsyncMapper, PTAMController, MultiprocessingCLIPProcessor
from replica_prompts import IMAGENET_TEMPLATES
from llm_integration import initialize_llm_prompts, run_filtered_confusion_analysis, save_llm_metrics, DEFAULT_LLM_CONFIG
from llm_prompt_generator import select_prompt_for_sam3

# ============================================================
# SECTION 1: PROMPTS & CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_SEMANTIC_CACHE = {}

# ============================================================
# CONFIGURATION BLOCK
# ============================================================
SEMANTIC_CONFIGS = [
    {
        "name": "siglip",
        "kind": "dual_encoder",
        "model_id": "google/siglip-so400m-patch14-224",
    },
    {
        "name": "siglip2",
        "kind": "dual_encoder",
        "model_id": "timm/ViT-SO400M-14-SigLIP2-384",
    },
]

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
import os
import cv2
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        
        # --- 1. GEOMETRY SETUP ---
        # Load raw parameters first
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        
        # SOTA Logic: Handle Crop *Before* Resize
        # Many datasets require cropping edges (removing artifacts) before downscaling
        self.crop_edge = dataset_config.get("crop_edge", 0)
        if self.crop_edge > 0:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        # Apply Resize Ratio
        resize_ratio = dataset_config.get("resize_ratio", 1.0)
        self.height = int(self.height * resize_ratio)
        self.width = int(self.width * resize_ratio)
        self.fx *= resize_ratio
        self.fy *= resize_ratio
        self.cx *= resize_ratio
        self.cy *= resize_ratio

        # Final Intrinsics Matrix
        self.K = np.array([
            [self.fx, 0, self.cx], 
            [0, self.fy, self.cy], 
            [0, 0, 1]
        ])

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(dataset_config['distortion']) if 'distortion' in dataset_config else None
        
        self.color_paths = []
        self.depth_paths = []

    def __len__(self):
        if self.frame_limit < 0:
            return len(self.color_paths)
        return min(int(self.frame_limit), len(self.color_paths))


class Replica(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        
        # --- 1. FILE LOADING ---
        # Replica/NICE-SLAM directory structure
        img_dir = self.dataset_path / "results"
        if not img_dir.exists(): img_dir = self.dataset_path # Fallback
            
        self.color_paths = sorted(list(img_dir.glob("frame*.jpg")))
        self.depth_paths = sorted(list(img_dir.glob("depth*.png")))
        
        if len(self.color_paths) == 0:
            raise ValueError(f"No frames found at {img_dir}")

        # Load Poses
        traj_path = self.dataset_path / "traj.txt"
        if traj_path.exists():
            self.load_poses(traj_path)
            print(f"[Loader] Loaded {len(self.color_paths)} frames.")
        else:
            print(f"[Loader] Warning: Poses not found at {traj_path}")

        # --- 2. EVALUATION CONFIG (The SOTA Fix) ---
        # We prioritize eval_info.yaml to get the reduced class list
        self.eval_config = {}
        self.class_names = []
        self._load_eval_config()

        # If eval_info didn't populate classes, fallback to info_semantic.json or raw defaults
        if not self.class_names:
            self._load_legacy_semantics()

        print(f"[Loader] Semantic Classes set to: {len(self.class_names)} labels.")
        
        # --- 3. GT FILE LOCATOR ---
        self.gt_vtx = None
        self.gt_lbl = None
        self._locate_gt_files()

    def _load_eval_config(self):
        """Loads the SOTA mapping config."""
        # Check specific paths
        candidates = [
            self.dataset_path / "eval_info.yaml",
            Path("eval_info.yaml"),
            Path(__file__).parent / "eval_info.yaml"
        ]
        
        eval_path = next((p for p in candidates if p.exists()), None)
        
        if eval_path:
            print(f"[Loader] Loading Eval Config from: {eval_path}")
            with open(eval_path, 'r') as f:
                self.eval_config = yaml.safe_load(f)
            
            # CRITICAL: Use the reduced list for the classifier
            if "class_names_reduced" in self.eval_config:
                self.class_names = self.eval_config["class_names_reduced"]
            
            self.map_to_reduced = self.eval_config.get("map_to_reduced", {})
            self.ignore_ids = self.eval_config.get("ignore", [])
            self.ignore_ids.extend(self.eval_config.get("background_reduced_ids", []))

    def _load_legacy_semantics(self):
        """Fallback for non-SOTA runs."""
        sem_path = self.dataset_path / "info_semantic.json"
        if sem_path.exists():
            print("[Loader] Falling back to info_semantic.json")
            with open(sem_path, 'r') as f:
                d = json.load(f)
                unique = set()
                if 'objects' in d:
                    for obj in d['objects']: unique.add(obj['class_name'])
                else:
                    for v in d.values(): unique.add(v['class'] if isinstance(v,dict) else v)
                self.class_names = sorted(list(unique))
        else:
            print("[Loader] No semantic info found. Using empty class list.")

    def _locate_gt_files(self):
        """Finds mesh and labels for evaluation."""
        # 1. Look for Mesh
        scene_name = self.dataset_path.name
        mesh_candidates = [
            self.dataset_path.parent / f"{scene_name}_mesh.ply",
            self.dataset_path / f"{scene_name}_mesh.ply",
            self.dataset_path / "mesh.ply"
        ]
        self.gt_vtx = str(next((p for p in mesh_candidates if p.exists()), ""))
        
        # 2. Look for Labels
        lbl_candidates = [
            self.dataset_path.parent / "semantic_gt" / f"{scene_name}.txt",
            self.dataset_path / f"{scene_name}.txt"
        ]
        self.gt_lbl = str(next((p for p in lbl_candidates if p.exists()), ""))

        if self.gt_vtx: print(f"[Loader] GT Mesh:   {Path(self.gt_vtx).name}")
        if self.gt_lbl: print(f"[Loader] GT Labels: {Path(self.gt_lbl).name}")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # No flips! Load exactly what is in the txt file.
            self.poses.append(c2w.astype(np.float32))

    def __getitem__(self, index):
        # Image
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.dataset_config.get("resize_ratio", 1.0) != 1.0:
            color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        
        # Depth
        depth_data = cv2.imread(str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        if self.dataset_config.get("resize_ratio", 1.0) != 1.0:
            depth_data = cv2.resize(depth_data, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale

        return index, color_data, depth_data, self.poses[index]

    def get_frame(self, idx):
        if idx >= len(self): return None
        _, rgb, depth, pose = self[idx]
        return rgb, depth, pose


class ScanNet(BaseDataset):
    """
    ScanNet dataset loader.
    Expects extracted structure from .sens file:
        scene_dir/
            color/*.jpg
            depth/*.png  
            pose/*.txt
    """
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        
        # --- 1. FILE LOADING ---
        self.color_paths = sorted(
            list((self.dataset_path / "color").glob("*.jpg")),
            key=lambda x: int(os.path.basename(x)[:-4])
        )
        self.depth_paths = sorted(
            list((self.dataset_path / "depth").glob("*.png")),
            key=lambda x: int(os.path.basename(x)[:-4])
        )
        
        if len(self.color_paths) == 0:
            raise ValueError(f"No frames found at {self.dataset_path / 'color'}. "
                           "Run: python scannet_utils.py <scene_path> to extract .sens file.")
        
        # Load poses
        self.load_poses(self.dataset_path / "pose")
        print(f"[ScanNet] Loaded {len(self.color_paths)} frames.", flush=True)
        
        # Depth threshold (filter far depth values)
        depth_th = dataset_config.get("depth_th", 0)
        self.depth_th = depth_th if depth_th > 0 else None
        
        # --- 2. EVALUATION CONFIG ---
        self.eval_config = {}
        self.class_names = []
        self._load_eval_config()
        
        if not self.class_names:
            print("[ScanNet] Warning: No class names loaded from eval config.", flush=True)
        
        print(f"[ScanNet] Semantic Classes: {len(self.class_names)} labels.", flush=True)
        
        # --- 3. GT FILE LOCATOR ---
        self.gt_vtx = None
        self.gt_lbl = None
        self._locate_gt_files()
    
    def _load_eval_config(self):
        """Load evaluation config from dataset-specific YAML file."""
        # First check if eval_config_path was passed in dataset_config
        eval_path = self.dataset_config.get("eval_config_path")
        
        if eval_path and Path(eval_path).exists():
            eval_path = Path(eval_path)
        else:
            # Fallback to searching for config files
            candidates = [
                self.dataset_path / "eval_info.yaml",
                Path("scannet200.yaml"),
                Path("scannet20.yaml"),
                Path(__file__).parent / "scannet200.yaml",
                Path(__file__).parent / "eval_info.yaml"
            ]
            eval_path = next((p for p in candidates if p.exists()), None)
        
        if eval_path:
            print(f"[ScanNet] Loading Eval Config from: {eval_path}", flush=True)
            with open(eval_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            # Check for scannet-specific config section
            dataset_type = full_config.get("dataset", "replica")
            
            if dataset_type in ["scannet", "scannet20", "scannet200", "scannetv2"]:
                self.eval_config = full_config
                all_class_names = full_config.get("class_names_reduced", [])
                self.map_to_reduced = self.eval_config.get("map_to_reduced", {})
                self.ignore_ids = self.eval_config.get("ignore", [])
                self.ignore_ids.extend(self.eval_config.get("background_reduced_ids", []))
                
                # Use per-scene filtered classes for SAM3 (for speed)
                # But ensure proper mapping to full taxonomy at evaluation time
                scene_classes = self._load_scene_classes()
                if scene_classes:
                    # Filter to only classes that exist in the full config
                    self.class_names = [c for c in scene_classes if c in all_class_names]
                    print(f"[ScanNet] Using per-scene classes: {len(self.class_names)} (filtered from {len(scene_classes)})", flush=True)
                    # Store full taxonomy for evaluation
                    self.full_class_names = all_class_names
                else:
                    self.class_names = all_class_names
                    self.full_class_names = all_class_names
                    print(f"[ScanNet] Using full ScanNet200 taxonomy: {len(self.class_names)} classes", flush=True)
            else:
                # Look for scannet section in config
                if "scannet" in full_config:
                    self.eval_config = full_config["scannet"]
                    self.class_names = self.eval_config.get("class_names_reduced", [])
                    self.map_to_reduced = self.eval_config.get("map_to_reduced", {})
                    self.ignore_ids = self.eval_config.get("ignore", [])
    
    def _load_scene_classes(self):
        """
        Load class names from scene's aggregation file using standardized taxonomy.
        
        Maps aggregation label IDs to class_names_reduced from scannet200.yaml
        to ensure SAM3 uses proper taxonomy names that match ground truth.
        """
        scene_name = self.dataset_path.name
        
        # Get full taxonomy from eval_config (already loaded in _load_eval_config)
        all_class_names = self.eval_config.get("class_names_reduced", [])
        if not all_class_names:
            return None
        
        # Build ID -> class_name mapping from scannet200.yaml
        valid_ids = self.eval_config.get("valid_class_ids", [])
        if len(valid_ids) != len(all_class_names):
            print(f"[ScanNet] Warning: valid_class_ids ({len(valid_ids)}) != class_names ({len(all_class_names)})", flush=True)
            return None
        
        id_to_class = {vid: cname for vid, cname in zip(valid_ids, all_class_names)}
        
        # ScanNet aggregation file contains object instances with label IDs
        agg_candidates = [
            self.dataset_path / f"{scene_name}.aggregation.json",
            self.dataset_path / f"{scene_name}_vh_clean.aggregation.json",
        ]
        
        agg_path = next((p for p in agg_candidates if p.exists()), None)
        if agg_path:
            try:
                with open(agg_path, 'r') as f:
                    agg_data = json.load(f)
                
                # Build reverse mapping: class_name -> label_id for quick lookup
                class_to_id = {cname: vid for vid, cname in id_to_class.items()}
                
                unique_class_names = set()
                if "segGroups" in agg_data:
                    for seg in agg_data["segGroups"]:
                        # Get raw label string from aggregation file
                        raw_label = seg.get("label", "").lower().strip()
                        
                        if not raw_label or raw_label == "unannotated":
                            continue
                        
                        # Try to find matching standardized class name
                        matched = False
                        
                        # 1. Exact match (case-insensitive)
                        for std_name in all_class_names:
                            if raw_label == std_name.lower():
                                unique_class_names.add(std_name)
                                matched = True
                                break
                        
                        if not matched:
                            # 2. Handle plural -> singular (e.g., "doors" -> "door")
                            if raw_label.endswith('s'):
                                singular = raw_label[:-1]
                                for std_name in all_class_names:
                                    if singular == std_name.lower():
                                        unique_class_names.add(std_name)
                                        matched = True
                                        break
                        
                        if not matched:
                            # 3. Handle compound words (e.g., "piano bench" -> "bench")
                            # Try matching the last word
                            words = raw_label.split()
                            if len(words) > 1:
                                for std_name in all_class_names:
                                    if words[-1] == std_name.lower():
                                        unique_class_names.add(std_name)
                                        matched = True
                                        break
                
                if unique_class_names:
                    scene_classes = sorted(list(unique_class_names))
                    print(f"[ScanNet] Loaded {len(scene_classes)} standardized classes from {agg_path.name}", flush=True)
                    return scene_classes
            except Exception as e:
                print(f"[ScanNet] Failed to load aggregation file: {e}", flush=True)
        
        return None
    
    def _locate_gt_files(self):
        """Find ground truth mesh and labels for evaluation."""
        scene_name = self.dataset_path.name
        
        # ScanNet GT mesh format
        mesh_candidates = [
            self.dataset_path / f"{scene_name}_vh_clean_2.labels.ply",
            self.dataset_path / f"{scene_name}_vh_clean_2.ply",
            self.dataset_path / f"{scene_name}_vh_clean.ply"
        ]
        self.gt_vtx = str(next((p for p in mesh_candidates if p.exists()), ""))
        
        if self.gt_vtx:
            print(f"[ScanNet] GT Mesh: {Path(self.gt_vtx).name}", flush=True)
    
    def load_poses(self, path):
        """Load camera poses from individual .txt files."""
        self.poses = []
        pose_paths = sorted(
            path.glob('*.txt'),
            key=lambda x: int(os.path.basename(x)[:-4])
        )
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                ls.append(list(map(float, line.split())))
            c2w = np.array(ls).reshape(4, 4).astype(np.float32)
            self.poses.append(c2w)
    
    def __getitem__(self, index):
        # Color image
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None and np.any(self.distortion != 0):
            color_data = cv2.undistort(color_data, self.K, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if self.dataset_config.get("resize_ratio", 1.0) != 1.0:
            color_data = cv2.resize(color_data, (self.width, self.height), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Depth image
        depth_data = cv2.imread(str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        
        # Resize depth to match color image dimensions (ScanNet has different resolutions)
        if depth_data.shape[:2] != color_data.shape[:2]:
            depth_data = cv2.resize(depth_data, (color_data.shape[1], color_data.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Apply depth threshold
        if self.depth_th is not None:
            depth_data[depth_data > self.depth_th] = 0
        
        # Crop edges if specified
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        
        return index, color_data, depth_data, self.poses[index]
    
    def get_frame(self, idx):
        if idx >= len(self): return None
        _, rgb, depth, pose = self[idx]
        return rgb, depth, pose


class SemanticClassifier:
    # We use the previous fix signature
    def __init__(self, sem_name, backbone, label_list, temperature_args: Tuple[torch.Tensor] | None = None):
        self.backbone = backbone
        self.labels = label_list
        self.device = backbone.device
        self.name = sem_name
        self.text_emb = None # Initialize 

        # Inside SemanticClassifier.__init__
        if hasattr(backbone, 'text_bank') and backbone.text_bank is not None:
            bank = backbone.text_bank
            
            if isinstance(bank, dict):
                emb_list = []
                for label in label_list:
                    if label in bank:
                        emb_list.append(bank[label].to(self.device))
                    else:
                        print(f"{Log.RED}[Error] Class '{label}' missing from text bank!{Log.END}")
                        
                        # --- Safe Dimension Access ---
                        # Try to get dim from backbone, otherwise guess based on name
                        dim = getattr(backbone, 'embed_dim', None)
                        if dim is None:
                            dim = 1152 if "siglip" in backbone.name else 768
                        
                        emb_list.append(torch.zeros(dim, device=self.device))
                        
                self.text_emb = torch.stack(emb_list, dim=0)
            
            elif isinstance(bank, torch.Tensor):
                self.text_emb = bank.to(self.device)
                
            # Ensure normalization (Crucial for classification calculation)
            if self.text_emb is not None:
                self.text_emb = self.text_emb / (self.text_emb.norm(dim=-1, keepdim=True) + 1e-6)
        
        # If no text_emb is found, raise a proper error or fallback
        if self.text_emb is None:
             raise ValueError(f"Backbone '{sem_name}' has no 'text_bank' loaded. Check main() setup.")
        
        # --- Continue with SigLIP Parameter Fix (Previous Step) ---
        self.siglip = (backbone.name in ["siglip", "siglip2"])

        if temperature_args and len(temperature_args) > 0:
            self.temperature = temperature_args[0]
            self.bias = temperature_args[1] if len(temperature_args) > 1 else None
        else:
            self.temperature = None
            self.bias = None

    @torch.no_grad()
    def _encode_images_batch(self, rgb_list):
        return self.backbone.encode_image_batch(rgb_list)
    def _similarity_logits(self, img_emb):
        """ Calculates similarity between image embedding and all text classes """
        img_emb = img_emb.to(self.device).float() 
        
        if img_emb.ndim == 1: img_emb = img_emb.unsqueeze(0)
        
        # self.text_emb is already loaded and normalized from text_bank
        sim = img_emb @ self.text_emb.T
        
        if self.siglip and self.temperature is not None:
            # We use the temperature (scale), but we DROP the bias for Softmax classification
            # The bias is for Sigmoid thresholds, not for relative Softmax ranking.
            t = self.temperature.to(self.device) if isinstance(self.temperature, torch.Tensor) else self.temperature
            
            # SigLIP temperature is usually log(scale), so we exp() it. 
            # If your loader already exp() it, check that. 
            # Based on your logs "Temp=4.7215", this is likely log(scale).
            # If t is ~4.7, exp(t) is ~112. 
            
            # Use dot product scaled by temperature, NO BIAS.
            sim = sim * t.exp() 

        else:
            # Fallback for standard CLIP (usually requires a fixed scale factor like 100.0)
            sim = sim * 100.0 
            
        return sim
    
    def classify_objects(self, objects_dict):
        """ 
        Stable Classification V2: Uses Voting History.
        Prevents 'Fridge' -> 'Wall' flickering.
        """
        results = {}
        if self.text_emb is None: return results

        # 1. Collect Valid Objects (that have features)
        valid_ids = []
        valid_feats = []
        
        for oid, obj in objects_dict.items():
            if obj.clip_feature is not None:
                valid_ids.append(oid)
                # Ensure feature is float and on correct device
                valid_feats.append(obj.clip_feature.float().to(self.device))

        if not valid_ids: return results

        # 2. Batch Classification (Fast)
        # Shape: [N_Objects, D_Dim]
        img_features = torch.stack(valid_feats)
        
        # Shape: [N_Objects, N_Classes]
        # Calculate raw similarity
        sims = self._similarity_logits(img_features)
        
        # Softmax to get probabilities (confidence scores)
        probs = torch.nn.functional.softmax(sims, dim=1)
        
        # Get top predictions
        best_scores, best_indices = torch.max(probs, dim=1)

        # 3. Update History & Retrieve Stable Label
        for i, oid in enumerate(valid_ids):
            obj = objects_dict[oid]
            
            # Get the instantaneous guess
            idx = best_indices[i].item()
            score = best_scores[i].item()
            
            current_label = "Unknown"
            if 0 <= idx < len(self.labels):
                current_label = self.labels[idx]

            # --- VOTE ---
            # We add this frame's opinion to the object's history
            # Weight by combined SAM + CLIP confidence if available
            if hasattr(obj, 'add_label_vote'):
                if hasattr(obj, 'get_combined_confidence'):
                    # Use SAM-weighted confidence for more accurate voting
                    combined_score = obj.get_combined_confidence(score)
                else:
                    combined_score = score
                obj.add_label_vote(current_label, combined_score)
                # Ask the object for the accumulated winner
                results[oid] = obj.get_stable_label()
            else:
                # Fallback if Instance3D wasn't updated yet
                results[oid] = current_label
                
        return results
    
    # --- MULTI-CROP LOGIC ---
    def _get_context_bbox(self, bbox, h_img, w_img, expansion=0.3):
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0
        cx, cy = x0 + w / 2, y0 + h / 2
        size = max(w, h) * (1 + expansion)
        nx0 = max(0, int(cx - size / 2))
        ny0 = max(0, int(cy - size / 2))
        nx1 = min(w_img, int(cx + size / 2))
        ny1 = min(h_img, int(cy + size / 2))
        if nx1 <= nx0: nx1 = nx0 + 1
        if ny1 <= ny0: ny1 = ny0 + 1
        return nx0, ny0, nx1, ny1
    
    @torch.no_grad()
    def classify_masks_multicrop_batched(self, frame_rgb, sam_masks, full_img_emb):
        H, W, _ = frame_rgb.shape
        img_area = H * W
        batch_crops = []
        crop_meta = []

        # 1. Prepare Crops
        for idx, mask_data in enumerate(sam_masks):
            if "segmentation" not in mask_data or "bbox" not in mask_data: continue
            seg = mask_data["segmentation"]
            if seg.sum() < 20: continue
            
            # --- Calculate Mask Ratio for later ---
            mask_ratio = seg.sum() / img_area
            
            bx, by, bw, bh = map(int, mask_data["bbox"])
            if bw <= 0 or bh <= 0: continue
            x0, y0, x1, y1 = max(0, bx), max(0, by), min(W, bx + bw), min(H, by + bh)
            
            cx0, cy0, cx1, cy1 = self._get_context_bbox((x0, y0, x1, y1), H, W, expansion=0.3)
            crop_rgb = frame_rgb[cy0:cy1, cx0:cx1].copy()
            if crop_rgb.size == 0: continue

            mask_crop = seg[cy0:cy1, cx0:cx1]
            masked_rgb = crop_rgb.copy()
            masked_rgb[~mask_crop] = 0

            # 1. The "Masked" View
            batch_crops.append(masked_rgb)
            crop_meta.append((idx, "masked", mask_ratio)) # Pass ratio in meta
            
            # 2. The "BBox" View
            batch_crops.append(crop_rgb)
            crop_meta.append((idx, "bbox", mask_ratio))

        if not batch_crops: return {}

        # 2. Batch Encode
        embs = self._encode_images_batch(batch_crops)
        if embs.ndim == 3: embs = embs.squeeze(1)

        per_mask_embs = defaultdict(dict)
        per_mask_ratios = {} # Store ratios
        
        cursor = 0
        for (mask_idx, role, ratio) in crop_meta:
            per_mask_embs[mask_idx][role] = embs[cursor]
            per_mask_ratios[mask_idx] = ratio
            cursor += 1

        mask_logits = {}
        e_global = full_img_emb.squeeze(0) 

        # 3. Fuse with Adaptive Weights
        for mask_idx in range(len(sam_masks)):
            if mask_idx not in per_mask_embs: continue
            roles = per_mask_embs[mask_idx]
            ratio = per_mask_ratios.get(mask_idx, 0.0)
            
            e_masked = roles.get("masked", e_global)
            e_bbox = roles.get("bbox", e_global)
            
            # Stack views: [Global, Masked, BBox]
            clips = torch.stack([e_global, e_masked, e_bbox], dim=0)
            clips = clips / (clips.norm(dim=-1, keepdim=True) + 1e-6)
            
            # --- ADAPTIVE WEIGHTING (Tuned via diagnostic ablation) ---
            # Diagnostic: pure_object (66.3%) > masked_only (63.6%) > bbox (48.3%) > global (41.7%)
            # Pure masked is best - use it exclusively
            if ratio < 0.30:
                # Small/Medium: 100% masked (diagnostic showed this is optimal)
                weights = torch.tensor([0.0, 1.0, 0.0], device=self.device).view(3, 1)
            else:
                # Large: Tiny bbox for boundary context, still mostly masked
                weights = torch.tensor([0.0, 0.95, 0.05], device=self.device).view(3, 1)
                            
            fused = (clips * weights).sum(dim=0)
            fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-6)

            sim = self._similarity_logits(fused).squeeze(0)
            
            best_idx = int(torch.argmax(sim).item())
            best_score = float(sim[best_idx].item())
            
            label = self.labels[best_idx] if 0 <= best_idx < len(self.labels) else "other"
            mask_logits[mask_idx] = (sim.cpu(), best_score, label)

        return mask_logits

class SemanticMapper:
    def __init__(self, config: Dict[str, Any], sam_model, clip_model, scene_name: str | None = None, cam_intrinsics: torch.Tensor | None = None, eval: bool = False, device = "cuda") -> None:
        if not eval:
            assert cam_intrinsics is not None, "Camera intrinsics required for reconstruction!"

        config["sam"]["multi_crop"] = False if config["clip"]["embed_type"] == "vanilla" else True
        self.cam_intrinsics = cam_intrinsics
        self.config = config
        self.device = device
        self.n_top_views = config["clip"].get("k_top_views", 8)  # Top-K heap size for decoupled DBSCAN selection
        Instance3D.n_top_kf = self.n_top_views

        self.object_centroids = {}
        self.clip_generator = clip_model
        self.mask_generator = sam_model

        self.keyframes = {
            "ins_descriptors": dict(),
            "frame_id": list(),
            "ins_maps": list(),
        }

        # --- SLIDING WINDOW HISTORY ---
        self.history_len = config.get("history_len", 5) 
        self.history = deque(maxlen=self.history_len)
        
        self.keyframes_queue = deque([])
        self.objects = {}
        self._time_cache = []

        self.next_ins_id = 0
        self.kf_id = 0

        self.th_centroid = config.get("th_centroid", 1.5)
        self.th_cossim = config.get("th_cossim", 0.75)  # Lowered from 0.81 for better fusion (OVO uses 0.8)
        self.th_points = config.get("th_points", 0.1)
        
        # --- NEW: Config for Structural Fusion (Path C) ---
        # Objects with >50k points are forced to check semantic similarity regardless of distance
        self.min_semantic_fusion_points = config.get("min_semantic_fusion_points", 50000) 

        # --- OVO-style: Allow geometric-only fusion when CLIP features pending ---
        self.allow_geometric_only = config.get("allow_geometric_only", True) 

        self.current_global_map = None 
        self.last_seen_centroids = {}
        
        # --- OVO Spec III.3: Async CLIP Queue ---
        self.async_clip = None  # Will be set by PTAMController or pipeline
        self.use_async_clip = config.get("async_clip", True)

    def profil(func):
        def wrapper(self, *args, **kwargs):
            if self.config.get("log", False):
                torch.cuda.synchronize()
                t0 = time.time()
                out = func(self, *args, **kwargs)
                torch.cuda.synchronize()
                self._time_cache.append(time.time()-t0)
                return out
            else:
                return func(self, *args, **kwargs)
        return wrapper
    
    def update_map(self, map_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], kfs=None, classifier=None) -> torch.Tensor | None:
        # 0. Clean the queue 
        if self.keyframes_queue:
            self._compute_semantic_info()
            
        points_3d, points_ids, points_ins_ids = map_data

        # 1. Clean deleted Keyframes
        if kfs is not None:
            self._cleanup_deleted_kfs(kfs)

        # 2. Filter Active Objects
        objects_list = []
        objects_to_del = []
        map_ins_ids = points_ins_ids.unique()
        
        for ins_id in list(self.objects.keys()):
            if ins_id in map_ins_ids:
                objects_list.append(self.objects[ins_id])
            else:
                objects_to_del.append(self.objects[ins_id])
        
        # 3. Precompute Point Clouds (Required for SOTA geometric check)
        obj_pcds = {}
        for instance in objects_list:
            obj_pcd = points_3d[points_ins_ids == instance.id]
            if len(obj_pcd) > 0:
                obj_pcds[instance.id] = [obj_pcd, obj_pcd.mean(dim=0)]
            else:
                obj_pcds[instance.id] = [torch.empty(0, 3, device=self.device), torch.zeros(3, device=self.device)]

        # 4. Fusion Loop
        objects = {}
        fused_objects = {}
        
        for i, instance1 in enumerate(objects_list):
            if instance1.id in fused_objects: continue
            
            for instance2 in objects_list[i+1:]:
                if instance2.id in fused_objects: continue
                
                # SOTA Geometric Check (OVO-style with optional geometric-only fusion)
                is_same = instance_utils.same_instance(
                    instance1, instance2, 
                    obj_pcds[instance1.id], obj_pcds[instance2.id], 
                    self.th_centroid, self.th_cossim, self.th_points,
                    allow_geometric_only=self.allow_geometric_only
                )
                
                if is_same:
                    # [LOGGING ADDED HERE]
                    Log.fuse(f"Fusing {instance2.id} into {instance1.id}")
                    
                    # Fuse using utils
                    instance1, points_ins_ids = instance_utils.fuse_instances(instance1, instance2, map_data)
                    fused_objects[instance2.id] = instance1.id
            
            objects[instance1.id] = instance1

        # 5. Cleanup Deleted Objects
        for oid in objects_to_del:
            if oid.id in self.objects: del self.objects[oid.id]

        # 6. Update Descriptors from Merged Objects
        for id2, id1 in fused_objects.items():
            if id2 in self.objects: del self.objects[id2]
            
            # Transfer descriptors from history (SOTA Logic)
            for kf in self.keyframes["frame_id"]:
                if kf == "Deleted": continue
                if kf in self.keyframes["ins_descriptors"] and id2 in self.keyframes["ins_descriptors"][kf]:
                    feat = self.keyframes["ins_descriptors"][kf].pop(id2)
                    if id1 not in self.keyframes["ins_descriptors"][kf]:
                        self.keyframes["ins_descriptors"][kf][id1] = feat

        self.objects = objects
        self.update_objects_clip(force_update=True)
        
        return points_ins_ids

    def _cleanup_deleted_kfs(self, kfs):
        """Helper to remove keyframes that SLAM deleted."""
        # This handles the sliding window logic if SLAM drops old frames
        if "frame_id" in self.keyframes:
            for i, kf in enumerate(self.keyframes["frame_id"]):
                if kf != "Deleted" and kf not in kfs:
                    if kf in self.keyframes["ins_descriptors"]:
                        del self.keyframes["ins_descriptors"][kf]
                    self.keyframes["frame_id"][i] = "Deleted"

    def _merge_instances_internals(self, target, source):
        """Helper to move data from Source object to Target object."""
        # 1. Merge Keyframes
        for kf in source.kfs_ids:
            target.add_keyframes(kf)
            
            # Move descriptors if they exist in the cache
            if kf in self.keyframes["ins_descriptors"]:
                frame_descs = self.keyframes["ins_descriptors"][kf]
                if source.id in frame_descs:
                    feat = frame_descs.pop(source.id)
                    # Only add to target if target didn't see this frame
                    if target.id not in frame_descs:
                        frame_descs[target.id] = feat
        
        # 2. Merge Top Views (The Step 2 Logic)
        for area, kf in source.top_kf:
            target.add_top_kf(kf, area)
            
        target.to_update = True

    def update_objects_clip(self, force_update: bool = False) -> None:
        """ Helper to update descriptors after fusion """
        for obj in self.objects.values():
            obj.update_clip(self.keyframes["ins_descriptors"], force_update=force_update)

    def track(self, frame_data, c2w, map_data, binary_maps_input):
        frame_id = frame_data[0]
        image = frame_data[1]
        depth = frame_data[2]

        points_3d, points_ids, points_ins_ids = map_data

        current_max_ins = points_ins_ids.max().item()
        if current_max_ins >= self.next_ins_id:
            self.next_ins_id = current_max_ins + 1

        updated_ins_ids = self.detect_and_track_objects(
            (frame_id, image, depth, []),
            map_data,
            c2w,
            binary_maps_input=binary_maps_input
        )
        return updated_ins_ids
    
    def _convert_binary_to_seg(self, binary_maps, shape):
        H, W = shape
        seg_map = torch.full((H, W), -1, dtype=torch.long, device=self.device)

        if binary_maps.shape[0] > 0:
            active_indices = torch.nonzero(binary_maps.flatten(1).sum(1) > 0).squeeze(1)
            areas = binary_maps[active_indices].sum(dim=(1,2))
            sorted_local_indices = torch.argsort(areas, descending=True)

            for local_idx in sorted_local_indices:
                real_id = active_indices[local_idx]
                seg_map[binary_maps[real_id]] = real_id

        return seg_map, binary_maps

    @profil
    def _get_masks(self, image: np.ndarray, frame_id: int):
        seg_map_t, batch_tensor, valid_idx = self.mask_generator.process_frame(image)

        if seg_map_t.numel() == 0:
            return torch.tensor([]), torch.tensor([])

        unique_ids = torch.unique(seg_map_t)
        unique_ids = unique_ids[unique_ids != -1]
        if len(unique_ids) == 0:
            return seg_map_t, torch.empty((0, *seg_map_t.shape), device=self.device, dtype=torch.bool)

        binary_maps = []
        for uid in unique_ids:
            binary_maps.append(seg_map_t == uid)
        binary_maps = torch.stack(binary_maps)

        return seg_map_t, binary_maps

    def get_debug_label(self, oid, classifier):
        """Helper to get a string label for an Object ID for debug prints."""
        if classifier is None: return "?"
        if oid not in self.objects: return "New"
        
        obj = self.objects[oid]
        
        # FORCE UPDATE: If feature is missing, try to compute it from history
        if obj.clip_feature is None:
            obj.update_clip(self.keyframes["ins_descriptors"])
        
        # If still None, it really has no data
        if obj.clip_feature is None: return "NoCLIP"
        
        # Ask classifier for the label
        try:
            res = classifier.classify_objects({oid: obj})
            return res.get(oid, "Unknown")
        except:
            return "Error"
    
    def detect_and_track_objects(self, frame_data, map_data, c2w, binary_maps_input=None, classifier=None, sam3_class_indices=None) -> torch.Tensor:
        """
        Refactored Step 1: Simplified Flow. 
        Removed the dependency on 'self.history' for tracking. 
        sam3_class_indices: List of class indices from SAM3 semantic mode (one per mask)
        """
        frame_id, image = frame_data[:2]
        
        # 1. Get Masks (SAM)
        if binary_maps_input is not None:
            seg_maps, binary_maps = self._convert_binary_to_seg(binary_maps_input, image.shape[:2])
        else:
            seg_maps, binary_maps = self._get_masks(image, frame_id)

        if len(seg_maps) == 0 or binary_maps.shape[0] == 0:
            print(f"[SemMapper] No masks found in frame {frame_id}")
            return map_data[2] # Return original points_ins_ids

        # 2. Match & Track (The Core Logic)
        # We pass map_data structure: (points_3d, points_ids, points_ins_ids)
        matched_ins_ids, binary_maps, n_matched_points, updated_points_ins_ids = self._match_and_track_instances(
            frame_data[1:], map_data, c2w, seg_maps, binary_maps, sam3_class_indices=sam3_class_indices
        )

        # 3. Queue Data for Semantic Processing (Lazy Feature Extraction)
        # We store the result to process CLIP later (or immediately if you prefer)
        current_data = [matched_ins_ids, binary_maps, image, self.kf_id]
        
        # Immediate processing for now to keep your pipeline working
        self._compute_semantic_info(data=current_data)
        
        # Update State
        self.kf_id += 1
        
        # Keep history for visualization/debugging
        self.history.appendleft({
            "binary_maps": binary_maps, 
            "ids": matched_ins_ids       
        })

        return updated_points_ins_ids

    @profil
    def _match_and_track_instances(self, frame_data, map_data, c2w, seg_map, binary_maps, classifier=None, sam3_class_indices=None):
        """
        Refactored Step 1: Geometric Projection Only.
        sam3_class_indices: List of class indices from SAM3 (one per mask index)
        """
        kf_id = self.kf_id
        image, depth = frame_data[:2]
        rgb_depth_ratio = frame_data[2] if len(frame_data) > 2 else []
        
        points_3d, points_ids, points_ins_ids = map_data
        depth = torch.from_numpy(depth).to(self.device)

        # 1. Frustum Culling (Filter points to only those visible in camera)
        camera_frustum_corners = geometry_utils.compute_camera_frustum_corners(depth, c2w, self.cam_intrinsics)
        frustum_mask = geometry_utils.compute_frustum_point_ids(points_3d, camera_frustum_corners, device=self.device)
        frustum_points_3d = points_3d[frustum_mask]

        if self.config.get("depth_filter", False):
            depth = geometry_utils.depth_filter(depth)

        # 2. Project 3D Points -> 2D Pixels
        matched_points_idxs, matches = geometry_utils.match_3d_points_to_2d_pixels(
            depth, 
            torch.linalg.inv(c2w), 
            frustum_points_3d, 
            self.cam_intrinsics, 
            self.config.get("match_distance_th", 0.05)
        )

        # --- Apply Scaling & Bounds Checking ---
        if len(rgb_depth_ratio) > 0:
            matches = matches + rgb_depth_ratio[2]
            matches = matches.float()
            matches[:, 1] = matches[:, 1] * rgb_depth_ratio[0]
            matches[:, 0] = matches[:, 0] * rgb_depth_ratio[1]
            matches = matches.long()
            
        # CRITICAL FIX: Filter pixels strictly within image bounds
        H, W = seg_map.shape
        valid_mask = (
            (matches[:, 0] >= 0) & (matches[:, 0] < W) &
            (matches[:, 1] >= 0) & (matches[:, 1] < H)
        )
        
        matched_points_idxs = matched_points_idxs[valid_mask]
        matches = matches[valid_mask]

        # 3. Sample the Segmentation Map at the projected pixels
        matched_seg_idxs = seg_map[matches[:, 1], matches[:, 0]]

        frustum_points_ids = points_ids[frustum_mask]
        frustum_points_ins_ids = points_ins_ids[frustum_mask]

        # 4. STRICT TRACKING (The SOTA Logic)
        updated_frustum_ins_ids, matched_ins_info = self._track_objects(
            frustum_points_ids,
            frustum_points_ins_ids,
            matched_points_idxs,
            matched_seg_idxs,
            seg_map,
            self.config.get("track_th", 100),
            kf_id,
            sam3_class_indices=sam3_class_indices
        )

        # 5. Fuse local masks that mapped to the same global ID
        matched_ins_ids, binary_maps = self._fuse_masks_with_same_ins_id(binary_maps, matched_ins_info, kf_id)

        # 6. Update the Global Map State
        updated_points_ins_ids = points_ins_ids.clone()
        updated_points_ins_ids[frustum_mask] = updated_frustum_ins_ids

        # Debug visualization hook
        if self.config.get("debug_info", False):
            ins_maps = torch.ones(image.shape[:2], dtype=torch.int, device=self.device) * -1
            for ins_id, matches_info in matched_ins_info.items():
                for map_idx, _ in matches_info:
                    ins_maps[binary_maps[map_idx]] = ins_id
            self.keyframes["ins_maps"].append(ins_maps.cpu().numpy())

        return matched_ins_ids, binary_maps, len(matched_points_idxs), updated_points_ins_ids
    
    @profil
    def _track_objects(self, points_ids: torch.Tensor, points_ins_ids: torch.Tensor, matched_points_idxs: torch.Tensor, matched_seg_idxs: torch.Tensor, seg_map: torch.Tensor, track_th: float, kf_id: int, sam_confidences: List[float] = None, sam3_class_indices: List[int] = None) -> tuple[torch.Tensor, Dict[int, List[Tuple[int, int]]]]:
        matched_ins_info = {}
        
        # Iterate over every unique 2D mask in the current frame
        for map_idx in range(seg_map.max() + 1):
            map_ins_id = -1
            
            # Get 3D points projected into this 2D mask
            map_points_indices = matched_points_idxs[matched_seg_idxs == map_idx]

            # Only process if we have enough 3D support
            if len(map_points_indices) > track_th:
                mask_area = (seg_map == map_idx).sum().item()
                
                # Check which points already have a Global Object ID (> -1)
                assigned_mask = points_ins_ids[map_points_indices] > -1
                unassigned_p_ids = points_ids[map_points_indices[~assigned_mask]].cpu().tolist()

                # Get SAM3's class for this mask (if available)
                current_mask_class = -1
                if sam3_class_indices and map_idx < len(sam3_class_indices):
                    current_mask_class = sam3_class_indices[map_idx]

                # --- CASE A: VOTING (SOTA Logic with Semantic Class Voting) ---
                # If enough points already have an ID, vote for the majority ID.
                # Add semantic class vote to enable majority voting across frames.
                vote_for_existing = False
                if assigned_mask.sum().item() > track_th:
                    mode_result = torch.mode(points_ins_ids[map_points_indices[assigned_mask]])
                    candidate_ins_id = mode_result.values.item()
                    
                    if candidate_ins_id in self.objects:
                        # Add semantic class vote (majority voting across frames)
                        if current_mask_class >= 0:
                            self.objects[candidate_ins_id].add_semantic_class_vote(current_mask_class)
                        
                        vote_for_existing = True
                        map_ins_id = candidate_ins_id
                        self.objects[map_ins_id].update(unassigned_p_ids, kf_id, mask_area)
                        if sam_confidences and map_idx < len(sam_confidences):
                            self.objects[map_ins_id].update_sam_confidence(
                                sam_confidences[map_idx], 1.0
                            )
                        if map_ins_id not in matched_ins_info: matched_ins_info[map_ins_id] = []
                        matched_ins_info[map_ins_id].append((map_idx, mask_area))

                # --- CASE B: DISCOVERY (New Object) ---
                # If mostly unassigned OR SAM3 class differs from existing object, create NEW object.
                if not vote_for_existing and len(unassigned_p_ids) > track_th:
                    map_ins_id = self.next_ins_id
                    self.next_ins_id += 1
                    
                    # Get SAM3 semantic class index if available
                    sem_class_idx = -1
                    if sam3_class_indices and map_idx < len(sam3_class_indices):
                        sem_class_idx = sam3_class_indices[map_idx]
                    
                    self.objects[map_ins_id] = Instance3D(
                        map_ins_id, 
                        kf_id=kf_id, 
                        points_ids=unassigned_p_ids, 
                        mask_area=mask_area,
                        semantic_class_idx=sem_class_idx
                    )
                    
                    # Initialize SAM confidence if available
                    if sam_confidences and map_idx < len(sam_confidences):
                        self.objects[map_ins_id].update_sam_confidence(
                            sam_confidences[map_idx], 1.0
                        )
                    
                    matched_ins_info[map_ins_id] = [(map_idx, mask_area)]
                    
                    # [LOGGING ADDED HERE] - Include class index for debugging
                    Log.track(f"Created New ID {map_ins_id} (Pts: {len(unassigned_p_ids)}, map_idx={map_idx}, class_idx={sem_class_idx})")

                # --- CASE C: ASSIGNMENT ---
                # Assign the determined ID to the previously unassigned points
                if map_ins_id > -1:
                    points_ins_ids[map_points_indices[~assigned_mask]] = map_ins_id

        return points_ins_ids, matched_ins_info
    
    def process_queue(self, classifier=None):
        if self.keyframes_queue:
            self._compute_semantic_info()

    @torch.no_grad()
    def query(self, queries: List[str], templates: List[str] = ['{}']) -> torch.Tensor:
        """SOTA Feature: Calculate similarity between text queries and 3D objects."""
        assert len(self.objects) > 0, "No 3D instances to query!"
        
        # --- Auto-detect dimension from CLIP generator or object features ---
        # SigLIP is 1152, CLIP is 512. Use generator's known dimension as fallback.
        feature_dim = getattr(self.clip_generator.backbone, 'embed_dim', 512)
        for obj in self.objects.values():
            if obj.clip_feature is not None:
                feature_dim = obj.clip_feature.shape[-1]
                break
        
        # Gather all object CLIP features using the detected dimension
        object_clips = torch.zeros((len(self.objects), feature_dim), device=self.device)
        
        for j, obj in enumerate(self.objects.values()):
            if obj.clip_feature is not None:
                object_clips[j] = obj.clip_feature.to(self.device)
        
        # Calculate similarity map
        return self.clip_generator.get_embed_txt_similarity(object_clips, queries, templates=templates)

    @torch.no_grad()
    def classify_instances(self, classes: List[str], th: float = 0):
        """SOTA Feature: Auto-label the scene based on a list of classes."""
        sim_map = self.query(classes)
        instances_classes = torch.argmax(sim_map, dim=1)
        max_conf = torch.gather(sim_map, -1, instances_classes[:, None]).squeeze()
        
        # Apply confidence threshold
        instances_classes[max_conf <= th] = -1
        max_conf[max_conf <= th] = 0
        
        return {"classes": instances_classes.cpu().numpy(), "conf": max_conf.cpu().numpy()}
        
    def _fuse_masks_with_same_ins_id(self, binary_maps: torch.Tensor, matched_ins_info: Dict[int, List[Tuple[int, int]]], kf_id: int):
        """
        Refactored Step 2A: Register 'Top Views' with Instance3D.
        """
        matched_ins_ids = []
        maps_idxs = []
        
        # matched_ins_info is {obj_id: [(mask_idx, area), ...]}
        for ins_id, data_list in matched_ins_info.items():
            
            # 1. Merge split masks (standard logic)
            first_map_idx = data_list[0][0]
            if len(data_list) > 1:
                for j in range(1, len(data_list)):
                    other_map_idx = data_list[j][0]
                    binary_maps[first_map_idx] = torch.logical_or(binary_maps[first_map_idx], binary_maps[other_map_idx])
            
            # 2. Get the final merged mask area
            mask_area = binary_maps[first_map_idx].sum().item()

            # 3. CRITICAL: Tell the object about this view!
            if ins_id in self.objects:
                self.objects[ins_id].add_top_kf(kf_id, mask_area)

            # 4. Decoupled approach: always keep ALL views for CLIP extraction.
            matched_ins_ids.append(ins_id)
            maps_idxs.append(first_map_idx)

        # Filter the binary_maps tensor to only include the ones we kept
        if len(maps_idxs) > 0:
            binary_maps = binary_maps[maps_idxs]
        else:
            binary_maps = torch.empty((0, *binary_maps.shape[1:]), device=self.device, dtype=torch.bool)

        return matched_ins_ids, binary_maps

    def _compute_semantic_info(self, data=None) -> None:
        if data is not None:
            matched_ins_ids, binary_maps, image, kf_id = data
        elif self.keyframes_queue:
            matched_ins_ids, binary_maps, image, kf_id = self.keyframes_queue.popleft()
        else:
            return

        if len(matched_ins_ids) > 0:
            # --- Decoupled approach: extract CLIP for ALL views ---
            # Top-K selection happens in update_clip() for DBSCAN only.

            # --- OVO Spec III.3: Async CLIP Processing ---
            if self.use_async_clip and self.async_clip is not None:
                # Non-blocking: Submit to background thread
                self.async_clip.submit(kf_id, image, binary_maps, matched_ins_ids)
            else:
                # Synchronous fallback
                clip_embeds = self._extract_clip(image, binary_maps).cpu()
                self._update_matched_objects_clip(clip_embeds, matched_ins_ids, kf_id)

    def collect_async_clip_results(self) -> int:
        """
        OVO Spec III.3: Collect completed async CLIP results.
        Call this periodically to integrate background-computed features.
        Returns: Number of results collected.
        """
        if self.async_clip is None:
            return 0
            
        results = self.async_clip.get_results()
        count = 0
        
        for kf_id, (clip_embeds, matched_ins_ids) in results.items():
            self._update_matched_objects_clip(clip_embeds, matched_ins_ids, kf_id)
            count += 1
            
        return count
    
    def set_async_clip_processor(self, async_clip):
        """Set the async CLIP processor (called by PTAMController)"""
        self.async_clip = async_clip
        
    def flush_async_clip(self, timeout: float = 30.0):
        """Wait for all pending async CLIP work to complete"""
        if self.async_clip is not None:
            self.async_clip.wait_for_completion(timeout)
            self.collect_async_clip_results()
    
    @profil
    def _extract_clip(self, image: torch.Tensor, binary_maps: torch.Tensor) -> List[Any]:
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose((2,0,1))).to(self.device).float()
        return self.clip_generator.extract_clip(image, binary_maps).cpu()

    @profil
    def _update_matched_objects_clip(self, clip_embeds: torch.Tensor, matched_ins_ids: List[int], kf_id: int) -> None:
        ins_embeds = dict()
        for i, ins_id in enumerate(matched_ins_ids):
            if ins_id != -1:
                ins_embeds[ins_id] = clip_embeds[i]

        self.keyframes["ins_descriptors"][kf_id] = ins_embeds
        for ins_id in matched_ins_ids:
            if ins_id in self.objects:
                self.objects[ins_id].update_clip(self.keyframes["ins_descriptors"])

    def cpu(self):
        if self.mask_generator: del self.mask_generator
        if self.clip_generator: del self.clip_generator
        torch.cuda.empty_cache()

import torch
import torch.multiprocessing as mp
import numpy as np
import time
import os
import gc
import cv2
import json
from pathlib import Path

# --- WORKER PROCESS: Handles Visualization on a separate Core ---
def viz_worker(queue, viz_colors, scene_name):
    """
    Standalone worker process. Consumes data from the pipeline and renders 
    the Tri-View without blocking the main SLAM loop.
    """
    print("[Viz Worker] Started.", flush=True)
    try:    
        while True:
            # Block until data is available
            payload = queue.get()
            
            # Poison Pill: Stop signal
            if payload is None: 
                break
                
            # Unpack data (Must be CPU/Numpy types, no Cuda Tensors here)
            rgb, frame_id, obj_class_dict, raw_seg_map, q_data, h_data = payload
            
            # --- YOUR ORIGINAL TRI-VIEW LOGIC ---
            img_1 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            img_2 = img_1.copy()
            img_3 = img_1.copy()

            # 1. Raw SAM
            if raw_seg_map is not None:
                unique_ids = np.unique(raw_seg_map)
                for uid in unique_ids:
                    if uid == -1: continue
                    mask = (raw_seg_map == uid)
                    np.random.seed(int(uid))
                    color = np.random.randint(50, 255, 3)
                    bgr = (int(color[0]), int(color[1]), int(color[2]))
                    img_1[mask] = (img_1[mask].astype(np.float32) * 0.5 + np.array(bgr) * 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_1, contours, -1, bgr, 1)
            cv2.putText(img_1, "1. Raw SAM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 2. Robust Queue
            if q_data is not None:
                q_ids, q_maps_np = q_data
                if len(q_maps_np) > 0:
                    sorted_indices = np.argsort([np.sum(m) for m in q_maps_np])[::-1]
                    for idx in sorted_indices:
                        mask = q_maps_np[idx].astype(bool)
                        if not np.any(mask): continue
                        obj_id = q_ids[idx]
                        if obj_id == -1: continue
                        
                        color = viz_colors[obj_id % 5000].tolist()
                        bgr = (int(color[2]), int(color[1]), int(color[0]))
                        img_2[mask] = (img_2[mask].astype(np.float32) * 0.5 + np.array(bgr) * 0.5).astype(np.uint8)
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img_2, contours, -1, bgr, 2)
                        
                        # Label
                        ys, xs = np.where(mask)
                        if len(xs) > 0:
                            cx, cy = int(np.mean(xs)), int(np.mean(ys))
                            cv2.putText(img_2, str(obj_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(img_2, "2. Robust Tracker", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 3. Fused History
            if h_data is not None:
                h_maps_np, h_ids = h_data
                if len(h_maps_np) > 0:
                    sorted_indices = np.argsort([np.sum(m) for m in h_maps_np])[::-1]
                    for idx in sorted_indices:
                        mask = h_maps_np[idx].astype(bool)
                        if not np.any(mask): continue
                        obj_id = h_ids[idx] if idx < len(h_ids) else -1
                        if obj_id == -1: continue

                        color = viz_colors[obj_id % 5000].tolist()
                        bgr = (int(color[2]), int(color[1]), int(color[0]))
                        label_text = f"{obj_id}: {obj_class_dict.get(obj_id, '...')}"

                        img_3[mask] = (img_3[mask].astype(np.float32) * 0.4 + np.array(bgr) * 0.6).astype(np.uint8)
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img_3, contours, -1, bgr, 2)
                        
                        M = cv2.moments(mask.astype(np.uint8))
                        if M["m00"] > 0:
                            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                            cv2.putText(img_3, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(img_3, "3. Fused History", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Combine
            h, w, c = img_1.shape
            separator = np.zeros((h, 5, c), dtype=np.uint8)
            combined = np.hstack([img_1, separator, img_2, separator, img_3])
            
            # Combine
            h, w, c = img_1.shape
            separator = np.zeros((h, 5, c), dtype=np.uint8)
            combined = np.hstack([img_1, separator, img_2, separator, img_3])
            
            # --- FIX: Strictly Divide Size by 2 ---
            new_w = int(combined.shape[1] // 1.2)
            new_h = int(combined.shape[0] // 1.2)
            combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # --------------------------------------

            # Keep window normal so you can resize manually if needed
            win_name = f"Pipeline - {scene_name}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, combined)
            cv2.waitKey(1)
            # Use WINDOW_NORMAL to allow manual resizing with mouse
            win_name = f"Pipeline - {scene_name}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, combined)
            cv2.waitKey(1)
            cv2.imshow(f"Pipeline - {scene_name}", combined)
            cv2.waitKey(1)
    except Exception as e:
        traceback.print_exc()
        print(f"[Viz Worker Error] {e}", flush=True)

class ReplicaFusionPipeline:
    def __init__(self, scene_name, config, loader, output_dir, device="cuda"):
        self.config = config
        self.device = device
        self.loader = loader
        self.scene_name = scene_name
        self.output_dir = output_dir  # <--- STORE THIS

        # --- FIX: Build mapping from filtered per-scene class indices to full taxonomy indices ---
        # This is needed for ScanNet200 where SAM3 uses a subset of classes per scene
        self.filtered_to_full_class_idx = {}
        if hasattr(loader, 'full_class_names') and hasattr(loader, 'class_names'):
            full_names = loader.full_class_names
            filtered_names = loader.class_names
            if len(filtered_names) < len(full_names):
                for filtered_idx, class_name in enumerate(filtered_names):
                    if class_name in full_names:
                        full_idx = full_names.index(class_name)
                        self.filtered_to_full_class_idx[filtered_idx] = full_idx
                print(f"[Init] Built filtered->full class mapping: {len(self.filtered_to_full_class_idx)} classes")

        print(f"{Log.HEADER}--- DEBUG CAMERA PARAMS ---{Log.END}")
        
        # --- FIX: Prioritize Loader's Calculated K ---
        # The Loader already handled crop/resize logic. Trust it.
        if hasattr(loader, 'K'):
            print("[Init] Using Intrinsics from Loader.K")
            self.intrinsics = torch.from_numpy(loader.K).float().to(device)
            # Back-fill scalars for logging
            fx, fy = loader.K[0,0], loader.K[1,1]
        elif hasattr(loader, 'intrinsics'):
            self.intrinsics = torch.from_numpy(loader.intrinsics).float().to(device)
            fx, fy = loader.intrinsics[0,0], loader.intrinsics[1,1]
        else:
            # Fallback (Only if loader is broken)
            fx = getattr(self.loader, 'fx', 320.0)
            fy = getattr(self.loader, 'fy', 320.0)
            cx = getattr(self.loader, 'cx', 320.0)
            cy = getattr(self.loader, 'cy', 240.0)
            K_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            self.intrinsics = torch.from_numpy(K_matrix).float().to(device)

        print(f"Fx: {fx} | Fy: {fy}")
        print(f"-------------------------------")

        # --- FIX 2: Robust Eval Config Loading ---
        # Look in output_dir, then project root, then dataset path
        self.eval_config = {}
        possible_paths = [
            os.path.join(output_dir, "eval_info.yaml"),
            "eval_info.yaml",
            os.path.join(loader.dataset_path, "eval_info.yaml")
        ]
        
        eval_path = next((p for p in possible_paths if os.path.exists(p)), None)
        
        if eval_path:
            with open(eval_path, 'r') as f:
                self.eval_config = yaml.safe_load(f)
            print(f"[Init] Loaded eval_info.yaml from {eval_path}")
        else:
            print(f"{Log.YELLOW}[Init] Warning: eval_info.yaml not found!{Log.END}")

        # Robust Intrinsics Loading
        if hasattr(loader, 'intrinsics'):
            K_matrix = loader.intrinsics
        elif hasattr(loader, 'K'):
            K_matrix = loader.K
        else:
            cx = getattr(self.loader, 'cx', 320.0)
            cy = getattr(self.loader, 'cy', 240.0)
            K_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
        self.intrinsics = torch.from_numpy(K_matrix).float().to(device)
        self.frame_count = 0

        # Initialize Components
        self.mapper = SLAMMapper(config, self.intrinsics)
        backbone = _SEMANTIC_CACHE[config['sem_name']]
        clip_gen = CLIPGenerator(config["clip"], backbone, device=device)
        
        self.classifier = SemanticClassifier(
            config['sem_name'], backbone, loader.class_names, 
            temperature_args=clip_gen.similarity_args
        )
        
        self.semantic_mapper = SemanticMapper(
            config, sam_model=None, clip_model=clip_gen, 
            cam_intrinsics=self.intrinsics, device=device
        )
        
        # --- OVO Spec III.3: Async CLIP Processing (Multiprocessing) ---
        self.use_async_clip = config.get("async_clip", True)
        # Multiprocessing disabled by default - worker process has issues loading CUDA models
        self.use_multiprocessing = config.get("multiprocessing_clip", False)
        self.async_clip = None
        
        if self.use_async_clip:
            if self.use_multiprocessing:
                # Use multiprocessing for true parallelism (bypasses GIL)
                clip_config = {"sem_name": config['sem_name'], "clip": config["clip"]}
                self.async_clip = MultiprocessingCLIPProcessor(clip_config, device=device)
                print("[Pipeline] Using MultiprocessingCLIPProcessor (bypasses GIL)")
            else:
                # Fallback to threading
                self.async_clip = AsyncCLIPProcessor(clip_gen, device=device)
                print("[Pipeline] Using AsyncCLIPProcessor (threaded)")
            
            self.semantic_mapper.set_async_clip_processor(self.async_clip)
            self.async_clip.start()
        
        # Visualization
        self.use_stream = config.get("vis", {}).get("stream", True)
        self.viz_queue = None
        self.viz_process = None
        self.monitor = eval_utils.PerformanceMonitor()
        
        rng = np.random.RandomState(42)
        self.viz_colors = rng.randint(0, 255, (5000, 3), dtype=np.uint8)

        if self.use_stream:
            try: mp.set_start_method('spawn', force=True)
            except RuntimeError: pass
            self.viz_queue = mp.Queue()
            self.viz_process = mp.Process(target=viz_worker, args=(self.viz_queue, self.viz_colors, self.scene_name))
            self.viz_process.start()

    def _load_gt_id_mapping(self, class_names: list) -> dict:
        """
        Load GT ID mapping for both ScanNet and Replica datasets.
        Maps raw label IDs to indices in the provided class_names list.
        
        ScanNet: GT PLY uses NYU40 label IDs (0-40), NOT ScanNet200 fine-grained IDs.
                 We map NYU40 IDs -> class names -> indices in class_names.
        Replica: Uses map_to_reduced from eval_info.yaml
        
        Returns:
            dict: {raw_label_id: index_in_class_names}
        """
        map_to_reduced = {}
        
        if isinstance(self.loader, ScanNet):
            # ScanNet: Try scannet200.yaml / scannet20.yaml
            scannet_config_paths = [
                Path(__file__).parent / "scannet200.yaml",
                Path(__file__).parent / "scannet20.yaml",
            ]
            for cfg_path in scannet_config_paths:
                if cfg_path.exists():
                    with open(cfg_path, 'r') as f:
                        scannet_cfg = yaml.safe_load(f)
                    if "valid_class_ids" in scannet_cfg and "class_names_reduced" in scannet_cfg:
                        if "map_to_reduced" in scannet_cfg:
                            map_to_reduced = {int(k): int(v) for k, v in scannet_cfg["map_to_reduced"].items()}
                            print(f"[GT] Using ScanNet YAML map_to_reduced: {len(map_to_reduced)} entries")
                            return map_to_reduced
                        
                        valid_ids = scannet_cfg["valid_class_ids"]
                        full_class_names = scannet_cfg["class_names_reduced"]
                        class_name_to_idx = {name: idx for idx, name in enumerate(class_names)}
                        for raw_id, full_class_name in zip(valid_ids, full_class_names):
                            if full_class_name in class_name_to_idx:
                                map_to_reduced[raw_id] = class_name_to_idx[full_class_name]
                        print(f"[GT] Built ScanNet ID mapping: {len(map_to_reduced)} classes")
                        return map_to_reduced
        else:
            # Replica: Use eval_info.yaml
            candidates = [
                "eval_info.yaml",
                os.path.join(str(self.loader.dataset_path), "eval_info.yaml"),
                str(Path(__file__).parent / "eval_info.yaml"),
            ]
            eval_config_path = next((p for p in candidates if os.path.exists(p)), None)
            if eval_config_path:
                with open(eval_config_path, 'r') as f:
                    eval_cfg = yaml.safe_load(f)
                map_to_reduced = {int(k): int(v) for k, v in eval_cfg.get("map_to_reduced", {}).items()}
                print(f"[GT] Loaded Replica ID mapping: {len(map_to_reduced)} entries")
        
        return map_to_reduced

    def run(self, mask_generator, classifier, detailed_logging_enabled):
        print(f"--- Starting Fusion Pipeline for {self.scene_name} ---", flush=True)
        
        # 1. Reset Peak Memory Stats for accurate reading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # 2. Config Setup
        map_every = self.config["mapping"].get("map_every", 1)
        segment_every = self.config["semantic"].get("segment_every", 1)
        
        max_frames = self.config.get("max_frames", 4000)
        if self.loader.frame_limit > 0:
             max_frames = min(max_frames, self.loader.frame_limit)

        t_start = time.time()
        total_frames = len(self.loader)
        
        # Trackers for performance monitoring
        vram_history = [] 
        
        print(f"[Pipeline] Processing {min(total_frames, max_frames)} frames. Map every: {map_every}", flush=True)

        # FPS tracking
        last_log_time = t_start
        last_log_frame = 0

        # --- MAIN LOOP ---
        for frame_id in range(total_frames):
            if frame_id >= max_frames: break
            
            self.monitor.start_frame()

            # A. LOAD DATA
            data = self.loader[frame_id]
            _, rgb, depth, pose = data
            c2w = torch.from_numpy(pose).float().to(self.device)
            frame_data_full = [frame_id, rgb, depth, pose]

            # RGB-Depth Ratio logic
            if self.loader.height != rgb.shape[0] or self.loader.width != rgb.shape[1]:
                rgb_depth_ratio = (rgb.shape[0] / self.loader.dataset_config["H"], 
                                   rgb.shape[1] / self.loader.dataset_config["W"], 
                                   self.loader.dataset_config.get("crop_edge", 0))
            else:
                rgb_depth_ratio = []

            # B. GET MASKS
            self.monitor.start_component("semantics")
            binary_maps = torch.tensor([], device=self.device)
            seg_map_np = np.array([]) 
            
            if frame_id % segment_every == 0:
                print(f"[Frame {frame_id}] Starting mask generation...", flush=True)
                seg_maps_t, binary_maps_t = mask_generator.get_masks(rgb, frame_id)
                if binary_maps_t.numel() > 0:
                    binary_maps = binary_maps_t
                    seg_map_np = seg_maps_t.cpu().numpy()
            self.monitor.end_component("semantics")

            # C. TRACKING
            self.monitor.start_component("tracking")
            track_stride = self.config["tracking"].get("track_every", 1)
            
            should_track = (
                frame_id == 0 or 
                frame_id % track_stride == 0 or 
                frame_id % map_every == 0 or 
                frame_id % segment_every == 0
            )

            if should_track:
                self.mapper.track_camera(frame_data_full)
            self.monitor.end_component("tracking")

            # D. MAPPING & FUSION
            if frame_id % map_every == 0:
                self.monitor.start_component("mapping")
                
                # 1. Standard Mapping Integration (NO HEURISTICS)
                self.mapper.map(frame_data_full, c2w)
                
                # Manual Loop Closure Signal
                if getattr(self.mapper, "map_updated", False):
                    updated_ids = self.semantic_mapper.update_map(
                        self.mapper.get_map(), 
                        self.mapper.get_kfs(), 
                        classifier=classifier
                    )
                    if updated_ids is not None:
                        self.mapper.update_pcd_obj_ids(updated_ids)
                    self.mapper.map_updated = False
                
                self.monitor.end_component("mapping")

            # E. OBJECT TRACKING (Semantic Association)
            if frame_id % segment_every == 0 and binary_maps.numel() > 0:
                self.monitor.start_component("semantics")
                estimated_c2w = self.mapper.get_c2w(frame_id)
                
                # Get SAM3 class indices if available (for sam3_semantic mode)
                sam3_class_indices = mask_generator.get_semantic_class_indices() if hasattr(mask_generator, 'get_semantic_class_indices') else []
                
                # --- FIX: Map filtered per-scene class indices to full taxonomy indices ---
                # This ensures Instance3D stores indices in the full 200-class taxonomy
                if self.filtered_to_full_class_idx and sam3_class_indices:
                    sam3_class_indices = [
                        self.filtered_to_full_class_idx.get(idx, idx) for idx in sam3_class_indices
                    ]
                
                if estimated_c2w is not None:
                    updated_obj_ids = self.semantic_mapper.detect_and_track_objects(
                        (frame_id, rgb, depth, rgb_depth_ratio), 
                        self.mapper.get_map(), 
                        estimated_c2w, 
                        binary_maps_input=binary_maps, 
                        classifier=classifier,
                        sam3_class_indices=sam3_class_indices
                    )
                    if updated_obj_ids is not None:
                        self.mapper.update_pcd_obj_ids(updated_obj_ids)
                else:
                    print(f"[Warn] Frame {frame_id}: Tracking lost. Skipping semantics.")
                self.monitor.end_component("semantics")

            # F. VISUALIZATION
            if self.use_stream and self.viz_queue is not None:
                if not self.viz_queue.full():
                    labels = {}
                    if len(self.semantic_mapper.objects) > 0:
                         labels = classifier.classify_objects(self.semantic_mapper.objects)
                    q_data = None
                    h_data = None
                    if self.semantic_mapper.history:
                        latest_hist = self.semantic_mapper.history[0] 
                        kf_ids = np.array(latest_hist["ids"])
                        kf_maps = latest_hist["binary_maps"]
                        if isinstance(kf_maps, torch.Tensor): kf_maps = kf_maps.detach().cpu().numpy()
                        q_data = (kf_ids, kf_maps)
                        h_data = (kf_maps, kf_ids)
                    self.viz_queue.put((rgb, frame_id, labels, seg_map_np, q_data, h_data))

            # G. LAZY SEMANTICS QUEUE + ASYNC CLIP COLLECTION
            self.monitor.start_component("semantics")
            self.semantic_mapper.process_queue(classifier)
            
            # OVO Spec III.3: Collect completed async CLIP results
            if self.use_async_clip and self.async_clip is not None:
                n_collected = self.semantic_mapper.collect_async_clip_results()
                if n_collected > 0 and frame_id % 100 == 0:
                    pending = self.async_clip.pending_count()
                    print(f"[AsyncCLIP] Collected {n_collected} results, {pending} pending")
            
            self.monitor.end_component("semantics")

            # --- METRICS & LOGGING ---
            if detailed_logging_enabled:
                self._log_object_birth(classifier)
            
            current_mem = 0.0
            if torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                vram_history.append(current_mem)

            self.monitor.end_frame()

            # PERIODIC LOG: Shows Progress + Current VRAM + FPS
            if frame_id % 20 == 0:
                current_time = time.time()
                elapsed = current_time - last_log_time
                frames_since_log = max(1, frame_id - last_log_frame)
                step_fps = frames_since_log / elapsed if elapsed > 0 else 0
                overall_fps = max(1, frame_id) / (current_time - t_start) if (current_time - t_start) > 0 else 0
                
                # Async CLIP stats
                clip_pending = self.async_clip.pending_count() if self.async_clip else 0
                
                print(f"[Run] Frame {frame_id}/{min(total_frames, max_frames)} | "
                      f"FPS: {step_fps:.1f} (avg: {overall_fps:.1f}) | "
                      f"VRAM: {current_mem:.0f} MB | "
                      f"CLIP pending: {clip_pending}", flush=True)
                
                last_log_time = current_time
                last_log_frame = frame_id
                
                # --- MEMORY OPTIMIZATION: Periodic cleanup ---
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # --- FINALIZATION ---
        if self.viz_process:
            self.viz_queue.put(None)
            self.viz_process.join()
            
        cv2.destroyAllWindows()
        
        # --- CALCULATE & SAVE PERFORMANCE STATS ---
        avg_vram = sum(vram_history) / len(vram_history) if vram_history else 0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        total_time = time.time() - t_start
        avg_fps = (frame_id + 1) / total_time if total_time > 0 else 0

        # Create Report String
        report_text = (
            f"\n--- Performance Monitor (Final) ---\n"
            f"Scene:        {self.scene_name}\n"
            f"Total Time:   {total_time:.2f}s\n"
            f"Average FPS:  {avg_fps:.2f} Hz\n"
            f"Avg VRAM:     {avg_vram:.2f} MB\n"
            f"Peak VRAM:    {peak_vram:.2f} MB\n"
            f"===================================\n"
        )

        # 1. Print to Console
        print(report_text)

        # 2. Save to .txt File
        if hasattr(self, 'output_dir') and self.output_dir:
            perf_path = os.path.join(self.output_dir, "performance.txt")
            with open(perf_path, "w") as f:
                f.write(report_text)
            print(f"[IO] Performance stats saved to: {perf_path}")

        return self._finalize(t_start, frame_id, classifier)
    
    def _finalize(self, t_start, last_frame, classifier):
        print(f"\n[Pipeline] Finished. Processing Time: {time.time() - t_start:.2f}s")
        
        # --- FIX: Print Performance Stats ---
        print(f"{Log.HEADER}--- Performance Monitor ---{Log.END}")
        self.monitor.report()
        print(f"-------------------------------")
        # ------------------------------------

        # 1. Ensure Semantics are flushed
        self.semantic_mapper.process_queue(classifier)
        
        # OVO Spec III.3: Flush and stop async CLIP processor
        if self.use_async_clip and self.async_clip is not None:
            print("[Pipeline] Flushing async CLIP queue...")
            self.semantic_mapper.flush_async_clip(timeout=30.0)
            self.async_clip.stop()
            print(f"[AsyncCLIP] Final stats: {self.async_clip.items_processed} items, avg {self.async_clip.avg_process_time*1000:.1f}ms/item")
        
        # 1b. Final instance fusion pass (replaces per-frame update_map)
        # OVO only runs update_map on SLAM loop closures (rare). With vanilla SLAM there are none.
        # We run it ONCE here to merge duplicate instances before evaluation.
        print("[Pipeline] Running final instance fusion pass...")
        t_fusion = time.time()
        map_data = self.mapper.get_map()
        updated_ids = self.semantic_mapper.update_map(map_data, self.mapper.get_kfs(), classifier=classifier)
        if updated_ids is not None:
            self.mapper.update_pcd_obj_ids(updated_ids)
        print(f"[Pipeline] Instance fusion complete ({time.time() - t_fusion:.1f}s, {len(self.semantic_mapper.objects)} objects)")

        pcd_pred, _, pcd_ins_ids = self.mapper.get_map()
        
        # 2. Load Evaluation Config
        # For ScanNet: Use FULL scannet200.yaml for evaluation (not filtered loader.class_names)
        # For Replica: Use eval_info.yaml class_names_reduced
        eval_config = {}
        
        if isinstance(self.loader, ScanNet):
            # ScanNet: Use FULL taxonomy for evaluation
            # loader.class_names are filtered per-scene classes (used for SAM3 segmentation)
            # But evaluation must use ALL 200 classes to match ground truth
            eval_config_path = "scannet200.yaml"
            if not os.path.exists(eval_config_path):
                eval_config_path = os.path.join(self.loader.dataset_path.parent.parent, "scannet200.yaml")
            
            if os.path.exists(eval_config_path):
                with open(eval_config_path, 'r') as f:
                    eval_config = yaml.safe_load(f)
                reduced_names = eval_config.get("class_names_reduced", [])
                print(f"[Eval] Using ScanNet200 full taxonomy: {len(reduced_names)} classes")
                
                # Map filtered predictions to full taxonomy
                # loader.class_names are a subset of reduced_names
                print(f"[Eval] SAM3 used {len(self.loader.class_names)} filtered classes for segmentation")
            else:
                print(f"{Log.RED}[Error] scannet200.yaml not found for evaluation!{Log.END}")
                return None
        else:
            # Fallback to eval_info.yaml (Replica)
            eval_config_path = "eval_info.yaml"
            if not os.path.exists(eval_config_path):
                eval_config_path = os.path.join(self.loader.dataset_path, "eval_info.yaml")

            if os.path.exists(eval_config_path):
                with open(eval_config_path, 'r') as f:
                    eval_config = yaml.safe_load(f)
                reduced_names = eval_config.get("class_names_reduced", [])
                print(f"[Eval] Using Replica taxonomy: {len(reduced_names)} classes")
            else:
                print(f"{Log.RED}[Error] No class names available. Cannot evaluate.{Log.END}")
                return None

        # 3. CLASSIFICATION - SAM3 Semantic vs CLIP
        # Check if we have SAM3 semantic labels (sam3_semantic mode)
        ordered_obj_ids = list(self.semantic_mapper.objects.keys())
        
        # Count how many objects have direct SAM3 semantic labels
        sam3_labeled_count = sum(
            1 for obj in self.semantic_mapper.objects.values() 
            if obj.semantic_class_idx >= 0
        )
        
        use_sam3_labels = sam3_labeled_count > len(ordered_obj_ids) * 0.5  # Use SAM3 if >50% have labels
        
        if use_sam3_labels:
            # --- SAM3 SEMANTIC MODE: Use direct labels from text-prompted segmentation ---
            print(f"{Log.GREEN}[Eval] Using SAM3 Semantic labels ({sam3_labeled_count}/{len(ordered_obj_ids)} objects){Log.END}")
            
            # Note: With the filtered->full index mapping fix, Instance3D now stores
            # full taxonomy indices directly (0-199), so no additional mapping is needed here
            filtered_classes = self.loader.class_names if hasattr(self.loader, 'class_names') else reduced_names
            print(f"[Eval] SAM3 indices already mapped to full taxonomy ({len(reduced_names)} classes)")
            
            # --- SigLIP K-Pool Validation ---
            # Pass full taxonomy (reduced_names) since Instance3D stores full taxonomy indices
            validated_labels = self._validate_with_siglip_kpool(reduced_names)
            
            # Fix #3: Pre-compute CLIP classifications as fallback for low-vote objects
            clip_fallback = {}
            try:
                clip_info = self.semantic_mapper.classify_instances(reduced_names, th=0)
                for i, oid in enumerate(ordered_obj_ids):
                    if i < len(clip_info["classes"]):
                        clip_fallback[oid] = int(clip_info["classes"][i])
            except Exception as e:
                print(f"[Eval] CLIP fallback failed: {e}")
            
            lookup_table = np.full(max(ordered_obj_ids) + 1 if ordered_obj_ids else 1, -1, dtype=np.int64)
            matched_count = 0
            override_count = 0
            clip_fallback_count = 0
            
            for obj_id in ordered_obj_ids:
                obj = self.semantic_mapper.objects[obj_id]
                sam3_class = obj.semantic_class_idx
                
                if obj_id in validated_labels:
                    # High-vote objects: use SAM3 + SigLIP validated label
                    class_idx = validated_labels[obj_id]
                    if class_idx != sam3_class:
                        override_count += 1
                elif obj_id in clip_fallback and clip_fallback[obj_id] >= 0:
                    # Fix #3: Low-vote objects: use CLIP classification instead of unreliable single SAM3 vote
                    class_idx = clip_fallback[obj_id]
                    clip_fallback_count += 1
                else:
                    class_idx = sam3_class
                
                if class_idx >= 0 and class_idx < len(reduced_names):
                    lookup_table[obj_id] = class_idx
                    matched_count += 1
            
            print(f"[Eval] Assigned {matched_count} objects ({override_count} overridden by SigLIP, {clip_fallback_count} via CLIP fallback).")
        else:
            # --- CLIP MODE: Standard similarity-based classification ---
            print(f"[Eval] Running CLIP Classification via Semantic Mapper Query on {len(reduced_names)} classes...")
            
            # This returns {'classes': [id, ...], 'conf': [...]}}
            instances_info = self.semantic_mapper.classify_instances(reduced_names, th=0)
            pred_class_indices = instances_info["classes"]
            pred_confidences = instances_info["conf"]
            
            # Log classification confidence stats
            valid_confs = pred_confidences[pred_class_indices >= 0]
            if len(valid_confs) > 0:
                print(f"[Eval] Confidence stats: min={valid_confs.min():.3f}, max={valid_confs.max():.3f}, mean={valid_confs.mean():.3f}")
            
            # Map Object ID -> Predicted Class ID
            lookup_table = np.full(max(ordered_obj_ids) + 1 if ordered_obj_ids else 1, -1, dtype=np.int64)
            
            matched_count = 0
            for i, obj_id in enumerate(ordered_obj_ids):
                class_idx = pred_class_indices[i]
                if class_idx >= 0:
                    lookup_table[obj_id] = class_idx
                    matched_count += 1
                    
            print(f"[Eval] Classified {matched_count} objects via CLIP.")
        
        # --- DIAGNOSTIC: Log vote distributions for debugging ---
        self._log_vote_distributions(reduced_names)

        # 4. Load GT Mesh - supports both ScanNet and Replica formats
        scene_name = self.scene_name
        gt_mesh_candidates = [
            # ScanNet format
            self.loader.dataset_path / f"{scene_name}_vh_clean_2.ply",
            self.loader.dataset_path / f"{scene_name}_vh_clean.ply",
            # Replica format
            self.loader.dataset_path.parent / f"{scene_name}_mesh.ply",
            self.loader.dataset_path / f"{scene_name}_mesh.ply",
        ]
        gt_mesh_path = next((p for p in gt_mesh_candidates if p.exists()), None)
        
        if gt_mesh_path is None:
            print(f"[Eval] GT Mesh not found. Tried: {[str(p.name) for p in gt_mesh_candidates]}")
            return None
        
        print(f"[Eval] Using GT Mesh: {gt_mesh_path.name}")

        from plyfile import PlyData
        plydata = PlyData.read(str(gt_mesh_path))
        gt_vertices = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        gt_vertices = torch.from_numpy(gt_vertices).float()

        pcd_eval = pcd_pred.clone()

        print(f"[Eval] Projecting {len(pcd_eval)} SLAM points to {len(gt_vertices)} GT vertices...")
        
        # 5. Instance-First Projection
        mesh_instance_ids, _, _ = eval_utils.match_labels_to_vtx(
            pcd_ins_ids.squeeze().cpu(), 
            pcd_eval.cpu(), 
            gt_vertices, 
            filter_unasigned=False
        )

        # 6. Generate Final Labels
        # lookup_table maps Instance ID -> Reduced Class ID directly now
        max_ins_id = lookup_table.shape[0]
        # Handle instances that might be out of bounds (though unlikely with tight loop)
        safe_indices = mesh_instance_ids.numpy().astype(int)
        safe_indices[safe_indices >= max_ins_id] = -1
        pred_labels_np = np.full(safe_indices.shape, -1, dtype=np.int64)
        valid_mask = safe_indices >= 0
        pred_labels_np[valid_mask] = lookup_table[safe_indices[valid_mask]]

        # 7. Save & Eval
        os.makedirs(self.output_dir, exist_ok=True)
        pred_txt_path = os.path.join(self.output_dir, f"{self.scene_name}.txt")
        np.savetxt(pred_txt_path, pred_labels_np, fmt="%d")

        # Find GT labels file
        # Priority: scannet200_gt (fine-grained) > semantic_gt > Replica fallback
        scannet_root = self.loader.dataset_path.parent.parent  # scannet_data/
        gt_txt_path = None

        # 1. ScanNet200: pre-generated fine-grained GT labels (needs map_to_reduced)
        scannet200_gt = scannet_root / "scannet200_gt" / f"{self.scene_name}.txt"
        if scannet200_gt.exists():
            gt_txt_path = scannet200_gt
            print(f"[Eval] Using ScanNet200 GT: {gt_txt_path}")

        # 2. semantic_gt (ScanNet20 / Replica)
        if gt_txt_path is None:
            for candidate in [
                self.loader.dataset_path.parent / "semantic_gt" / f"{self.scene_name}.txt",
                scannet_root / "semantic_gt" / f"{self.scene_name}.txt",
            ]:
                if candidate.exists():
                    gt_txt_path = candidate
                    print(f"[Eval] Using semantic GT: {gt_txt_path}")
                    break

        # 3. Replica fallback: scene directory
        if gt_txt_path is None:
            replica_gt = self.loader.dataset_path / f"{self.scene_name}.txt"
            if replica_gt.exists():
                gt_txt_path = replica_gt
                print(f"[Eval] Using Replica GT: {gt_txt_path}")

        if gt_txt_path is not None and gt_txt_path.exists():
            print(f"[Eval] Comparing against GT: {gt_txt_path}")

            # Ensure eval_config is loaded (contains map_to_reduced)
            if not eval_config:
                eval_config_path = "eval_info.yaml"
                if not os.path.exists(eval_config_path):
                    eval_config_path = os.path.join(self.loader.dataset_path, "eval_info.yaml")
                if os.path.exists(eval_config_path):
                    with open(eval_config_path, 'r') as f:
                        eval_config = yaml.safe_load(f)
                    print(f"[Eval] Loaded eval_config from: {eval_config_path}")

            dataset_info = eval_config.copy() if eval_config else {}
            dataset_info["num_classes"] = len(reduced_names)
            dataset_info["class_names"] = reduced_names

            # map_to_reduced converts raw GT IDs → reduced class indices (0-N)
            # ScanNet200 GT has fine-grained IDs (1-1191) → needs map_to_reduced
            # Replica GT has raw IDs → needs map_to_reduced
            # Both cases: KEEP map_to_reduced in dataset_info

            miou, _ = eval_utils.eval_semantics(
                output_path=self.output_dir, 
                gt_path=str(gt_txt_path.parent),
                scenes=[self.scene_name],
                dataset_info=dataset_info,
                verbose=True
            )
            
            # Print metrics to console
            print(f"\n{Log.GREEN}[Eval] Scene: {self.scene_name} | mIoU: {miou:.2f}%{Log.END}\n")
            
            return {"scene": self.scene_name, "miou": miou}
        
        return 0.0
    
    def _generate_gt_txt(self, mesh_path, out_path):
        from plyfile import PlyData
        plydata = PlyData.read(mesh_path)
        v = plydata['vertex']
        labels = None
        for name in ['object_id', 'semantic_id', 'class_id']:
            if name in v.properties:
                labels = np.asarray(v[name])
                break
        if labels is not None:
            np.savetxt(out_path, labels, fmt="%d")
    
    def _log_object_birth(self, classifier):
        """
        Checks for objects that have features but haven't been logged yet.
        Prints a detailed breakdown of their first classification.
        """
        # Iterate over all objects
        for oid, obj in self.semantic_mapper.objects.items():
            
            # Condition: Has CLIP feature BUT hasn't been reported
            if obj.clip_feature is not None and not obj.reported:
                
                # 1. Get Geometry Stats
                pts = obj.birth_stats["n_points"]
                area = obj.birth_stats["mask_area"]
                
                # 2. Get Semantic Stats
                # We manually run the similarity logic to capture intermediate values
                with torch.no_grad():
                    # Normalize image feature
                    img_feat = obj.clip_feature.to(self.device).float()
                    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-6)
                    
                    # Get Raw Similarity (Dot Product)
                    # Shape: [1, N_Classes]
                    raw_sim = classifier._similarity_logits(img_feat).squeeze()
                    
                    # Get Temperature/Bias (Specific to SigLIP/CLIP)
                    temp_str = "N/A"
                    if classifier.siglip and classifier.temperature is not None:
                        t = classifier.temperature.item() if isinstance(classifier.temperature, torch.Tensor) else classifier.temperature
                        b = classifier.bias.item() if classifier.bias is not None else 0.0
                        temp_str = f"Temp={t:.4f}, Bias={b:.4f}"
                    
                    # Get Probabilities
                    probs = torch.nn.functional.softmax(raw_sim, dim=0)
                    
                    # Get Top 3 Predictions
                    top_vals, top_idxs = torch.topk(probs, 3)
                
                # 3. Formulate the Log
                print(f"\n{Log.GREEN}=== [BIRTH CERTIFICATE] ID {oid} ==={Log.END}")
                print(f"  {Log.BLUE}Geometry:{Log.END}  Points={pts}, Initial Area={area} px")
                print(f"  {Log.BLUE}Model Params:{Log.END} {temp_str}")
                print(f"  {Log.BLUE}Top 3 Classifications:{Log.END}")
                
                for i in range(len(top_idxs)):
                    idx = top_idxs[i].item()
                    label = classifier.labels[idx]
                    score = top_vals[i].item()
                    logit = raw_sim[idx].item()
                    print(f"    {i+1}. {Log.BOLD}{label}{Log.END} \t(Conf: {score:.4f} | Raw Logit: {logit:.4f})")
                
                print(f"{Log.GREEN}===================================={Log.END}\n")
                
                # Mark as reported so we don't spam
                obj.reported = True

    def _log_vote_distributions(self, class_names: list):
        """Log detailed vote distributions for all objects to diagnose classification issues."""
        print(f"\n{Log.HEADER}=== VOTE DISTRIBUTION DIAGNOSTIC ==={Log.END}")
        
        # --- Load GT labels for comparison ---
        gt_instance_labels = {}
        try:
            from plyfile import PlyData
            from scipy.spatial import KDTree
            
            # Load map_to_reduced - supports both ScanNet and Replica formats
            map_to_reduced = self._load_gt_id_mapping(class_names)

            gt_vertices = None
            gt_semantic_ids = None

            # Try ScanNet200 GT txt first (fine-grained IDs)
            scannet_root = self.loader.dataset_path.parent.parent
            scannet200_gt_path = scannet_root / "scannet200_gt" / f"{self.scene_name}.txt"
            scannet_mesh_ply = self.loader.dataset_path / f"{self.scene_name}_vh_clean_2.ply"

            if scannet200_gt_path.exists() and scannet_mesh_ply.exists():
                plydata = PlyData.read(str(scannet_mesh_ply))
                gt_vertices = np.vstack([
                    plydata['vertex']['x'],
                    plydata['vertex']['y'],
                    plydata['vertex']['z']
                ]).T
                gt_semantic_ids = np.loadtxt(str(scannet200_gt_path), dtype=int)
                print(f"[VoteDiag] Loaded ScanNet200 GT: {len(gt_semantic_ids)} vertices")

            # Fallback: ScanNet labels.ply (NYU40 IDs)
            if gt_vertices is None:
                scannet_labeled_ply = self.loader.dataset_path / f"{self.scene_name}_vh_clean_2.labels.ply"
                if scannet_labeled_ply.exists():
                    plydata = PlyData.read(str(scannet_labeled_ply))
                    gt_vertices = np.vstack([
                        plydata['vertex']['x'],
                        plydata['vertex']['y'],
                        plydata['vertex']['z']
                    ]).T
                    if 'label' in plydata['vertex'].data.dtype.names:
                        gt_semantic_ids = np.array(plydata['vertex']['label'])
                        print(f"[VoteDiag] Loaded ScanNet labeled PLY (NYU40): {len(gt_semantic_ids)} vertices")

            # Fallback: Replica format (separate mesh + labels files)
            if gt_vertices is None:
                gt_mesh_path = self.loader.dataset_path.parent / f"{self.scene_name}_mesh.ply"
                if not gt_mesh_path.exists():
                    gt_mesh_path = self.loader.dataset_path / f"{self.scene_name}_mesh.ply"

                gt_labels_path = self.loader.dataset_path / f"{self.scene_name}.txt"
                if not gt_labels_path.exists():
                    gt_labels_path = self.loader.dataset_path.parent / "semantic_gt" / f"{self.scene_name}.txt"

                if gt_mesh_path.exists() and gt_labels_path.exists():
                    plydata = PlyData.read(str(gt_mesh_path))
                    gt_vertices = np.vstack([
                        plydata['vertex']['x'],
                        plydata['vertex']['y'],
                        plydata['vertex']['z']
                    ]).T
                    gt_semantic_ids = np.loadtxt(str(gt_labels_path), dtype=int)

            # Match GT labels to predicted instances
            if gt_vertices is not None and gt_semantic_ids is not None:
                if len(gt_semantic_ids) == len(gt_vertices):
                    gt_tree = KDTree(gt_vertices)
                    pcd_pred, _, pcd_ins_ids = self.mapper.get_map()
                    pcd_np = pcd_pred.cpu().numpy()
                    ins_ids_np = pcd_ins_ids.squeeze().cpu().numpy()

                    for oid in self.semantic_mapper.objects.keys():
                        instance_mask = ins_ids_np == oid
                        if not np.any(instance_mask):
                            continue
                        instance_points = pcd_np[instance_mask]
                        if len(instance_points) == 0:
                            continue
                        if len(instance_points) > 100:
                            indices = np.random.choice(len(instance_points), 100, replace=False)
                            instance_points = instance_points[indices]
                        _, gt_indices = gt_tree.query(instance_points, k=1)
                        gt_labels_for_instance = gt_semantic_ids[gt_indices]
                        unique, counts = np.unique(gt_labels_for_instance, return_counts=True)
                        majority_raw_id = int(unique[np.argmax(counts)])
                        # Convert raw class ID to reduced index using map_to_reduced
                        reduced_idx = map_to_reduced.get(majority_raw_id, -1)
                        if 0 <= reduced_idx < len(class_names):
                            gt_instance_labels[oid] = class_names[reduced_idx]
                        else:
                            gt_instance_labels[oid] = f"unmapped({majority_raw_id})"
                    print(f"[VoteDiag] Loaded GT labels for {len(gt_instance_labels)} instances")
                else:
                    print(f"[VoteDiag] GT mismatch: {len(gt_semantic_ids)} labels vs {len(gt_vertices)} vertices")
            else:
                print(f"[VoteDiag] GT files not found for {self.scene_name}")
        except Exception as e:
            print(f"[VoteDiag] Could not load GT labels: {e}")
        
        # Collect stats for ALL objects
        all_stats = []
        misclassified = []
        low_confidence = []
        
        # Get CLIP classification for fallback (mobile_sam mode)
        clip_results = {}
        clip_confidences = {}
        try:
            instances_info = self.semantic_mapper.classify_instances(class_names, th=0)
            ordered_ids = list(self.semantic_mapper.objects.keys())
            for i, oid in enumerate(ordered_ids):
                if i < len(instances_info["classes"]):
                    clip_results[oid] = int(instances_info["classes"][i])
                    clip_confidences[oid] = float(instances_info["conf"][i])
        except:
            pass
        
        for oid, obj in self.semantic_mapper.objects.items():
            n_points = len(obj.points_ids) if hasattr(obj, 'points_ids') else 0
            gt_label = gt_instance_labels.get(oid, None)
            
            # Get SAM3 semantic class votes
            sem_votes = dict(obj.semantic_class_votes) if hasattr(obj, 'semantic_class_votes') else {}
            total_sem_votes = sum(sem_votes.values())
            
            # Get predicted class - prefer SAM3, fallback to CLIP
            pred_idx = obj.semantic_class_idx if hasattr(obj, 'semantic_class_idx') else -1
            if pred_idx < 0 and oid in clip_results:
                pred_idx = clip_results[oid]
            pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else "unassigned"
            
            # Calculate confidence - SAM3 vote-based or CLIP similarity
            confidence = 0.0
            if total_sem_votes > 0 and pred_idx in sem_votes:
                confidence = sem_votes[pred_idx] / total_sem_votes
            elif oid in clip_confidences:
                confidence = clip_confidences[oid]
            
            # Build vote distribution with class names (SAM3 mode only)
            vote_dist = []
            for cls_idx, vote_count in sorted(sem_votes.items(), key=lambda x: -x[1]):
                cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"idx_{cls_idx}"
                pct = 100.0 * vote_count / total_sem_votes if total_sem_votes > 0 else 0
                vote_dist.append((cls_name, vote_count, pct))
            
            stat = {
                'oid': oid,
                'n_points': n_points,
                'pred_idx': pred_idx,
                'pred_name': pred_name,
                'gt_label': gt_label,
                'total_votes': total_sem_votes,
                'confidence': confidence,
                'vote_dist': vote_dist,
                'n_keyframes': len(obj.kfs_ids) if hasattr(obj, 'kfs_ids') else 0
            }
            all_stats.append(stat)
            
            # Track issues
            if gt_label and pred_name != gt_label:
                misclassified.append(stat)
            if confidence < 0.6 and total_sem_votes > 3:
                low_confidence.append(stat)
        
        # Summary
        n_with_votes = len([s for s in all_stats if s['total_votes'] > 0])
        n_correct = len([s for s in all_stats if s['gt_label'] and s['pred_name'] == s['gt_label']])
        n_with_gt = len([s for s in all_stats if s['gt_label']])
        avg_conf = sum(s['confidence'] for s in all_stats if s['total_votes'] > 0) / max(n_with_votes, 1)
        avg_votes = sum(s['total_votes'] for s in all_stats) / max(len(all_stats), 1)
        
        print(f"\n[VoteDiag] Total objects: {len(all_stats)}")
        print(f"[VoteDiag] Objects with SAM3 votes: {n_with_votes}")
        print(f"[VoteDiag] Avg votes per object: {avg_votes:.1f}")
        print(f"[VoteDiag] Avg confidence: {avg_conf:.1%}")
        if n_with_gt > 0:
            print(f"[VoteDiag] Instance accuracy: {n_correct}/{n_with_gt} ({100*n_correct/n_with_gt:.1f}%)")
        
        # Show misclassified
        if misclassified:
            print(f"\n{Log.RED}[VoteDiag] MISCLASSIFIED ({len(misclassified)}):{Log.END}")
            for s in misclassified[:10]:
                print(f"  ID {s['oid']}: pred={s['pred_name']}, GT={s['gt_label']}, conf={s['confidence']:.1%}")
                for name, votes, pct in s['vote_dist'][:3]:
                    marker = " <-- GT" if name == s['gt_label'] else ""
                    print(f"    {name}: {votes} votes ({pct:.1f}%){marker}")
        
        # Show low confidence
        if low_confidence:
            print(f"\n{Log.YELLOW}[VoteDiag] LOW CONFIDENCE ({len(low_confidence)}):{Log.END}")
            for s in low_confidence[:10]:
                gt_str = f", GT={s['gt_label']}" if s['gt_label'] else ""
                print(f"  ID {s['oid']}: pred={s['pred_name']}, conf={s['confidence']:.1%}, votes={s['total_votes']}{gt_str}")
                for name, votes, pct in s['vote_dist'][:3]:
                    print(f"    {name}: {votes} votes ({pct:.1f}%)")
        
        # Save detailed log to file
        if hasattr(self, 'output_dir') and self.output_dir:
            log_path = os.path.join(self.output_dir, "vote_distributions.txt")
            with open(log_path, "w") as f:
                f.write("SAM3 SEMANTIC CLASS VOTE DISTRIBUTION\n")
                f.write("="*70 + "\n\n")
                f.write(f"Total Objects: {len(all_stats)}\n")
                f.write(f"Objects with Votes: {n_with_votes}\n")
                f.write(f"Avg Confidence: {avg_conf:.1%}\n")
                if n_with_gt > 0:
                    f.write(f"Instance Accuracy: {n_correct}/{n_with_gt} ({100*n_correct/n_with_gt:.1f}%)\n")
                f.write("\n" + "="*70 + "\n\n")
                
                # Sort by points (largest objects first)
                for stat in sorted(all_stats, key=lambda x: -x['n_points']):
                    oid = stat['oid']
                    gt = stat['gt_label'] or "N/A"
                    status = "CORRECT" if gt == stat['pred_name'] else "WRONG" if stat['gt_label'] else "NO_GT"
                    
                    f.write(f"Object ID: {oid} [{status}]\n")
                    f.write(f"  Predicted: {stat['pred_name']} (idx={stat['pred_idx']})\n")
                    f.write(f"  GT Label: {gt}\n")
                    f.write(f"  Points: {stat['n_points']:,}\n")
                    f.write(f"  Keyframes: {stat['n_keyframes']}\n")
                    f.write(f"  Total Votes: {stat['total_votes']}\n")
                    f.write(f"  Confidence: {stat['confidence']:.1%}\n")
                    
                    if stat['vote_dist']:
                        f.write(f"  Vote Distribution:\n")
                        for name, votes, pct in stat['vote_dist']:
                            marker = " <-- WINNER" if name == stat['pred_name'] else ""
                            marker = " <-- GT" if name == gt else marker
                            f.write(f"    {name}: {votes} votes ({pct:.1f}%){marker}\n")
                    else:
                        f.write(f"  Vote Distribution: (no votes)\n")
                    f.write("\n")
                    
            print(f"\n[VoteDiag] Detailed log saved to: {log_path}")
        print(f"{Log.HEADER}======================================{Log.END}\n")

    def _validate_with_siglip_kpool(self, class_names: list) -> dict:
        """
        SigLIP K-Pool Validation: Override SAM3's classification using SigLIP
        against the pool of classes that SAM3 has voted for.
        
        For each object:
          1. Get all classes SAM3 voted for (the K-pool)
          2. Compute SigLIP similarity against ONLY those classes
          3. Return SigLIP's pick as the validated class
        
        Returns:
            dict: {obj_id: validated_class_idx} for all validated objects
        """
        validation_config = self.config.get("siglip_validation", {})
        enabled = validation_config.get("enabled", True)
        min_votes = validation_config.get("min_votes", 3)
        min_keyframes = validation_config.get("min_keyframes", 3)  # Filter noisy single-frame objects
        min_siglip_conf = validation_config.get("min_siglip_conf", 0.50)  # For K-pool mode
        temperature = validation_config.get("temperature", 0.1)  # For K-pool mode
        synonym_threshold = validation_config.get("synonym_threshold", 0.85)  # Text-embed cosine sim to detect synonyms
        
        if not enabled:
            print(f"[SigLIP-Val] Validation disabled, using SAM3 labels directly")
            return {}
        
        print(f"\n{Log.HEADER}=== SIGLIP K-POOL VALIDATION ==={Log.END}")
        print(f"[SigLIP-Val] Min votes: {min_votes}")
        print(f"[SigLIP-Val] Min SigLIP conf: {min_siglip_conf}")
        print(f"[SigLIP-Val] Temperature: {temperature}")
        print(f"[SigLIP-Val] Synonym threshold: {synonym_threshold}")
        
        # Debug: Print SigLIP scaling parameters
        if hasattr(self.semantic_mapper, 'clip_generator'):
            cg = self.semantic_mapper.clip_generator
            if cg.siglip and len(cg.similarity_args) == 2:
                scale, bias = cg.similarity_args
                print(f"[SigLIP-Val] Scale: {scale.item():.4f} (exp={scale.exp().item():.4f}), Bias: {bias.item():.4f}")
        
        validated_labels = {}
        validation_stats = {
            "total": 0, "validated": 0, "overrides": 0, "agreements": 0,
            "skipped_no_clip": 0, "skipped_no_votes": 0, "skipped_few_keyframes": 0,
            "synonym_overrides": 0
        }
        validation_log = []
        
        # Fix #1: Find "object" class index to block overrides to it
        object_class_idx = -1
        for i, name in enumerate(class_names):
            if name.lower() == "object":
                object_class_idx = i
                break
        if object_class_idx >= 0:
            print(f"[SigLIP-Val] Blocking overrides TO 'object' (idx={object_class_idx})")
        
        for obj_id, obj in self.semantic_mapper.objects.items():
            validation_stats["total"] += 1
            
            sam3_class = obj.semantic_class_idx
            sam3_votes = dict(obj.semantic_class_votes) if hasattr(obj, 'semantic_class_votes') else {}
            total_votes = sum(sam3_votes.values())
            
            if total_votes < min_votes:
                validation_stats["skipped_no_votes"] += 1
                # Fix #3: Don't add to validated_labels — let these fall through to CLIP
                continue
            
            # Minimum keyframe filter - skip noisy single-frame objects
            n_keyframes = len(obj.kfs_ids) if hasattr(obj, 'kfs_ids') else 0
            if n_keyframes < min_keyframes:
                validation_stats["skipped_few_keyframes"] += 1
                validated_labels[obj_id] = sam3_class
                continue
            
            if obj.clip_feature is None:
                validation_stats["skipped_no_clip"] += 1
                validated_labels[obj_id] = sam3_class
                continue
            
            sam3_conf = sam3_votes.get(sam3_class, 0) / total_votes if total_votes > 0 else 0.0
            
            # K-pool mode: only use classes SAM3 voted for
            candidate_indices = list(sam3_votes.keys())
            candidate_names = [class_names[i] if 0 <= i < len(class_names) else f"idx_{i}" for i in candidate_indices]
            
            if len(candidate_indices) == 0:
                validated_labels[obj_id] = sam3_class
                continue
            
            try:
                img_feat = obj.clip_feature.to(self.device).float()
                img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-6)
                
                # Use RAW COSINE SIMILARITY (not sigmoid-scaled)
                # This gives interpretable 0-1 range instead of SigLIP's sharp transitions
                cg = self.semantic_mapper.clip_generator
                
                # Compute ensemble text embeddings (mean across 80 templates per class)
                n_cands = len(candidate_names)
                txt_embeds = torch.zeros((n_cands, img_feat.shape[-1]), device=self.device)
                for j, name in enumerate(candidate_names):
                    # Format with all templates and mean-pool
                    templated = [t.format(name.replace("-", " ")) for t in IMAGENET_TEMPLATES]
                    embed = cg.get_txt_embedding(templated).mean(0, keepdim=True).float()
                    txt_embeds[j] = torch.nn.functional.normalize(embed, p=2, dim=-1)
                
                # Raw cosine similarity (no sigmoid scaling)
                if img_feat.dim() == 1:
                    img_feat = img_feat.unsqueeze(0)
                similarities = (img_feat @ txt_embeds.T).squeeze()
                
                # Handle edge case: single candidate (0-dim tensor after squeeze)
                if similarities.dim() == 0:
                    best_local_idx = 0
                    siglip_conf = 1.0  # Only one candidate = 100% confident
                    raw_sim = similarities.item()
                else:
                    # Apply softmax temperature scaling for relative confidence
                    probs = torch.softmax(similarities / temperature, dim=0)
                    best_local_idx = probs.argmax().item()
                    siglip_conf = probs[best_local_idx].item()  # Now 0-1 relative confidence
                    raw_sim = similarities[best_local_idx].item()
                
                siglip_class = candidate_indices[best_local_idx]
                
                # Compute text-embedding similarity between SAM3's pick and SigLIP's pick
                # for synonym detection (no hardcoded alias maps needed)
                text_sim = 0.0
                if siglip_class != sam3_class and sam3_class in candidate_indices:
                    sam3_local_idx = candidate_indices.index(sam3_class)
                    text_sim = (txt_embeds[sam3_local_idx] @ txt_embeds[best_local_idx]).item()
                
            except Exception as e:
                print(f"[SigLIP-Val] Error validating obj {obj_id}: {e}")
                validated_labels[obj_id] = sam3_class
                continue
            
            # Decide whether to override
            # Three-tier logic:
            # 0. Synonym: SAM3 & SigLIP picks are near-synonyms → trust SigLIP (any confidence)
            # 1. SigLIP > 60% → override regardless of SAM3 (high confidence)
            # 2. SigLIP > min_siglip_conf AND SAM3 < 80% → override (medium confidence + weak SAM3)
            should_override = False
            override_reason = ""
            
            if siglip_class != sam3_class:
                if siglip_class == object_class_idx:
                    override_reason = "blocked_object_override"
                elif text_sim >= synonym_threshold:
                    should_override = True
                    override_reason = f"synonym_override(sim={text_sim:.3f})"
                    validation_stats["synonym_overrides"] += 1
                elif siglip_conf > 0.60:
                    should_override = True
                    override_reason = "siglip_confident"
                elif siglip_conf > min_siglip_conf and sam3_conf < 0.80:
                    should_override = True
                    override_reason = "simple_override"
                elif siglip_conf <= min_siglip_conf:
                    override_reason = f"siglip_low({siglip_conf:.2f})"
                else:
                    override_reason = f"sam3_protected({sam3_conf:.0%})"
            
            # Apply decision
            if should_override:
                final_class = siglip_class
                validation_stats["overrides"] += 1
            else:
                final_class = sam3_class
                validation_stats["agreements"] += 1
            
            validated_labels[obj_id] = final_class
            validation_stats["validated"] += 1
            
            sam3_name = class_names[sam3_class] if 0 <= sam3_class < len(class_names) else f"idx_{sam3_class}"
            siglip_name = class_names[siglip_class] if 0 <= siglip_class < len(class_names) else f"idx_{siglip_class}"
            final_name = class_names[final_class] if 0 <= final_class < len(class_names) else f"idx_{final_class}"
            # Build K-pool string from SAM3 votes (not full-class indices)
            kpool_indices = list(sam3_votes.keys())
            cand_dist = ", ".join([f"{class_names[idx]}:{sam3_votes[idx]}" for idx in kpool_indices[:5]])
            
            validation_log.append({
                "obj_id": obj_id,
                "n_points": len(obj.points_ids) if hasattr(obj, 'points_ids') else 0,
                "sam3_class": sam3_name, "sam3_conf": sam3_conf,
                "siglip_class": siglip_name, "siglip_conf": siglip_conf,
                "final_class": final_name,
                "overridden": siglip_class != sam3_class and final_class == siglip_class,
                "override_reason": override_reason,
                "text_sim": text_sim,
                "candidates": cand_dist
            })
        
        print(f"\n[SigLIP-Val] Results:")
        print(f"  Total objects:     {validation_stats['total']}")
        print(f"  Validated:         {validation_stats['validated']}")
        print(f"  Overrides:         {validation_stats['overrides']} (synonym: {validation_stats['synonym_overrides']})")
        print(f"  Agreements:        {validation_stats['agreements']}")
        print(f"  Skipped (no CLIP): {validation_stats['skipped_no_clip']}")
        
        overrides = [e for e in validation_log if e["overridden"]]
        if overrides:
            print(f"\n[SigLIP-Val] OVERRIDES ({len(overrides)}):")
            for e in overrides[:15]:
                reason = e.get('override_reason', '')
                sim_str = f" [txt_sim={e['text_sim']:.3f}]" if e.get('text_sim', 0) > 0 else ""
                print(f"  ID {e['obj_id']:3d}: SAM3={e['sam3_class']:12s} ({e['sam3_conf']:.1%}) -> SigLIP={e['siglip_class']:12s} ({e['siglip_conf']:.2f}) {reason}{sim_str}")
        
        if hasattr(self, 'output_dir') and self.output_dir:
            log_path = os.path.join(self.output_dir, "siglip_validation.txt")
            with open(log_path, "w") as f:
                f.write("SIGLIP K-POOL VALIDATION LOG\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Min Votes: {min_votes}\n")
                f.write(f"Min SigLIP Conf: {min_siglip_conf}, Temperature: {temperature}\n")
                f.write(f"Synonym Threshold: {synonym_threshold}\n\n")
                f.write(f"Total: {validation_stats['total']}, Validated: {validation_stats['validated']}\n")
                f.write(f"Overrides: {validation_stats['overrides']}, Agreements: {validation_stats['agreements']}\n")
                f.write(f"Synonym Overrides: {validation_stats['synonym_overrides']}\n\n")
                f.write("=" * 70 + "\n\n")
                
                for e in sorted(validation_log, key=lambda x: -x['n_points']):
                    status = "OVERRIDE" if e['overridden'] else "AGREE"
                    f.write(f"Object ID: {e['obj_id']} [{status}]\n")
                    f.write(f"  Points: {e['n_points']:,}\n")
                    f.write(f"  SAM3:   {e['sam3_class']} (conf: {e['sam3_conf']:.1%})\n")
                    f.write(f"  SigLIP: {e['siglip_class']} (conf: {e['siglip_conf']:.3f})\n")
                    if e.get('text_sim', 0) > 0:
                        f.write(f"  Text Sim: {e['text_sim']:.3f}\n")
                    if e.get('override_reason'):
                        f.write(f"  Reason: {e['override_reason']}\n")
                    f.write(f"  Final:  {e['final_class']}\n")
                    f.write(f"  K-Pool: [{e['candidates']}]\n\n")
            print(f"[SigLIP-Val] Detailed log saved to: {log_path}")
        
        print(f"{Log.HEADER}================================{Log.END}\n")
        return validated_labels

def run_scene_unified(scene, sem_cfg, project_root, cam_config=None, prompt_mode="ensemble", mask_mode="sam3", output_dir=None, script_dir=None, llm_prompts=None, sam3_selection_method="first", dataset_type="replica"):
    # 1. Output Directory Logic
    if output_dir is None:
        output_dir = os.path.join(project_root, "results", dataset_type, prompt_mode, mask_mode, scene)
    
    os.makedirs(output_dir, exist_ok=True)

    if cam_config is None:
        print("[Error] Camera config is missing!")
        return None

    c = cam_config['cam']
    
    # 2. Dataset Config - varies by dataset type
    if dataset_type in ["scannet", "scannet20", "scannet200"]:
        # Determine eval config path based on dataset type
        if dataset_type == "scannet20":
            eval_config_path = os.path.join(script_dir, "scannet20.yaml") if script_dir else "scannet20.yaml"
        else:  # scannet200 or scannet
            eval_config_path = os.path.join(script_dir, "scannet200.yaml") if script_dir else "scannet200.yaml"
        
        # ScanNet dataset path structure
        dataset_config = {
            "input_path": os.path.join(project_root, "scannet_data", "scans", scene),
            "H": c['H'],
            "W": c['W'],
            "fx": c['fx'], 
            "fy": c['fy'],
            "cx": c['cx'],
            "cy": c['cy'],
            "depth_scale": c['depth_scale'],
            "frame_limit": 4000,
            "distortion": [0,0,0,0,0],
            "crop_edge": c.get('crop_edge', 12),
            "depth_th": c.get('depth_th', 4.0),
            "eval_config_path": eval_config_path  # Pass correct config path
        }
        # 3. Instantiate ScanNet Loader
        loader = ScanNet(dataset_config)
    else:
        # Replica dataset (default)
        dataset_config = {
            "input_path": os.path.join(project_root, "Replica", scene),
            "H": c['H'],
            "W": c['W'],
            "fx": c['fx'], 
            "fy": c['fy'],
            "cx": c['cx'],
            "cy": c['cy'],
            "depth_scale": c['depth_scale'],
            "frame_limit": 2000,
            "distortion": [0,0,0,0,0],
            "crop_edge": 0
        }
        # 3. Instantiate Replica Loader
        loader = Replica(dataset_config)
    if len(loader) == 0:
        print(f"[Error] No frames found in {dataset_config['input_path']}")
        return None

    # 4. Setup Mask Generator (Internally constructed)
    weights_path = os.path.join(script_dir, "weights", "mobile_sam.pt") if script_dir else "weights/mobile_sam.pt"
    sam3_weights = os.path.join(script_dir, "weights", "sam3.pt") if script_dir else "weights/sam3.pt"

    mask_gen_config = {
        "precomputed": False,
        "masks_base_path": os.path.join(project_root, "masks"),
        "nms_iou_th": 0.8,
        "nms_score_th": 0.7,
        "model_type": mask_mode,
        "mobile_sam_weights": weights_path,
        "sam3_weights": sam3_weights
    }

    # For sam3_semantic mode, pass text prompts for text-prompted segmentation
    # If using LLM mode, select best prompt per class; otherwise use raw class names
    sam3_prompt_mapping = {}  # Track which prompts were selected for SAM3
    if mask_mode == "sam3_semantic":
        if prompt_mode == "llm" and llm_prompts:
            # Use configurable selection method to pick best prompt per class
            sam3_prompts = []
            for cn in loader.class_names:
                class_prompts = llm_prompts.get(cn, [cn])
                selected, selection_info = select_prompt_for_sam3(
                    class_prompts, 
                    method=sam3_selection_method,
                    all_class_prompts=llm_prompts
                )
                sam3_prompts.append(selected)
                sam3_prompt_mapping[cn] = {
                    "selected_prompt": selected,
                    "all_prompts": class_prompts,
                    **selection_info
                }
            print(f"[SAM3] Using LLM prompts (method: {sam3_selection_method}) for text-prompted segmentation")
        else:
            # Use raw class names (OpenAI/ensemble approach)
            sam3_prompts = loader.class_names
            for cn in loader.class_names:
                sam3_prompt_mapping[cn] = {
                    "selected_prompt": cn,
                    "all_prompts": [cn],
                    "method": "raw_class_name"
                }
            print(f"[SAM3] Using raw class names for text-prompted segmentation")
    else:
        sam3_prompts = None
    mask_gen = MaskGenerator(mask_gen_config, scene_name=scene, device=DEVICE, class_names=sam3_prompts)
    
    # Save SAM3 prompt selection to output directory
    if output_dir and sam3_prompt_mapping:
        sam3_prompts_path = os.path.join(output_dir, "sam3_selected_prompts.json")
        with open(sam3_prompts_path, 'w') as f:
            json.dump(sam3_prompt_mapping, f, indent=2)
        print(f"[SAM3] Prompt selection saved to: {sam3_prompts_path}")

    # 5. Setup Backbone & Text Embeddings
    backbone = _SEMANTIC_CACHE[sem_cfg['name']]
    
    scene_text_bank = load_replica_text_embeddings(
        backbone, 
        DEVICE, 
        loader.class_names, 
        cache_path=None,
        prompt_mode=prompt_mode,
        llm_prompts=llm_prompts
    )
    backbone.text_bank = scene_text_bank

    # 6. Pipeline Config
    pipeline_config = {
        "sem_name": sem_cfg['name'],
        "max_frames": 5000,
        "mapping": { 
            "map_every": 10, 
            "downscale_ratio": 3,
            "k_pooling": 3,            
            "max_frame_points": 80000,  # More points for accuracy
            "max_total_points": 500000, # Allow larger point cloud
            "voxel_size": 0.02          # 2cm voxel grid (finer)
        },
        "tracking": { "track_every": 1 },
        "semantic": { "segment_every": 10 },
        "track_th": 75,
        "th_centroid": 1.5,
        "th_cossim": 0.80,
        "match_distance_th": 0.03,
        "clip": { 
            "embed_type": "fused", 
            "k_top_views": 10, 
            "mask_res": 384,
            "dense_features": True
        },
        "sam": { "mask_res": 384, "nms_iou_th": 0.5, "nms_score_th": 0.8 },
        "detailed_logging": False,
        "vis": { "stream": False },
        "siglip_validation": {
            "enabled": True,
            "min_votes": 3,                # Require more votes for stable K-pool
            "min_keyframes": 5,            # Filter noisy single-frame objects
            "min_siglip_conf": 0.5,        # SigLIP must be this confident to override
            "temperature": 0.1,            # Softmax temperature for K-pool
            "synonym_threshold": 0.86,     # Text-embed cosine sim to detect near-synonyms
        },
        "output_dir": output_dir, # <--- Passed here
        "device": DEVICE
    }

    # 7. Instantiate Pipeline
    pipeline = ReplicaFusionPipeline(
        scene,
        pipeline_config,
        loader,
        output_dir=output_dir,
        device=DEVICE
    )

    # 8. Run
    result = pipeline.run(mask_gen, pipeline.classifier, detailed_logging_enabled=pipeline_config.get("detailed_logging", False))

    # Cleanup
    del pipeline
    del loader
    del mask_gen
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return result

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, "..")) 
    
    # =========================================================
    # CONFIGURATION CONTROL
    # =========================================================
    
    # 0. Dataset Selection
    # Options: 'replica' | 'scannet20' | 'scannet200'
    CURRENT_DATASET = 'scannet200'
    
    # 1. Prompt Mode
    # Options: 'ensemble' (OpenAI Standard) | 'handcrafted' (Your Custom Dict) | 'llm' (LLM-generated)
    CURRENT_PROMPT_MODE = 'ensemble'                                 

    # 2. Mask Generator Mode
    # Options: 'mobile_sam' (Fast, TinyViT) | 'sam3' (Heavy, Video) | 'sam3_semantic' (Text-prompted, needs sam3.pt)
    CURRENT_MASK_MODE = 'sam3_semantic'

    # 3. LLM Prompt Generation Config (only used when CURRENT_PROMPT_MODE = 'llm')
    LLM_CONFIG = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",     # HuggingFace model ID
        "num_prompts": 5,                              # Prompts per class
        "cache_prompts": True,                         # Cache generated prompts to disk
        "prompts_file": None,                          # Path to pre-generated prompts JSON (optional)
        "track_gpu_usage": True,                       # Track LLM GPU memory usage
        "force_regenerate": True,                      # Set True to regenerate ALL prompts (ignores cache)
        "sam3_selection_method": "first",              # Options: "first", "shortest", "longest", "random", "most_unique"
        "similarity_threshold": 0.80,                  # CLIP similarity threshold - prompts below this are rejected
        "max_retries": 3,                              # Max regeneration attempts per class if prompts fail quality gate
        "prompt_style": "comparative",         # Options: "standard", "cupl", "dclip", "comparative", "spatial",
    }

    # =========================================================

    # --- 1. Load Dataset-specific Config ---
    CLASSES = ["object"]
    SCENES = []
    
    if CURRENT_DATASET == "replica":
        eval_config_path = os.path.join(script_dir, "eval_info.yaml")
        cam_config_path = os.path.join(script_dir, "replica.yaml")
        SCENES = ["office1", "office4"]  # Default Replica scenes
        SCENES = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]
        # SCENES = ["office0", "office1", "office2", "office3", "office4"]
    elif CURRENT_DATASET == "scannet20":
        eval_config_path = os.path.join(script_dir, "scannet20.yaml")
        cam_config_path = os.path.join(script_dir, "scannet.yaml")  # Shared ScanNet camera config
        SCENES = ["scene0011_00", "scene0050_00", "scene0231_00", "scene0378_00", "scene0518_00"]
    elif CURRENT_DATASET == "scannet200":
        eval_config_path = os.path.join(script_dir, "scannet200.yaml")
        cam_config_path = os.path.join(script_dir, "scannet.yaml")  # Shared ScanNet camera config
        SCENES = ["scene0011_00", "scene0050_00", "scene0231_00", "scene0378_00", "scene0518_00"]
        SCENES = ["scene0011_00"]
    else:
        print(f"[Error] Unknown dataset: {CURRENT_DATASET}")
        return

    if os.path.exists(eval_config_path):
        print(f"[Global] Loading {CURRENT_DATASET} configuration from {eval_config_path}")
        with open(eval_config_path, 'r') as f:
            eval_cfg = yaml.safe_load(f)
            
        if "class_names_reduced" in eval_cfg:
            CLASSES = eval_cfg["class_names_reduced"]
            print(f"[Global] Loaded {len(CLASSES)} classes from YAML.")
        
    else:
        print(f"[Error] {eval_config_path} not found! Please ensure it exists.")
        return

    # --- 2. Load Camera Config ---
    cam_config = None
    if os.path.exists(cam_config_path):
        print(f"[Global] Loading camera config from {cam_config_path}")
        with open(cam_config_path, 'r') as f:
            cam_config = yaml.safe_load(f)
    else:
        print(f"[Global] Warning: {cam_config_path} not found. Using defaults.")

    # ---------------------------------------------------------
    # 2b. LLM Prompt Generation (if enabled)
    # ---------------------------------------------------------
    llm_prompts, llm_metrics, CURRENT_PROMPT_MODE = initialize_llm_prompts(
        CLASSES, LLM_CONFIG, script_dir, CURRENT_PROMPT_MODE
    )

    # ---------------------------------------------------------
    # 3. Load Semantic Backbones & Pre-compute Text Embeddings
    # ---------------------------------------------------------
    print("\n[Global] Initializing semantic backbones...")
    
    # Create a global cache for backbones if not exists
    global _SEMANTIC_CACHE
    if '_SEMANTIC_CACHE' not in globals(): _SEMANTIC_CACHE = {}

    # Switch between siglip and siglip2 by changing which config is in the list
    # To test SigLIP 2: use "siglip2" instead of "siglip"
    SEMANTIC_CONFIGS = [
        {
            "name": "siglip2", # Change to "siglip2" to test SigLIP 2
            "kind": "dual_encoder",
            "model_id": "google/siglip-so400m-patch14-224",
        },
    ]

    for cfg in SEMANTIC_CONFIGS:
        name = cfg['name']
        print(f" > Loading {name}...")
        
        # Helper to load backbone (assuming this function exists)
        backbone = load_semantic_backbone(name, device=DEVICE)
        
        # Cache name now includes the prompt mode
        cache_filename = f"replica_text_embs__{name}_{CURRENT_PROMPT_MODE}.pt"
        cache_path = os.path.join(script_dir, cache_filename) 
        
        print(f" > Loading text bank from: {cache_path}")
        
        # Helper to load embeddings (assuming this function exists)
        text_bank = load_replica_text_embeddings(
            backbone, 
            DEVICE, 
            CLASSES, 
            cache_path,
            prompt_mode=CURRENT_PROMPT_MODE,
            llm_prompts=llm_prompts
        )
        
        backbone.text_bank = text_bank 
        _SEMANTIC_CACHE[name] = backbone

    # ---------------------------------------------------------
    # 4. Run Scenes
    # ---------------------------------------------------------
    results = []
    
    for sem in SEMANTIC_CONFIGS:
        backbone_name = sem['name']
        
        # STRUCTURE: results / backbone / mask_mode / prompt_mode
        # Example: results/siglip/mobile_sam/ensemble/
        experiment_dir = os.path.join(
            RESULTS_DIR, 
            backbone_name, 
            CURRENT_MASK_MODE, 
            CURRENT_PROMPT_MODE
        )
        os.makedirs(experiment_dir, exist_ok=True)

        print(f"\n=== Running {backbone_name} | Mask: {CURRENT_MASK_MODE} | Output: {experiment_dir} ===")
        
        for scene in SCENES:
            try:
                # We define where this specific scene should save its detailed logs/maps
                scene_output_dir = os.path.join(experiment_dir, scene)
                os.makedirs(scene_output_dir, exist_ok=True)

                res = run_scene_unified(
                    scene, 
                    sem, 
                    project_root, 
                    cam_config=cam_config, 
                    prompt_mode=CURRENT_PROMPT_MODE,
                    mask_mode=CURRENT_MASK_MODE,
                    output_dir=scene_output_dir,
                    script_dir=script_dir,
                    llm_prompts=llm_prompts,
                    sam3_selection_method=LLM_CONFIG.get("sam3_selection_method", "first"),
                    dataset_type=CURRENT_DATASET
                )
                if res: 
                    results.append(res)

            except Exception as e:
                print(f"[Error] Failed on scene {scene}: {e}")
                import traceback
                traceback.print_exc()

        # ---------------------------------------------------------
        # 5. Save Summary (Per Backbone/Mask configuration)
        # ---------------------------------------------------------
        # Save the summary JSON inside the same experiment folder
        save_path = os.path.join(experiment_dir, "results_pipeline.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[Done] Group results saved to {save_path}")
        
        # Save LLM metrics if LLM mode was used
        if llm_metrics:
            save_llm_metrics(llm_metrics, experiment_dir)
        
        # Clear results list for the next backbone iteration (if you have multiple backbones)
        results = []

if __name__ == "__main__":
    main()