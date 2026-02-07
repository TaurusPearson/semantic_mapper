from __future__ import annotations
from typing import Dict, Any, Tuple, List
from PIL import Image 
import torchvision.transforms.functional as F
from copy import deepcopy
import numpy as np
import heapq
import torch
import torch.nn as nn
import os
class CLIPGenerator(nn.Module):
    """
    SOTA Tri-View Fusion Generator.
    Computes Global + Mask + BBox features and fuses them.
    Integrates with OVO's Top-K View logic by providing a robust fused embedding.
    """
    def __init__(self, config: Dict[str, Any], backbone, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        self.config = config
        self.backbone = backbone
        
        # --- NEW: Capture Tokenizer for SOTA Templates ---
        # Try to get tokenizer from backbone wrapper, or fallback to internal model
        self.tokenizer = getattr(backbone, "tokenizer", None)
        if self.tokenizer is None and hasattr(backbone, "model"):
             self.tokenizer = getattr(backbone.model, "tokenizer", None)
        
        # --- Fusion Weights ---
        self.w_masked = config.get("w_masked", 0.45)
        self.w_global = config.get("w_global", 0.1)
        self.w_bbox = 1.0 - self.w_masked - self.w_global
        
        # --- Backbone Dim Detection ---
        if hasattr(backbone, "dim"):
            self.clip_dim = backbone.dim
        elif hasattr(backbone, "model") and hasattr(backbone.model.visual, "output_dim"):
            self.clip_dim = backbone.model.visual.output_dim
        else:
            self.clip_dim = 512 
            
        self.mask_res = config.get("mask_res", 384) 
        
        # Ensure model is in eval mode
        self.backbone.model.to(self.device).eval()

        # --- SigLIP Parameters ---
        self.siglip = (self.backbone.name in ["siglip", "siglip2"])
        self.similarity_args = ()
        
        if self.siglip:
            # Extract learned scaling params
            logit_scale = getattr(self.backbone, "logit_scale", None)
            logit_bias = getattr(self.backbone, "logit_bias", None)
            
            if logit_scale is None:
                logit_scale = getattr(self.backbone.model, "logit_scale", None)
            if logit_bias is None:
                logit_bias = getattr(self.backbone.model, "logit_bias", None)

            if logit_scale is not None:
                if not isinstance(logit_scale, torch.Tensor):
                     logit_scale = torch.tensor([logit_scale], device=self.device)
                if logit_bias is None:
                    logit_bias = torch.tensor([0.0], device=self.device)
                elif not isinstance(logit_bias, torch.Tensor):
                    logit_bias = torch.tensor([logit_bias], device=self.device)
                self.similarity_args = (logit_scale.to(self.device), logit_bias.to(self.device))

    def cuda(self):
        self.backbone.model.to(self.device)

    def cpu(self):
        self.backbone.model.cpu()

    @torch.no_grad()
    def extract_clip(self, image: torch.Tensor, binary_maps: torch.Tensor, return_all=False) -> torch.Tensor:
        """
        Computes Tri-View Fusion (Global + Mask + BBox).
        Uses the helper functions (segmap2segimg) provided in SOTA code.
        """
        N = binary_maps.shape[0]
        if N == 0:
            return torch.empty((0, self.clip_dim), device=self.device)

        # 1. ENCODE GLOBAL (Scene Context)
        img_pil = Image.fromarray(image.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
        clip_g = self.backbone.encode_image(img_pil) # (1, D)
        clip_g = clip_g / (clip_g.norm(dim=-1, keepdim=True) + 1e-6)
        
        # 2. PREPARE CROPS (Using SOTA Helpers)
        seg_images = segmap2segimg(binary_maps, image, also_bbox=False, out_l=self.mask_res)
        bbox_images, _ = segmap2bboximg(binary_maps, image, bbox_margin=10, out_l=self.mask_res)

        # 3. CONVERT TO PIL & ENCODE
        def tensor_to_pil_list(t_imgs):
            return [Image.fromarray(t.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)) for t in t_imgs]

        pil_m = tensor_to_pil_list(seg_images)
        pil_b = tensor_to_pil_list(bbox_images)

        # Batch Encode
        if hasattr(self.backbone, "encode_image_batch"):
            clip_m = self.backbone.encode_image_batch(pil_m)
            clip_b = self.backbone.encode_image_batch(pil_b)
        else:
            clip_m = self.backbone.encode_image(pil_m)
            clip_b = self.backbone.encode_image(pil_b)
            
        # Normalize Views
        clip_m = clip_m / (clip_m.norm(dim=-1, keepdim=True) + 1e-6)
        clip_b = clip_b / (clip_b.norm(dim=-1, keepdim=True) + 1e-6)
        
        # Expand Global to match batch size
        clip_g_expanded = clip_g.repeat(N, 1)

        # 4. OUTPUT LOGIC
        if return_all:
            return torch.stack([clip_g_expanded, clip_m, clip_b], dim=1)
        
        # Weighted Fusion (SOTA Strategy)
        final_emb = (self.w_global * clip_g_expanded) + \
                    (self.w_masked * clip_m) + \
                    (self.w_bbox * clip_b)
                    
        final_emb = final_emb / (final_emb.norm(dim=-1, keepdim=True) + 1e-6)
        
        return final_emb

    @torch.no_grad()
    def get_txt_embedding(self, text_list: List[str]) -> torch.Tensor:
        """
        Compute text embeddings for a list of strings. 
        Mirrors SOTA logic: Tokenize -> Encode -> Normalize.
        """
        if self.tokenizer is None:
            # Fallback for models without exposed tokenizer
            print("[Warning] No tokenizer found. Returning zeros.")
            return torch.zeros((len(text_list), self.clip_dim), device=self.device)

        # Tokenize
        tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in text_list]).to(self.device)
        
        # Encode
        embeds = self.backbone.model.encode_text(tok_phrases)
        
        # Normalize
        embeds = embeds / (embeds.norm(dim=-1, keepdim=True) + 1e-6)
        return embeds

    @torch.no_grad()
    def get_embed_txt_similarity(self, image_embeds: torch.Tensor, txt_queries: List[str], templates: str | List[str] = ['{}']) -> torch.Tensor:
        """
        Computes similarity between Image Embeddings and Text Queries using Templates.
        Matches SOTA signature.
        """
        n_queries = len(txt_queries)
        
        # Pre-allocate text embeddings tensor
        # Shape: [N_Classes, D]
        txt_embeds = torch.zeros((n_queries, image_embeds.shape[1]), device=self.device)
        
        if isinstance(templates, str):
            templates = [templates]
            
        # Format queries with templates (e.g., "a photo of a {class}")
        queries_formatted = [[template.format(query) for template in templates] for query in txt_queries]

        # Compute Ensembled Text Embeddings
        for j in range(n_queries):
            # Encode all templates for this specific class/query
            # Mean pooling across templates -> Normalize
            embed = self.get_txt_embedding(queries_formatted[j]).mean(0, keepdim=True).float()
            txt_embeds[j] = torch.nn.functional.normalize(embed, p=2, dim=-1)

        # --- Compute Similarity ---
        # image_embeds: [N_Objects, D]
        # txt_embeds:   [N_Classes, D]
        
        # --- FIX: Match SOTA order (Objects x Classes) ---
        # We want [N_Objects, N_Classes] so OVO.classify_instances' argmax(dim=1) works correctly.
        sim_map = image_embeds @ txt_embeds.T
        # -------------------------------------------------
        
        # 2. Apply SigLIP/CLIP Scaling
        if self.siglip and len(self.similarity_args) == 2:
            scale, bias = self.similarity_args
            # SigLIP Formula: sigmoid( dot * exp(scale) + bias )
            sim_map = sim_map * scale.exp() + bias
            return torch.sigmoid(sim_map)
            
        else:
            # Standard CLIP
            return sim_map * 100.0
# =============================================================================
# SOTA HELPER FUNCTIONS (Geometry + SAM Utils)
# =============================================================================

def mask2segmap(masks: np.ndarray, image: np.ndarray, sort: bool = True) -> Tuple[np.ndarray, np.ndarray] :
    if sort:
        masks = heapq.nlargest(len(masks), masks, key= lambda x: x['stability_score'])  

    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    binary_maps = []
    for i, mask in enumerate(masks):
        binary_maps.append(mask['segmentation'])
        seg_map_mask = mask['segmentation'].copy()
        if sort:
            mask_overlap = np.logical_and(seg_map>-1,seg_map_mask)
            seg_map_mask[mask_overlap] = False
        seg_map[seg_map_mask] = i 
    
    binary_maps = np.stack(binary_maps) if len(binary_maps) > 0 else np.array([])
    return seg_map, binary_maps

def segmap2segimg(binary_map: torch.Tensor, image: torch.Tensor, also_bbox: bool, bbox_margin: int = 50, out_l: int = 224) -> torch.Tensor:
    seg_imgs = []
    
    bboxes_xyxy = batched_mask_to_box(binary_map)
    bboxes_xyhw = batched_box_xyxy_to_xywh(bboxes_xyxy)
    
    for i in range(binary_map.shape[0]):
        # Hardcode also_bbox=False because we handle bbox separately in extract_clip
        padded_img = seg_img_from_image(binary_map[i], bboxes_xyhw[i], image, also_bbox, bbox_margin, out_l)
        seg_imgs.append(padded_img)

    seg_imgs = torch.stack(seg_imgs, axis=0) # b,3,H,W
    return seg_imgs

def segmap2bboximg(binary_map: torch.Tensor, image: torch.Tensor,  bbox_margin: int = 50, out_l: int = 224) -> Tuple[torch.Tensor, torch.Tensor]:
    seg_imgs = []
    bmaps = []
    if len(binary_map)>0:
        bboxes_xyxy = batched_mask_to_box(binary_map)
        bboxes_xyhw = batched_box_xyxy_to_xywh(bboxes_xyxy)
        
        for i in range(binary_map.shape[0]):
            padded_img, bmap = bbox_img_from_image(binary_map[i], bboxes_xyhw[i], image, bbox_margin, out_l)
            seg_imgs.append(padded_img)
            bmaps.append(bmap)
 
        seg_imgs = torch.stack(seg_imgs, axis=0) # b,3,H,W
        bmaps = torch.stack(bmaps, axis=0)

    return seg_imgs, bmaps

def bbox_img_from_image(mask: torch.Tensor, bbox: torch.Tensor, image: torch.Tensor, bbox_margin: int = 50, size: int = 224) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_crop = get_bbox_img(bbox, image, bbox_margin)
    raw_mask = get_bbox_img(bbox, mask[None], bbox_margin)
    
    padded_crop = pad_img(raw_crop)
    padded_mask = pad_img(raw_mask)
    
    bbox_img = F.resize(padded_crop, (size, size), interpolation=F.InterpolationMode.BICUBIC)
    bmap = F.resize(padded_mask, (size, size), interpolation=F.InterpolationMode.NEAREST)
    return bbox_img, bmap

def seg_img_from_image(mask: torch.Tensor, bbox: torch.Tensor, image: torch.Tensor, also_bbox: bool, bbox_margin: int = 50, size: int = 224) -> torch.Tensor :
    seg_img = get_seg_img(mask, bbox, image)
    padded_img = pad_img(seg_img)
    padded_img = F.resize(padded_img, (size,size), interpolation=F.InterpolationMode.BICUBIC)
    
    if also_bbox:
        bbox_img, _ = bbox_img_from_image(mask, bbox, image, bbox_margin, size)
        padded_img = torch.concatenate([padded_img, bbox_img], axis=0)
        
    return padded_img

def get_seg_img(mask: torch.Tensor, bbox: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    x,y,w,h = bbox
    seg_img = torch.zeros((3, h, w), dtype = image.dtype, device=image.device)
    seg_img[:,mask[y:y+h, x:x+w]] = image[..., y:y+h, x:x+w][:,mask[y:y+h, x:x+w]].clone()
    return seg_img

def get_bbox_img(bbox: Tuple[int, int, int ,int], image: torch.Tensor, bbox_margin: int) ->  torch.Tensor:
    x, y, w, h = increase_bbox_by_margin(bbox, bbox_margin, image.shape[-2:])
    bbox_img = image[..., y:y+h, x:x+w].clone()
    return bbox_img

def pad_img(img: torch.Tensor) -> torch.Tensor:
    c, h, w = img.shape
    biggest_side = max(w,h)
    pad = torch.zeros((c, biggest_side, biggest_side), dtype=img.dtype, device = img.device)
    if h > w:
        pad[...,(h-w)//2:(h-w)//2 + w] = img
    else:
        pad[:,(w-h)//2:(w-h)//2 + h,:] = img
    return pad

def increase_bbox_by_margin(bbox: Tuple[int, int, int ,int], margin: int, img_shape: Tuple[int, int]) ->  Tuple[int, int, int ,int]:
    x, y, w, h = bbox
    H, W = img_shape
    x_new = max(0, x - margin)
    y_new = max(0, y - margin)
    w_new = min(W - x_new, (x + w + margin) - x_new)
    h_new = min(H - y_new, (y + h + margin) - y_new)
    return (x_new, y_new, w_new, h_new)

def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]
    return out

def batched_box_xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    box_xywh = deepcopy(box_xyxy)
    box_xywh[:,2] = box_xywh[:,2] - box_xywh[:,0]
    box_xywh[:,3] = box_xywh[:,3] - box_xywh[:,1]
    return box_xywh

def load_sam(config: Dict[str, Any], device: str = "cuda"): # Removed type hint for simplicity, or ensure you import all return types
    """ Load SAM, MobileSAM, or SAM2 model
    """
    sam_version = config.get("sam_version","2.1")
    sam_encoder = config.get("sam_encoder","hiera_l")
    checkpoint_path = os.path.join(config["sam_ckpt_path"], config["checkpoint_file"])
    
    sam_generator_class = None
    
    # --- MobileSAM Logic ---
    if sam_version == "mobile":
        from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry # Local Import
        sam_generator_class = SamAutomaticMaskGenerator

        if not os.path.exists(checkpoint_path):
             raise FileNotFoundError(f"MobileSAM checkpoint not found: {os.path.abspath(checkpoint_path)}")
        
        sam = sam_model_registry[sam_encoder](checkpoint=checkpoint_path).to(device).eval()
        sam_config = {
            "points_per_side": config.get("points_per_side", 32),
            "pred_iou_thresh": config.get("nms_iou_th", 0.88),
            "stability_score_thresh": config.get("stability_score_th", 0.95),
            "min_mask_region_area": config.get("min_mask_region_area", 100),
        }

    # --- SAM V1 Logic (if different from MobileSAM) ---
    elif sam_version == "":
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry # Local Import
        sam_generator_class = SamAutomaticMaskGenerator

        if not os.path.exists(checkpoint_path):
            # You may want a different message here for full SAM checkpoint
            raise FileNotFoundError(f"SAM v1 checkpoint not found: {os.path.abspath(checkpoint_path)}")

        sam = sam_model_registry[sam_encoder](checkpoint=checkpoint_path).to(device).eval()
        sam_config = {
            "points_per_side": config.get("points_per_side", 32),
            "pred_iou_thresh": config.get("nms_iou_th", 0.8),
            "stability_score_thresh": config.get("stability_score_th", 0.85),
            "min_mask_region_area": config.get("min_mask_region_area", 100),
        }

    # --- SAM 2 Logic (Using original SAM2 class name) ---
    else: # SAM 2.1
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator # Use original name
        sam_generator_class = SAM2AutomaticMaskGenerator # Assign the class object

        model_cfg = os.path.join("configs",f"sam{sam_version}",f"sam{sam_version}_{sam_encoder}.yaml")
        sam = build_sam2(model_cfg, checkpoint_path, device=device, mode="eval", apply_postprocessing=False)
        sam_config = {
            "points_per_side":config.get("points_per_side",32),
            "pred_iou_thresh": config.get("nms_iou_th",0.8),
            "stability_score_thresh": config.get("stability_score_th",0.95),
            "min_mask_region_area": config.get("min_mask_region_area", 0),
            "use_m2m": config.get("use_m2m", False),
        }
        
    # Instantiate the correct generator class found above
    if sam_generator_class is None:
         raise ValueError(f"Unknown sam_version: {sam_version}")

    mask_generator = sam_generator_class(
        model=sam,
        **sam_config
    )
    return mask_generator

def masks_update(*args, **kwargs) -> Tuple[np.ndarray]:
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))
        
        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        
        # USE RENAME: Call filter_masks, passing the Tensor directly (not a lambda)
        masks_lvl = filter_masks(keep_mask_nms, masks_lvl)
        
        masks_new += (masks_lvl,)
    return masks_new

def filter_masks(keep: torch.Tensor, masks_result) -> List[np.ndarray]: 
    # This helper expects a Tensor of indices to keep
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.7, score_thr: float = 0.1, inner_thr: float = 0.2, **kwargs) -> torch.Tensor:
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l
    selected_idx = idx[keep]
    return selected_idx
