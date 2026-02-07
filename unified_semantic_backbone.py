import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from PIL import Image
import open_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SemanticBackbone:
    name: str               # "clip" or "siglip"
    model: torch.nn.Module  # open_clip model
    preprocess: object      # torchvision.transforms.Compose
    tokenizer: object       # open_clip tokenizer
    device: str
    
    # We add this field to allow manual assignment, 
    # but give it a default of None so initialization doesn't fail.
    embed_dim: int = field(default=None, init=False)

    # ------------------------ IMAGE ENCODING ------------------------ #
    @torch.no_grad()
    def encode_image(self, image_rgb):
        """
        Return (1, D) L2-normalised image embedding.
        """
        if isinstance(image_rgb, np.ndarray):
            pil = Image.fromarray(image_rgb.astype("uint8"))
        elif isinstance(image_rgb, Image.Image):
            pil = image_rgb
        else:
            raise ValueError(f"encode_image expects np.ndarray or PIL.Image, got {type(image_rgb)}")

        # open_clip preprocess -> (3, H, W), normalized
        img_tensor = self.preprocess(pil).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        feats = self.model.encode_image(img_tensor)  # (1, D)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
        return feats  # (1, D)

    # ------------------------ BATCH IMAGE ENCODING ------------------------ #
    @torch.no_grad()
    def encode_image_batch(self, images):
        if not isinstance(images, list):
            raise ValueError(f"encode_image_batch expects list, got {type(images)}")

        processed = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil = Image.fromarray(img.astype("uint8"))
            elif isinstance(img, Image.Image):
                pil = img
            else:
                raise ValueError(f"Unexpected image type: {type(img)}")

            tensor = self.preprocess(pil)
            processed.append(tensor)

        img_batch = torch.stack(processed, dim=0).to(self.device)
        feats = self.model.encode_image(img_batch)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
        return feats

    # ------------------------ TEXT ENCODING ------------------------- #
    @torch.no_grad()
    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        tokens = self.tokenizer(texts)
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.to(self.device)

        feats = self.model.encode_text(tokens)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
        return feats
    
    # ------------------------ BATCH TEXT ENCODING ------------------- #
    # Added this helper to match the API call in replica_text_encoder
    @torch.no_grad()
    def encode_text_batch(self, texts):
        return self.encode_text(texts)
# -------------------------------------------------------------------------
# Loader: CLIP + SigLIP from open_clip (OVO-style)
# -------------------------------------------------------------------------
def load_semantic_backbone(name: str, device: str = DEVICE) -> SemanticBackbone:
    """
    Load a CLIP/SigLIP backbone using open_clip.
    Manually attaches 'embed_dim' to the wrapper to prevent AttributeError in Classifier.

    Returns:
        SemanticBackbone(name, model, preprocess, tokenizer, device)
    """
    name = name.lower()

    if name == "siglip":
        # Same family as OVO: timm SigLIP via open_clip
        # Architecture: ViT-SO400M-14-SigLIP-384
        card = "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
        model, preprocess = open_clip.create_model_from_pretrained(
            card,
            precision="fp32",
        )
        tokenizer = open_clip.get_tokenizer(card)
        model = model.to(device).eval()
        print("[Backbone] Loaded SigLIP via open_clip:", card)
        
        wrapper = SemanticBackbone("siglip", model, preprocess, tokenizer, device)
        
        # --- FIX: Manually attach embed_dim ---
        # SigLIP SO400M output dimension is 1152
        wrapper.embed_dim = 1152 
        
        return wrapper

    if name == "siglip2":
        # SigLIP 2: Improved version with better zero-shot performance
        # Architecture: ViT-SO400M-14-SigLIP2-378 (378px, same params as siglip v1)
        card = "hf-hub:timm/ViT-SO400M-14-SigLIP2-378"
        model, preprocess = open_clip.create_model_from_pretrained(
            card,
            precision="fp32",
        )
        tokenizer = open_clip.get_tokenizer(card)
        model = model.to(device).eval()
        print("[Backbone] Loaded SigLIP2 via open_clip:", card)
        
        wrapper = SemanticBackbone("siglip2", model, preprocess, tokenizer, device)
        
        # SigLIP2 SO400M output dimension is 1152 (same as v1)
        wrapper.embed_dim = 1152 
        
        return wrapper

    if name == "clip":
        # Baseline: ViT-L/14, LAION2B
        card = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
        model, preprocess = open_clip.create_model_from_pretrained(
            card,
            precision="fp32",
        )
        tokenizer = open_clip.get_tokenizer(card)
        model = model.to(device).eval()
        print("[Backbone] Loaded CLIP via open_clip:", card)
        
        wrapper = SemanticBackbone("clip", model, preprocess, tokenizer, device)
        
        # --- FIX: Manually attach embed_dim ---
        # CLIP ViT-L/14 output dimension is 768
        wrapper.embed_dim = 768 
        
        return wrapper

    raise ValueError(f"Unknown semantic backbone: {name}")