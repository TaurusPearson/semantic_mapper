# replica_text_encoder.py

import os
import torch
from tqdm import tqdm
from replica_prompts import get_prompts, validate_prompt_keys

# -----------------------------------------------------
# Compute embeddings for all Replica semantic classes
# -----------------------------------------------------

def compute_replica_text_embeddings(backbone, device, all_labels):
    """
    Compute per-label text embeddings using the provided SemanticBackbone.
    Returns:
        dict(label -> (D,) torch.Tensor normalized)
    """
    backbone.model.to(device)
    text_emb_dict = {}


    for label in all_labels:
        prompts = get_prompts(label)

        if len(prompts) == 0:
            # fallback prompt
            prompts = [f"a photo of a {label}"]

        # Encode all prompts → (N, D)
        emb = backbone.encode_text(prompts)

        # average → (D,)
        emb = emb.mean(dim=0)

        # normalize
        emb = emb / (emb.norm() + 1e-6)

        text_emb_dict[label] = emb.detach().cpu()

    return text_emb_dict

def load_replica_text_embeddings(backbone, device, class_names, cache_path=None, prompt_mode='ensemble'):
    """
    Loads text embeddings with a selectable prompt mode ('handcrafted' or 'ensemble').
    """
    
    # 1. STRICT VALIDATION (Respects mode)
    print(f" > Validating prompt dictionary against {len(class_names)} dataset classes (Mode: {prompt_mode})...")
    validate_prompt_keys(class_names, mode=prompt_mode)

    # 2. Check Cache
    # Note: We assume the cache file corresponds to the requested mode. 
    # If you switch modes often, consider adding the mode to the filename.
    if cache_path is not None and os.path.exists(cache_path):
        print(f" > Loading text embeddings from cache: {cache_path}")
        text_bank = torch.load(cache_path, map_location=device)
        
        if isinstance(text_bank, dict):
            missing = [c for c in class_names if c not in text_bank]
            if not missing:
                return text_bank
            print(f" > Cache outdated (missing {len(missing)} classes). Recomputing...")
    
    print(f" > Computing fresh SOTA text embeddings for {len(class_names)} classes...")
    
    backbone.model.to(device)
    text_bank = {}

    with torch.no_grad():
        for label in tqdm(class_names, desc=f"Encoding ({prompt_mode})"):
            
            # PASS THE MODE HERE
            prompts = get_prompts(label, mode=prompt_mode)
            
            if not prompts:
                prompts = [f"a photo of a {label}"]

            text_features = None
            
            if hasattr(backbone, 'encode_text_batch'):
                text_features = backbone.encode_text_batch(prompts)
            elif hasattr(backbone, 'encode_text'):
                text_features = backbone.encode_text(prompts)
            elif hasattr(backbone, 'model') and hasattr(backbone, 'tokenizer'):
                tokenizer = backbone.tokenizer
                model = backbone.model
                text_tokens = tokenizer(prompts).to(device)
                text_features = model.encode_text(text_tokens)
            else:
                raise AttributeError("Backbone wrapper missing encode_text method")

            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
            avg_feature = text_features.mean(dim=0)
            avg_feature = avg_feature / (avg_feature.norm() + 1e-6)
            
            text_bank[label] = avg_feature

    if cache_path is not None:
        torch.save(text_bank, cache_path)
        print(f" > Saved text embeddings to {cache_path}")

    return text_bank