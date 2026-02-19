"""
llm_prompt_generator.py

LLM-based prompt generation for zero-shot segmentation.
Generates simple noun phrases as recommended by SAM3 paper.

Supports:
- Text-only LLMs (Qwen2.5, Llama3, etc.)
- Future: MLLMs with image input
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = Path(__file__).parent / "llm_prompt_cache"

# SAM3-style prompt: simple noun phrases, no spatial context
LLM_SYSTEM_PROMPT = """You are generating text prompts for zero-shot image segmentation in INDOOR SCENES.

CRITICAL: You MUST respond ONLY in English. Do NOT use Chinese, Mandarin, or any other language.

CONTEXT: These prompts are for segmenting objects in indoor environments like offices, living rooms, 
bedrooms, and residential spaces. Objects include furniture, fixtures, and common household items.

Your task: Generate simple English noun phrases that describe the TYPICAL visual appearance of objects 
as they appear in normal indoor settings.

Rules:
- Keep phrases SHORT (2-5 words)
- MUST include the object name or a close synonym (e.g., "wall" -> "white wall", NOT "painted surface")
- Focus on COMMON visual features: typical colors, materials, textures
- Prefer GENERIC descriptions over unusual variants (e.g., "white wall" over "graffiti wall")
- NO spatial context (don't mention "on the wall", "next to", etc.)
- NO articles at the start (say "wooden desk" not "a wooden desk")
- ENGLISH ONLY - no other languages

DISAMBIGUATION RULES (to avoid confusing similar classes):
- WALL vs PANEL: "wall" = large flat vertical room surface; "panel" = small mounted control/display panel
- WALL vs PIPE: "wall" = flat painted surface; "pipe" = cylindrical metal tube for plumbing/HVAC
- PILLOW vs CUSHION: "pillow" = soft head rest on bed; "cushion" = decorative seat pad on sofa/chair  
- MONITOR vs TV-SCREEN: "monitor" = computer display on desk; "tv-screen" = large entertainment display
- BLANKET vs CLOTH: "blanket" = large bed/sofa covering; "cloth" = small fabric piece like towel/napkin
- DESK vs TABLE: "desk" = work surface with drawers/storage; "table" = simple flat surface for dining/coffee

Examples of GOOD prompts (distinctive, avoid confusion):
- "flat painted wall surface" (clearly a wall, not a panel)
- "computer monitor screen" (clearly a monitor, not TV)
- "bed pillow" (clearly a pillow, not cushion)
- "sofa seat cushion" (clearly a cushion, not pillow)
- "office work desk" (clearly a desk, not table)
- "plumbing pipe tube" (clearly a pipe, not wall edge)

Examples of BAD prompts:
- "white panel" (too generic, could match walls)
- "metal pipe" (too generic, could match wall trim)
- "soft cushion" / "soft pillow" (too similar, causes confusion)
- "flat screen" (ambiguous: monitor or TV?)
- Any text containing non-English characters
"""

# =============================================================================
# PROMPT TEMPLATES - Literature References
# =============================================================================

# -----------------------------------------------------------------------------
# STANDARD TEMPLATE (Baseline)
# Based on: Basic CLIP prompt engineering (Radford et al., 2021)
# Reference: "Learning Transferable Visual Models From Natural Language Supervision"
# URL: https://arxiv.org/abs/2103.00020
# Approach: Simple noun phrases with class name grounding
# -----------------------------------------------------------------------------
LLM_USER_TEMPLATE_STANDARD = """Generate {num_prompts} simple noun phrases for the indoor object class: "{class_name}"

Requirements:
1. Each phrase MUST include "{class_name}" or a very close synonym
2. Describe typical appearance in offices/homes (not unusual variants)
3. Focus on visual features: color, texture, material

One phrase per line, no numbering.
IMPORTANT: Respond ONLY in English."""

# -----------------------------------------------------------------------------
# CuPL TEMPLATE
# Based on: "What does a platypus look like?" (Pratt et al., ICLR 2023)
# Reference: "Generating customized prompts for zero-shot image classification"
# URL: https://arxiv.org/abs/2209.03320
# Approach: Question-based prompts that ask LLM to describe visual appearance
# Key insight: Use diverse question templates, high temperature for variety
# -----------------------------------------------------------------------------
LLM_USER_TEMPLATE_CUPL = """Answer these questions about "{class_name}" with short descriptive phrases:

1. What does a {class_name} look like?
2. How can you visually identify a {class_name}?
3. Describe the typical appearance of a {class_name}:
4. What are the visual characteristics of a {class_name}?
5. Describe an image of a {class_name}:

For each answer:
- Give ONE short phrase (3-4 words)
- MUST include "{class_name}" in the phrase
- Focus on visual appearance only

One phrase per line, no numbering.
IMPORTANT: Respond ONLY in English."""

# -----------------------------------------------------------------------------
# DCLIP TEMPLATE (Descriptor-based)
# Based on: "Visual Classification via Description from Large Language Models" (Menon & Vondrick, 2023)
# Reference: DCLIP - GPT-Descriptor Extended CLIP
# URL: https://arxiv.org/abs/2210.07183
# Approach: Generate descriptors as "A {class}, which has {attribute}"
# Note: Useful for MLLMs that can verify visual attributes against images
# Limitation: May produce non-visual descriptors (taste, smell) - needs filtering
# -----------------------------------------------------------------------------
LLM_USER_TEMPLATE_DCLIP = """Generate {num_prompts} visual descriptors for "{class_name}" in the format:
"{class_name}, which has [visual attribute]"

Describe ONLY visual features:
- MATERIAL: What is it made of? (fabric, leather, wood, metal, plastic, glass)
- SHAPE: What is its form? (rectangular, cylindrical, flat, curved, padded)
- PARTS: What components does it have? (legs, armrests, backrest, seat, frame)
- TEXTURE: What is its surface? (smooth, quilted, tufted, matte, glossy)
- COLOR: Common colors for this object

Rules:
1. Each descriptor MUST start with "{class_name}, which"
2. Describe ONLY what you can SEE (no smell, taste, sound, function)
3. Be specific and concrete

Examples:
- "{class_name}, which has a wooden frame"
- "{class_name}, which has four metal legs"
- "{class_name}, which has a padded seat"

One descriptor per line, no numbering.
IMPORTANT: Respond ONLY in English."""

# -----------------------------------------------------------------------------
# COMPARATIVE DESCRIPTORS TEMPLATE
# Based on: "Enhancing Visual Classification using Comparative Descriptors" (2024)
# URL: https://arxiv.org/abs/2411.05357
# Approach: Generate descriptors that distinguish target class from similar classes
# Key insight: Most misclassifications occur between semantically similar classes
# Method: Ask "What features distinguish {target} from {similar}?"
# -----------------------------------------------------------------------------
LLM_USER_TEMPLATE_COMPARATIVE = """Generate {num_prompts} distinguishing phrases for "{class_name}"

CRITICAL: "{class_name}" is often confused with: {confusing_classes}

For each phrase, answer: "What visual features distinguish a {class_name} from {confusing_classes}?"

Focus on VISUAL DIFFERENCES:
- Shape differences (e.g., rectangular vs square, tall vs wide)
- Size differences (e.g., small vs large, thin vs thick)
- Part differences (e.g., has legs vs no legs, has armrests vs none)
- Material differences (e.g., fabric vs leather, wood vs metal)

Rules:
1. Each phrase MUST include "{class_name}"
2. Each phrase MUST highlight a feature that {confusing_classes} do NOT have
3. Be specific and concrete
4. Must be short (3-4 words)

One phrase per line, no numbering.
IMPORTANT: Respond ONLY in English."""

# -----------------------------------------------------------------------------
# CuPL + COMPARATIVE TEMPLATE (Combined)
# Based on: CuPL (Pratt et al., 2023) + Comparative Descriptors (2024)
# Approach: Question-based prompts with explicit disambiguation from similar classes
# Best for: Classes with known confusion pairs where visual distinction is critical
# -----------------------------------------------------------------------------
LLM_USER_TEMPLATE_CUPL_COMPARATIVE = """Describe "{class_name}" with phrases that distinguish it from similar objects.

CRITICAL: "{class_name}" is often confused with: {confusing_classes}

Answer these questions with distinguishing visual features:
1. What does a {class_name} look like that a {confusing_classes} does NOT?
2. What visual parts does a {class_name} have that {confusing_classes} lacks?
3. What shape/size makes a {class_name} different from {confusing_classes}?
4. What material/texture distinguishes a {class_name} from {confusing_classes}?
5. How would you visually tell a {class_name} apart from {confusing_classes}?

Rules:
1. Each phrase MUST include "{class_name}"
2. Focus on VISUAL differences only (not location or function)
3. Be specific: "wooden legs" not just "legs"

One phrase per line, no numbering.
IMPORTANT: Respond ONLY in English."""

# -----------------------------------------------------------------------------
# DCLIP + COMPARATIVE TEMPLATE (For future MLLM use)
# Based on: DCLIP (Menon & Vondrick, 2023) + Comparative Descriptors (2024)
# Approach: Descriptor format with disambiguation - ideal for MLLMs that can
#           verify "which has X" attributes against actual image content
# -----------------------------------------------------------------------------
LLM_USER_TEMPLATE_DCLIP_COMPARATIVE = """Generate {num_prompts} visual descriptors for "{class_name}" that distinguish it from similar objects.

CRITICAL: "{class_name}" is often confused with: {confusing_classes}

Format: "{class_name}, which has [distinguishing visual feature]"

For each descriptor, the feature MUST be something that {confusing_classes} does NOT have.

Think about:
- What SHAPE does {class_name} have that {confusing_classes} doesn't?
- What PARTS does {class_name} have that {confusing_classes} lacks?
- What MATERIAL is {class_name} typically made of vs {confusing_classes}?
- What SIZE is {class_name} compared to {confusing_classes}?

Examples (if distinguishing chair from sofa):
- "chair, which has four separate legs" (sofa has a base)
- "chair, which has a single seat" (sofa has multiple seats)

One descriptor per line, no numbering.
IMPORTANT: Respond ONLY in English."""

# -----------------------------------------------------------------------------
# SPATIAL TEMPLATE
# Based on: Indoor scene understanding and object co-occurrence patterns
# Reference: Scene context for semantic segmentation (no single paper - common practice)
# Approach: Use spatial/location context to disambiguate similar objects
# Key insight: Objects in indoor scenes have typical locations that distinguish them
# Note: Useful when visual attributes alone are insufficient for disambiguation
# -----------------------------------------------------------------------------
LLM_USER_TEMPLATE_SPATIAL = """Generate {num_prompts} location-aware phrases for: "{class_name}"

Describe WHERE this object is typically found in indoor scenes:
- ROOM TYPE: Which rooms? (bedroom, office, living room, kitchen)
- POSITION: Where in the room? (against wall, center, corner, near window)
- ADJACENT TO: What objects is it usually near? (next to desk, under table, on bed)
- HEIGHT: At what level? (floor-level, waist-height, eye-level, ceiling)

Rules:
1. Each phrase MUST contain "{class_name}"
2. Include spatial context that helps identify it
3. Focus on TYPICAL indoor locations

Examples:
- "{class_name} against the wall"
- "{class_name} in the corner of the room"  
- "{class_name} near the window"

One phrase per line, no numbering.
IMPORTANT: Respond ONLY in English."""

# -----------------------------------------------------------------------------
# SPATIAL + COMPARATIVE TEMPLATE
# Based on: Spatial context + Comparative Descriptors (2024)
# Approach: Use location differences to distinguish confusing classes
# Best for: Objects distinguished primarily by WHERE they appear (pillow vs cushion)
# -----------------------------------------------------------------------------
LLM_USER_TEMPLATE_SPATIAL_COMPARATIVE = """Generate {num_prompts} location-aware phrases for "{class_name}" that distinguish it from similar objects.

CRITICAL: "{class_name}" is often confused with: {confusing_classes}

Focus on WHERE these objects differ:
- pillow is ON THE BED (for sleeping) vs cushion is ON THE SOFA (for sitting)
- monitor is ON A DESK (for work) vs tv-screen is ON A WALL or TV STAND (for entertainment)
- desk has DRAWERS and is in OFFICE vs table is in DINING ROOM with no storage

For "{class_name}" vs {confusing_classes}, think about:
- What ROOM is "{class_name}" typically in vs {confusing_classes}?
- What is "{class_name}" placed ON or NEAR vs {confusing_classes}?
- What FUNCTION location suggests? (work area, sleeping area, entertainment area)

Rules:
1. Each phrase MUST contain "{class_name}"
2. Each phrase MUST include a location/context that {confusing_classes} does NOT have
3. Be specific about the spatial relationship

One phrase per line, no numbering.
IMPORTANT: Respond ONLY in English."""


# =============================================================================
# DEFAULT FOCUS CLASSES (for filtered confusion matrix)
# =============================================================================

DEFAULT_FOCUS_CLASSES = [
    "table", "desk", "chair", "cabinet", "shelf",
    "wall", "ceiling", "panel", "floor", "door"
]

# Confusing class pairs - prompts for class A should NOT be similar to class B
# Format: {class_name: [list of confusing classes to check against]}
# Comprehensive mapping based on semantic analysis and observed confusions
CONFUSING_CLASS_PAIRS = {
    # SOFT FURNISHINGS (High confusion cluster)
    "pillow": ["cushion", "blanket", "cloth"],
    "cushion": ["pillow", "blanket", "cloth", "sofa"],
    "blanket": ["cloth", "comforter", "pillow", "cushion", "rug"],
    "comforter": ["blanket", "cloth", "bed"],
    "cloth": ["blanket", "pillow", "cushion", "rug"],
    "rug": ["floor", "cloth", "blanket"],
    
    # FLAT SURFACES (High confusion cluster)
    "wall": ["panel", "ceiling", "floor", "door"],
    "ceiling": ["wall", "panel"],
    "floor": ["rug", "wall"],
    "panel": ["wall", "door", "monitor", "tv-screen", "picture"],
    "door": ["panel", "cabinet", "wall"],
    
    # SCREENS/DISPLAYS (High confusion cluster)
    "monitor": ["tv-screen", "panel", "picture", "tablet"],
    "tv-screen": ["monitor", "panel", "picture"],
    "tablet": ["monitor", "picture", "book"],
    "picture": ["monitor", "panel", "clock", "tv-screen"],
    
    # FURNITURE - TABLES (High confusion cluster)
    "desk": ["table", "tv-stand", "shelf", "nightstand"],
    "table": ["desk", "tv-stand", "nightstand"],
    "tv-stand": ["desk", "table", "shelf", "cabinet"],
    "nightstand": ["desk", "table", "cabinet"],
    "shelf": ["desk", "cabinet", "tv-stand"],
    
    # FURNITURE - SEATING
    "chair": ["stool", "sofa", "bench"],
    "sofa": ["chair", "bed", "bench", "cushion"],
    "stool": ["chair", "bench"],
    "bench": ["stool", "sofa", "bed"],
    "bed": ["sofa", "bench", "blanket", "comforter"],
    
    # STORAGE
    "cabinet": ["door", "shelf", "desk", "nightstand"],
    "box": ["tissue-paper", "desk-organizer", "bin"],
    "basket": ["bin", "box", "pot"],
    "bin": ["basket", "box"],
    "desk-organizer": ["box", "tissue-paper", "shelf"],
    
    # SMALL OBJECTS
    "tissue-paper": ["box", "cloth", "book"],
    "bottle": ["vase", "candle", "pot"],
    "vase": ["bottle", "pot", "sculpture"],
    "pot": ["vase", "bowl", "bottle", "basket"],
    "bowl": ["pot", "plate", "basket"],
    "plate": ["bowl", "clock", "panel"],
    "candle": ["bottle", "vase", "sculpture"],
    "book": ["box", "tablet", "tissue-paper"],
    
    # ELECTRICAL
    "switch": ["wall-plug", "panel"],
    "wall-plug": ["switch", "panel"],
    "lamp": ["candle", "sculpture", "vase"],
    
    # STRUCTURAL
    "pipe": ["wall", "pillar", "panel"],
    "pillar": ["pipe", "wall", "door"],
    "vent": ["panel", "switch", "wall-plug"],
    "window": ["door", "panel", "blinds"],
    "blinds": ["window", "panel", "door"],
    
    # DECORATIVE
    "sculpture": ["vase", "lamp", "indoor-plant", "candle"],
    "indoor-plant": ["vase", "sculpture", "plant-stand"],
    "plant-stand": ["table", "stool", "shelf"],
    "clock": ["picture", "panel", "plate"],
    "camera": ["sculpture", "clock"],
}

# =============================================================================
# LLM PROMPT GENERATOR CLASS
# =============================================================================

class LLMPromptGenerator:
    """
    Generates segmentation prompts using a text-only LLM.
    Includes CLIP similarity quality gate to ensure prompt quality.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        use_cache: bool = True,
        num_prompts: int = 5,
        similarity_threshold: float = 0.7,
        max_retries: int = 3,
        prompt_style: str = "cupl",  # Options: "standard", "cupl", "dclip", "comparative", "spatial", "cupl_comparative", "dclip_comparative", "spatial_comparative"
    ):
        self.model_name = model_name
        self.device = device
        self.use_cache = use_cache
        self.num_prompts = num_prompts
        self.similarity_threshold = similarity_threshold
        self.max_retries = max_retries
        self.prompt_style = prompt_style
        # Auto-enable comparative mode if prompt_style includes "comparative"
        self.comparative_prompting = "comparative" in prompt_style
        self.model = None
        self.tokenizer = None
        
        # CLIP/SigLIP model for similarity validation
        self.clip_model = None
        self.clip_processor = None
        
        # Create cache directory
        CACHE_DIR.mkdir(exist_ok=True)
        
    def _get_cache_path(self, class_name: str) -> Path:
        """Get cache file path for a class."""
        model_hash = hashlib.md5(self.model_name.encode()).hexdigest()[:8]
        return CACHE_DIR / f"{class_name}_{model_hash}_{self.num_prompts}.json"
    
    def _load_from_cache(self, class_name: str) -> Optional[List[str]]:
        """Load prompts from cache if available."""
        if not self.use_cache:
            return None
        cache_path = self._get_cache_path(class_name)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return data.get('prompts', None)
        return None
    
    def _save_to_cache(self, class_name: str, prompts: List[str]):
        """Save prompts to cache."""
        cache_path = self._get_cache_path(class_name)
        with open(cache_path, 'w') as f:
            json.dump({
                'class_name': class_name,
                'model': self.model_name,
                'num_prompts': self.num_prompts,
                'prompts': prompts
            }, f, indent=2)
    
    def load_model(self):
        """Load the LLM model (lazy loading)."""
        if self.model is not None:
            return
        
        print(f"[LLMPromptGenerator] Loading model: {self.model_name}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print(f"[LLMPromptGenerator] Model loaded successfully")
            
        except Exception as e:
            print(f"[LLMPromptGenerator] Error loading model: {e}")
            raise
    
    def load_clip_model(self):
        """Load SigLIP model for similarity validation (lazy loading)."""
        if self.clip_model is not None:
            return
        
        print(f"[LLMPromptGenerator] Loading SigLIP for quality validation...")
        try:
            import open_clip
            import torch
            
            # Use open_clip's SigLIP (same as pipeline uses)
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'hf-hub:timm/ViT-SO400M-14-SigLIP2-378',
                device=self.device
            )
            self.clip_tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP2-378')
            self.clip_model.eval()
            print(f"[LLMPromptGenerator] SigLIP2 loaded for quality gate (open_clip)")
        except Exception as e:
            print(f"[LLMPromptGenerator] Warning: Could not load SigLIP: {e}")
            print(f"[LLMPromptGenerator] Quality gate disabled")
            self.similarity_threshold = 0.0  # Disable threshold if can't load
    
    def compute_similarity(self, prompt: str, class_name: str) -> float:
        """Compute cosine similarity between prompt and class name using SigLIP."""
        if self.clip_model is None:
            return 1.0  # Skip validation if model not loaded
        
        import torch
        
        class_text = class_name.replace("-", " ").replace("_", " ")
        
        with torch.no_grad():
            # Encode class name using open_clip tokenizer
            class_tokens = self.clip_tokenizer([class_text]).to(self.device)
            class_embed = self.clip_model.encode_text(class_tokens)
            class_embed = class_embed / class_embed.norm(dim=-1, keepdim=True)
            
            # Encode prompt
            prompt_tokens = self.clip_tokenizer([prompt]).to(self.device)
            prompt_embed = self.clip_model.encode_text(prompt_tokens)
            prompt_embed = prompt_embed / prompt_embed.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = (class_embed @ prompt_embed.T).item()
        
        return similarity
    
    def check_cross_class_similarity(self, prompt: str, target_class: str, margin: float = 0.05) -> Tuple[bool, Optional[str], float]:
        """
        Check if prompt is distinctive enough from confusing classes.
        
        Args:
            prompt: The generated prompt
            target_class: The class this prompt is for
            margin: Required margin - prompt must be at least this much more similar to target
            
        Returns:
            (is_valid, confusing_class, max_confusing_sim)
            - is_valid: True if prompt is distinctive
            - confusing_class: The class that's too similar (if any)
            - max_confusing_sim: Highest similarity to a confusing class
        """
        if self.clip_model is None:
            return True, None, 0.0
        
        # Get confusing classes for this target
        confusing_classes = CONFUSING_CLASS_PAIRS.get(target_class, [])
        if not confusing_classes:
            return True, None, 0.0
        
        # Compute similarity to target
        target_sim = self.compute_similarity(prompt, target_class)
        
        # Check against each confusing class
        max_confusing_sim = 0.0
        most_confusing_class = None
        
        for confusing_class in confusing_classes:
            sim = self.compute_similarity(prompt, confusing_class)
            if sim > max_confusing_sim:
                max_confusing_sim = sim
                most_confusing_class = confusing_class
        
        # Prompt is valid if target_sim > max_confusing_sim + margin
        is_valid = target_sim > max_confusing_sim + margin
        
        return is_valid, most_confusing_class, max_confusing_sim
    
    def _generate_raw_prompts(self, class_name: str) -> List[str]:
        """Generate raw prompts from LLM (internal method, no caching).
        
        Supports multiple prompt styles based on literature:
        - "standard": Basic CLIP-style noun phrases (Radford et al., 2021)
        - "cupl": Question-based prompts (Pratt et al., ICLR 2023)
        - "dclip": Descriptor format "X, which has Y" (Menon & Vondrick, 2023)
        - "comparative": Disambiguation from confusing classes (arXiv 2411.05357)
        - "spatial": Location/context-based prompts (indoor scene understanding)
        - "cupl_comparative": CuPL + disambiguation (combined)
        - "dclip_comparative": DCLIP + disambiguation (for MLLMs)
        - "spatial_comparative": Spatial + disambiguation (location-based distinction)
        """
        class_name_clean = class_name.replace("-", " ").replace("_", " ")
        
        # Get confusing classes if comparative mode is enabled
        confusing_classes = CONFUSING_CLASS_PAIRS.get(class_name, []) if self.comparative_prompting else []
        confusing_str = ", ".join(confusing_classes) if confusing_classes else ""
        
        # Select template based on prompt_style
        # For *_comparative styles: use comparative version if confusing classes exist, else fallback to base
        if self.prompt_style == "spatial_comparative":
            if confusing_classes:
                user_prompt = LLM_USER_TEMPLATE_SPATIAL_COMPARATIVE.format(
                    num_prompts=self.num_prompts,
                    class_name=class_name_clean,
                    confusing_classes=confusing_str
                )
                print(f"    [Spatial+Comparative] Confusing: {confusing_str}")
            else:
                user_prompt = LLM_USER_TEMPLATE_SPATIAL.format(
                    num_prompts=self.num_prompts,
                    class_name=class_name_clean
                )
                print(f"    [Spatial] No confusing classes, using base spatial")
        elif self.prompt_style == "cupl_comparative":
            if confusing_classes:
                user_prompt = LLM_USER_TEMPLATE_CUPL_COMPARATIVE.format(
                    num_prompts=self.num_prompts,
                    class_name=class_name_clean,
                    confusing_classes=confusing_str
                )
                print(f"    [CuPL+Comparative] Confusing: {confusing_str}")
            else:
                user_prompt = LLM_USER_TEMPLATE_CUPL.format(
                    num_prompts=self.num_prompts,
                    class_name=class_name_clean
                )
                print(f"    [CuPL] No confusing classes, using base CuPL")
        elif self.prompt_style == "dclip_comparative":
            if confusing_classes:
                user_prompt = LLM_USER_TEMPLATE_DCLIP_COMPARATIVE.format(
                    num_prompts=self.num_prompts,
                    class_name=class_name_clean,
                    confusing_classes=confusing_str
                )
                print(f"    [DCLIP+Comparative] Confusing: {confusing_str}")
            else:
                user_prompt = LLM_USER_TEMPLATE_DCLIP.format(
                    num_prompts=self.num_prompts,
                    class_name=class_name_clean
                )
                print(f"    [DCLIP] No confusing classes, using base DCLIP")
        elif self.prompt_style == "comparative":
            if confusing_classes:
                user_prompt = LLM_USER_TEMPLATE_COMPARATIVE.format(
                    num_prompts=self.num_prompts,
                    class_name=class_name_clean,
                    confusing_classes=confusing_str
                )
                print(f"    [Comparative] Confusing: {confusing_str}")
            else:
                user_prompt = LLM_USER_TEMPLATE_STANDARD.format(
                    num_prompts=self.num_prompts,
                    class_name=class_name_clean
                )
                print(f"    [Standard] No confusing classes, using base standard")
        elif self.prompt_style == "spatial":
            # Spatial/location-based prompts
            user_prompt = LLM_USER_TEMPLATE_SPATIAL.format(
                num_prompts=self.num_prompts,
                class_name=class_name_clean
            )
            print(f"    [Spatial] Location-based prompts")
        elif self.prompt_style == "cupl":
            # CuPL (Pratt et al., ICLR 2023)
            user_prompt = LLM_USER_TEMPLATE_CUPL.format(
                num_prompts=self.num_prompts,
                class_name=class_name_clean
            )
            print(f"    [CuPL] Question-based prompts")
        elif self.prompt_style == "dclip":
            # DCLIP (Menon & Vondrick, 2023)
            user_prompt = LLM_USER_TEMPLATE_DCLIP.format(
                num_prompts=self.num_prompts,
                class_name=class_name_clean
            )
            print(f"    [DCLIP] Descriptor format")
        else:
            # Standard/fallback (basic CLIP-style)
            user_prompt = LLM_USER_TEMPLATE_STANDARD.format(
                num_prompts=self.num_prompts,
                class_name=class_name_clean
            )
            print(f"    [Standard] Basic CLIP-style prompts")
        
        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse response into list of prompts
        return self._parse_response(response, class_name)
    
    def generate_prompts(self, class_name: str, force_regenerate: bool = False) -> List[str]:
        """
        Generate prompts for a single class with CLIP similarity quality gate.
        
        Each prompt must have cosine similarity > threshold with class name.
        Prompts that fail are regenerated (up to max_retries).
        
        Args:
            class_name: The object class to generate prompts for
            force_regenerate: If True, bypass cache
            
        Returns:
            List of generated prompt strings (all pass quality gate)
        """
        # Check cache first
        if not force_regenerate:
            cached = self._load_from_cache(class_name)
            if cached:
                return cached
        
        # Load models if needed
        self.load_model()
        if self.similarity_threshold > 0:
            self.load_clip_model()
        
        validated_prompts = []
        attempts = 0
        max_total_attempts = self.max_retries * self.num_prompts
        
        while len(validated_prompts) < self.num_prompts and attempts < max_total_attempts:
            # Generate a batch of prompts
            raw_prompts = self._generate_raw_prompts(class_name)
            attempts += 1
            
            for prompt in raw_prompts:
                if len(validated_prompts) >= self.num_prompts:
                    break
                    
                # Skip if already have this prompt
                if prompt in validated_prompts:
                    continue
                
                # Check similarity if threshold is set
                if self.similarity_threshold > 0:
                    similarity = self.compute_similarity(prompt, class_name)
                    
                    if similarity < self.similarity_threshold:
                        print(f"    ✗ '{prompt}' (sim: {similarity:.3f} < {self.similarity_threshold}) - REJECTED")
                        continue
                    
                    # Cross-class similarity check
                    is_distinctive, confusing_class, confusing_sim = self.check_cross_class_similarity(prompt, class_name)
                    
                    if is_distinctive:
                        validated_prompts.append(prompt)
                        print(f"    ✓ '{prompt}' (sim: {similarity:.3f})")
                    else:
                        print(f"    ✗ '{prompt}' (sim: {similarity:.3f}) - TOO SIMILAR TO '{confusing_class}' ({confusing_sim:.3f})")
                else:
                    # No threshold, accept all
                    validated_prompts.append(prompt)
        
        # If still not enough prompts, fall back to class name
        if len(validated_prompts) < self.num_prompts:
            fallback = class_name.replace("-", " ").replace("_", " ")
            print(f"    [Warning] Only {len(validated_prompts)} prompts passed quality gate, adding fallback: '{fallback}'")
            while len(validated_prompts) < self.num_prompts:
                if fallback not in validated_prompts:
                    validated_prompts.append(fallback)
                else:
                    # Add variant
                    validated_prompts.append(f"{fallback} object")
                    break
        
        # Cache results
        self._save_to_cache(class_name, validated_prompts)
        
        return validated_prompts
    
    def _is_english_only(self, text: str) -> bool:
        """Check if text contains only English/ASCII characters."""
        try:
            text.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False
    
    def _contains_class_name(self, prompt: str, class_name: str) -> bool:
        """
        STRICT validation: Check if prompt contains the actual class name word(s).
        No synonyms - the class name itself must appear in the prompt.
        """
        prompt_lower = prompt.lower()
        
        # Normalize class name (handle hyphens, underscores)
        # e.g., "desk-organizer" -> ["desk", "organizer"]
        # e.g., "tv-screen" -> ["tv", "screen"]
        class_words = class_name.lower().replace("-", " ").replace("_", " ").split()
        
        # For compound classes, require AT LEAST ONE word to appear
        # e.g., "tv-screen" -> prompt must contain "tv" OR "screen"
        # e.g., "indoor-plant" -> prompt must contain "indoor" OR "plant"
        for word in class_words:
            # Skip very short words (like single letters)
            if len(word) < 2:
                continue
            # Check if word appears as a whole word (not substring)
            # e.g., "wall" should match "white wall" but not "swallow"
            import re
            if re.search(r'\b' + re.escape(word) + r'\b', prompt_lower):
                return True
        
        return False
    
    def _parse_response(self, response: str, class_name: str) -> List[str]:
        """Parse LLM response into list of clean prompts."""
        lines = response.strip().split('\n')
        prompts = []
        ungrounded_prompts = []  # Prompts that don't contain class name
        
        for line in lines:
            # Clean the line
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Remove numbering (1., 2., -, *, etc.)
            if line[0].isdigit():
                line = line.lstrip('0123456789.-) ').strip()
            elif line[0] in '-*•':
                line = line[1:].strip()
            
            # Remove quotes if present
            line = line.strip('"\'')
            
            # Remove leading articles
            for article in ['a ', 'an ', 'the ']:
                if line.lower().startswith(article):
                    line = line[len(article):]
            
            # Skip if too short or too long
            if len(line) < 3 or len(line) > 50:
                continue
            
            # CRITICAL: Skip non-English text (filter out Mandarin, etc.)
            if not self._is_english_only(line):
                print(f"  [Warning] Skipping non-English prompt: {line}")
                continue
            
            # Check if prompt contains class name (grounding check)
            if self._contains_class_name(line, class_name):
                prompts.append(line)
            else:
                ungrounded_prompts.append(line)
        
        # If we have grounded prompts, use those; otherwise fall back to ungrounded
        if prompts:
            if ungrounded_prompts:
                print(f"  [Quality] Filtered {len(ungrounded_prompts)} ungrounded prompts")
        else:
            # Fall back to ungrounded if no grounded prompts
            prompts = ungrounded_prompts
            if prompts:
                print(f"  [Warning] No grounded prompts, using ungrounded: {prompts}")
        
        # Ensure we have at least one prompt
        if not prompts:
            prompts = [class_name.replace("-", " ").replace("_", " ")]
        
        return prompts[:self.num_prompts]
    
    def generate_all_prompts(self, class_names: List[str]) -> Dict[str, List[str]]:
        """
        Generate prompts for all classes.
        
        Args:
            class_names: List of class names
            
        Returns:
            Dictionary mapping class names to prompt lists
        """
        all_prompts = {}
        
        for class_name in class_names:
            print(f"[LLMPromptGenerator] Generating prompts for: {class_name}")
            prompts = self.generate_prompts(class_name)
            all_prompts[class_name] = prompts
            print(f"  -> {prompts}")
        
        return all_prompts
    
    def export_prompts(self, prompts: Dict[str, List[str]], output_path: str):
        """Export generated prompts to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(prompts, f, indent=2)
        print(f"[LLMPromptGenerator] Exported prompts to: {output_path}")


# =============================================================================
# PROMPT QUALITY METRICS
# =============================================================================

class PromptMetrics:
    """
    Compute quality metrics for generated prompts.
    """
    
    @staticmethod
    def compute_diversity(prompts: Dict[str, List[str]], n: int = 2) -> Dict[str, float]:
        """
        Compute prompt diversity using unique n-grams.
        
        Args:
            prompts: Dictionary of class -> prompt list
            n: n-gram size (default: 2 for bigrams)
            
        Returns:
            Dictionary with diversity metrics
        """
        all_ngrams = []
        per_class_diversity = {}
        
        for class_name, prompt_list in prompts.items():
            class_ngrams = []
            for prompt in prompt_list:
                words = prompt.lower().split()
                ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
                class_ngrams.extend(ngrams)
            
            # Unique ratio for this class
            if class_ngrams:
                unique_ratio = len(set(class_ngrams)) / len(class_ngrams)
                per_class_diversity[class_name] = unique_ratio
            else:
                per_class_diversity[class_name] = 0.0
            
            all_ngrams.extend(class_ngrams)
        
        # Global metrics
        total_ngrams = len(all_ngrams)
        unique_ngrams = len(set(all_ngrams))
        
        return {
            'global_diversity': unique_ngrams / total_ngrams if total_ngrams > 0 else 0,
            'unique_ngrams': unique_ngrams,
            'total_ngrams': total_ngrams,
            'per_class_diversity': per_class_diversity,
            'avg_class_diversity': sum(per_class_diversity.values()) / len(per_class_diversity) if per_class_diversity else 0
        }
    
    @staticmethod
    def compute_avg_prompt_length(prompts: Dict[str, List[str]]) -> Dict[str, float]:
        """Compute average prompt length in words."""
        per_class_length = {}
        all_lengths = []
        
        for class_name, prompt_list in prompts.items():
            lengths = [len(p.split()) for p in prompt_list]
            per_class_length[class_name] = sum(lengths) / len(lengths) if lengths else 0
            all_lengths.extend(lengths)
        
        return {
            'global_avg_length': sum(all_lengths) / len(all_lengths) if all_lengths else 0,
            'per_class_avg_length': per_class_length
        }
    
    @staticmethod
    def compute_grounding_score(prompts: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Compute grounding score: what percentage of prompts contain their class name.
        
        Higher score = prompts are more grounded to their intended class.
        Uses STRICT validation - no synonyms, actual class words must appear.
        """
        import re
        
        def is_grounded(prompt: str, class_name: str) -> bool:
            """STRICT: Check if prompt contains actual class name word(s)."""
            prompt_lower = prompt.lower()
            class_words = class_name.lower().replace("-", " ").replace("_", " ").split()
            
            for word in class_words:
                if len(word) < 2:
                    continue
                # Word boundary check to avoid substring matches
                if re.search(r'\b' + re.escape(word) + r'\b', prompt_lower):
                    return True
            return False
        
        per_class_grounding = {}
        total_grounded = 0
        total_prompts = 0
        ungrounded_examples = []
        
        for class_name, prompt_list in prompts.items():
            grounded_count = sum(1 for p in prompt_list if is_grounded(p, class_name))
            per_class_grounding[class_name] = grounded_count / len(prompt_list) if prompt_list else 0
            total_grounded += grounded_count
            total_prompts += len(prompt_list)
            
            # Track ungrounded examples
            for p in prompt_list:
                if not is_grounded(p, class_name):
                    ungrounded_examples.append({"class": class_name, "prompt": p})
        
        return {
            'global_grounding': total_grounded / total_prompts if total_prompts > 0 else 0,
            'grounded_count': total_grounded,
            'total_prompts': total_prompts,
            'per_class_grounding': per_class_grounding,
            'ungrounded_examples': ungrounded_examples[:10]  # Top 10 examples
        }
    
    @staticmethod
    def compute_clip_similarity(
        prompts: Dict[str, List[str]], 
        model=None, 
        processor=None,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """
        Compute CLIP/SigLIP cosine similarity between prompts and their class names.
        
        This measures semantic alignment: how similar is the prompt embedding 
        to the class name embedding? Higher similarity = more semantically relevant.
        
        Args:
            prompts: Dictionary of class -> prompt list
            model: Pre-loaded CLIP/SigLIP model (optional, will load if None)
            processor: Pre-loaded processor (optional)
            device: Device to use
            
        Returns:
            Dictionary with similarity metrics per class and globally
        """
        import torch
        
        # Try to use existing model or load SigLIP
        if model is None:
            try:
                from transformers import AutoModel, AutoProcessor
                print("[PromptMetrics] Loading SigLIP for similarity computation...")
                model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
                processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
            except Exception as e:
                print(f"[PromptMetrics] Could not load model: {e}")
                return {"error": str(e), "global_similarity": 0.0}
        
        per_class_similarity = {}
        all_similarities = []
        low_similarity_examples = []
        
        model.eval()
        with torch.no_grad():
            for class_name, prompt_list in prompts.items():
                # Encode class name
                class_text = class_name.replace("-", " ").replace("_", " ")
                class_inputs = processor(text=[class_text], return_tensors="pt", padding=True).to(device)
                class_embed = model.get_text_features(**class_inputs)
                class_embed = class_embed / class_embed.norm(dim=-1, keepdim=True)
                
                class_sims = []
                for prompt in prompt_list:
                    # Encode prompt
                    prompt_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
                    prompt_embed = model.get_text_features(**prompt_inputs)
                    prompt_embed = prompt_embed / prompt_embed.norm(dim=-1, keepdim=True)
                    
                    # Cosine similarity
                    sim = (class_embed @ prompt_embed.T).item()
                    class_sims.append(sim)
                    all_similarities.append(sim)
                    
                    # Track low similarity examples
                    if sim < 0.7:
                        low_similarity_examples.append({
                            "class": class_name,
                            "prompt": prompt,
                            "similarity": round(sim, 3)
                        })
                
                avg_sim = sum(class_sims) / len(class_sims) if class_sims else 0
                per_class_similarity[class_name] = round(avg_sim, 3)
        
        # Sort low similarity examples
        low_similarity_examples.sort(key=lambda x: x["similarity"])
        
        global_sim = sum(all_similarities) / len(all_similarities) if all_similarities else 0
        
        return {
            'global_similarity': round(global_sim, 3),
            'per_class_similarity': per_class_similarity,
            'low_similarity_examples': low_similarity_examples[:15],
            'min_similarity': round(min(all_similarities), 3) if all_similarities else 0,
            'max_similarity': round(max(all_similarities), 3) if all_similarities else 0
        }


class FilteredConfusionMatrix:
    """
    Extract and analyze a filtered subset of a confusion matrix.
    Allows focusing on specific classes of interest.
    """
    
    def __init__(self, focus_classes: List[str] = None):
        """
        Args:
            focus_classes: List of class names to include in filtered matrix.
                          If None, uses DEFAULT_FOCUS_CLASSES.
        """
        self.focus_classes = focus_classes or DEFAULT_FOCUS_CLASSES
    
    @staticmethod
    def filter_confusion_matrix(
        full_matrix,
        all_class_names: List[str],
        focus_classes: List[str]
    ) -> Dict:
        """
        Extract a sub-matrix from a full confusion matrix.
        
        Args:
            full_matrix: numpy array of shape (N, N) - the full confusion matrix
            all_class_names: List of all class names (length N)
            focus_classes: List of class names to extract
            
        Returns:
            Dictionary with filtered matrix and metadata
        """
        import numpy as np
        
        # Get indices of focus classes
        focus_indices = []
        valid_focus_classes = []
        for cls in focus_classes:
            if cls in all_class_names:
                focus_indices.append(all_class_names.index(cls))
                valid_focus_classes.append(cls)
            else:
                print(f"[Warning] Class '{cls}' not found in class list, skipping")
        
        if len(focus_indices) == 0:
            return {"error": "No valid focus classes found"}
        
        # Extract sub-matrix
        focus_indices = np.array(focus_indices)
        filtered = full_matrix[np.ix_(focus_indices, focus_indices)]
        
        # Compute per-class metrics
        per_class_metrics = {}
        for i, cls in enumerate(valid_focus_classes):
            row_sum = filtered[i, :].sum()
            if row_sum > 0:
                accuracy = filtered[i, i] / row_sum
                # Top confusions for this class
                row = filtered[i, :].copy()
                row[i] = 0  # Exclude self
                if row.sum() > 0:
                    top_conf_idx = row.argmax()
                    top_confused_with = valid_focus_classes[top_conf_idx]
                    top_conf_rate = row[top_conf_idx] / row_sum
                else:
                    top_confused_with = None
                    top_conf_rate = 0
            else:
                accuracy = 0
                top_confused_with = None
                top_conf_rate = 0
            
            per_class_metrics[cls] = {
                "accuracy": accuracy,
                "total_samples": int(row_sum),
                "top_confused_with": top_confused_with,
                "top_confusion_rate": top_conf_rate
            }
        
        # Overall metrics for subset
        total = filtered.sum()
        correct = np.diag(filtered).sum()
        
        return {
            "filtered_matrix": filtered,
            "focus_classes": valid_focus_classes,
            "num_classes": len(valid_focus_classes),
            "total_samples": int(total),
            "correct_samples": int(correct),
            "subset_accuracy": correct / total if total > 0 else 0,
            "per_class_metrics": per_class_metrics
        }
    
    def analyze_and_save(
        self,
        full_matrix,
        all_class_names: List[str],
        output_path: str
    ) -> Dict:
        """
        Analyze and save filtered confusion matrix results.
        
        Args:
            full_matrix: Full confusion matrix
            all_class_names: All class names
            output_path: Path to save results
            
        Returns:
            Analysis results dictionary
        """
        results = self.filter_confusion_matrix(
            full_matrix, all_class_names, self.focus_classes
        )
        
        if "error" in results:
            return results
        
        # Save to file
        output = {
            "focus_classes": results["focus_classes"],
            "num_classes": results["num_classes"],
            "total_samples": results["total_samples"],
            "correct_samples": results["correct_samples"],
            "subset_accuracy": results["subset_accuracy"],
            "per_class_metrics": results["per_class_metrics"],
            "matrix": results["filtered_matrix"].tolist()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"[FilteredConfusionMatrix] Saved to: {output_path}")
        return results
    
    def print_summary(self, results: Dict):
        """Print a summary of the filtered confusion matrix analysis."""
        if "error" in results:
            print(f"[Error] {results['error']}")
            return
        
        print(f"\n=== Filtered Confusion Matrix ({results['num_classes']} classes) ===")
        print(f"Classes: {', '.join(results['focus_classes'])}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Subset accuracy: {results['subset_accuracy']:.1%}")
        print("\nPer-class breakdown:")
        for cls, metrics in results['per_class_metrics'].items():
            conf_str = f" (confused with: {metrics['top_confused_with']})" if metrics['top_confused_with'] else ""
            print(f"  {cls:15s}: {metrics['accuracy']:.1%} acc, {metrics['total_samples']:4d} samples{conf_str}")


# =============================================================================
# PROMPT SELECTION METHODS
# =============================================================================

def select_prompt_for_sam3(
    prompts: List[str],
    method: str = "first",
    all_class_prompts: Dict[str, List[str]] = None
) -> Tuple[str, Dict]:
    """
    Select the best prompt for SAM3 text-prompted segmentation.
    
    Args:
        prompts: List of prompts for a single class
        method: Selection method - "first", "shortest", "longest", "random", "most_unique"
        all_class_prompts: All prompts dict (needed for "most_unique" method)
        
    Returns:
        Tuple of (selected_prompt, selection_info)
    """
    import random
    
    if not prompts:
        return "", {"method": method, "error": "no_prompts"}
    
    if len(prompts) == 1:
        return prompts[0], {"method": method, "reason": "single_prompt"}
    
    selection_info = {"method": method, "candidates": len(prompts)}
    
    if method == "first":
        selected = prompts[0]
        selection_info["index"] = 0
        
    elif method == "shortest":
        lengths = [(len(p.split()), i, p) for i, p in enumerate(prompts)]
        lengths.sort()
        selected = lengths[0][2]
        selection_info["index"] = lengths[0][1]
        selection_info["word_count"] = lengths[0][0]
        
    elif method == "longest":
        lengths = [(len(p.split()), i, p) for i, p in enumerate(prompts)]
        lengths.sort(reverse=True)
        selected = lengths[0][2]
        selection_info["index"] = lengths[0][1]
        selection_info["word_count"] = lengths[0][0]
        
    elif method == "random":
        idx = random.randint(0, len(prompts) - 1)
        selected = prompts[idx]
        selection_info["index"] = idx
        
    elif method == "most_unique":
        # Select prompt with highest uniqueness (least overlap with other prompts)
        # Compute n-gram overlap with all other prompts in this class
        from collections import Counter
        
        def get_ngrams(text, n=2):
            words = text.lower().split()
            return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
        
        uniqueness_scores = []
        all_ngrams = Counter()
        
        # Count all ngrams across all prompts for this class
        for p in prompts:
            for ng in get_ngrams(p):
                all_ngrams[ng] += 1
        
        # Score each prompt by how unique its ngrams are
        for i, p in enumerate(prompts):
            prompt_ngrams = get_ngrams(p)
            if not prompt_ngrams:
                uniqueness_scores.append((0, i, p))
                continue
            # Lower frequency = more unique
            uniqueness = sum(1.0 / all_ngrams[ng] for ng in prompt_ngrams) / len(prompt_ngrams)
            uniqueness_scores.append((uniqueness, i, p))
        
        uniqueness_scores.sort(reverse=True)
        selected = uniqueness_scores[0][2]
        selection_info["index"] = uniqueness_scores[0][1]
        selection_info["uniqueness_score"] = uniqueness_scores[0][0]
        
    else:
        # Default to first
        selected = prompts[0]
        selection_info["index"] = 0
        selection_info["fallback"] = True
    
    return selected, selection_info


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_llm_prompts(label: str, llm_prompts: Dict[str, List[str]], fallback_mode: str = 'ensemble') -> List[str]:
    """
    Get prompts for a label, using LLM-generated prompts if available.
    
    Args:
        label: Class label
        llm_prompts: Dictionary of LLM-generated prompts
        fallback_mode: Mode to use if label not in llm_prompts
        
    Returns:
        List of prompts
    """
    from replica_prompts import get_prompts, IMAGENET_TEMPLATES
    
    if label in llm_prompts:
        return llm_prompts[label]
    
    # Fallback
    return get_prompts(label, mode=fallback_mode)


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LLM prompts for segmentation")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of prompts per class")
    parser.add_argument("--classes", nargs="+", help="Specific classes to generate (default: all)")
    parser.add_argument("--output", default="llm_prompts.json", help="Output JSON file")
    parser.add_argument("--focus-classes", nargs="+", help="Classes to include in filtered confusion matrix")
    
    args = parser.parse_args()
    
    if args.focus_classes:
        # Just print the focus classes that would be used
        print(f"[Info] Focus classes for filtered confusion matrix: {args.focus_classes}")
        print("[Info] Use FilteredConfusionMatrix.filter_confusion_matrix() with your confusion matrix data")
    else:
        # Generate prompts
        from replica_prompts import REPLICA_PROMPTS
        
        generator = LLMPromptGenerator(
            model_name=args.model,
            num_prompts=args.num_prompts
        )
        
        # Get class list
        if args.classes:
            classes = args.classes
        else:
            classes = list(REPLICA_PROMPTS.keys())
        
        # Generate
        prompts = generator.generate_all_prompts(classes)
        
        # Compute metrics
        diversity = PromptMetrics.compute_diversity(prompts)
        length = PromptMetrics.compute_avg_prompt_length(prompts)
        
        print("\n=== Prompt Quality Metrics ===")
        print(f"Global diversity (unique bigrams): {diversity['global_diversity']:.2%}")
        print(f"Average prompt length: {length['global_avg_length']:.1f} words")
        
        # Export
        generator.export_prompts(prompts, args.output)
