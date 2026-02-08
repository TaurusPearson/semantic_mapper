"""
replica_prompts.py

High-quality open-vocabulary prompt expansions for the Replica dataset.
Aligned specifically to the 51 reduced classes in eval_info.yaml.
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

PROMPT_MODE = 'handcrafted' 

# =============================================================================
# 1. ENSEMBLE TEMPLATES (OpenAI ImageNet Standard)
# =============================================================================

IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]
# =============================================================================
# 2. HANDCRAFTED PROMPTS (Aligned to 51 Reduced Classes)
# =============================================================================

REPLICA_PROMPTS = {
    # --- Structural ---
    "wall": [
        "a vertical painted wall surface",
        "the side walls of a room",
        "flat vertical plaster wall",
        "painted drywall",
        "structural wall",
    ],
    "floor": [
        "the ground floor surface",
        "tiled or carpeted flooring",
        "the bottom surface of the room",
        "floor tiles or wood planks",
        "walkable ground area",
    ],
    "ceiling": [
        "the overhead ceiling surface",
        "top of the room",
        "drywall ceiling with lights",
        "horizontal ceiling above",
    ],
    "window": [
        "a glass window pane",
        "window looking outside",
        "transparent glass window frame",
        "daylight entering through a window",
    ],
    "door": [
        "a wooden entrance door",
        "tall vertical door panel",
        "door with a handle or knob",
        "hinged door way",
    ],
    "pillar": [
        "a vertical structural column",
        "concrete support pillar",
        "load-bearing post",
        "vertical beam",
    ],
    "vent": [
        "air conditioning vent grille",
        "hvac air duct cover",
        "slatted metal vent on wall or ceiling",
        "rectangular air vent",
    ],
    "pipe": [
        "exposed plumbing pipe",
        "metal or pvc pipe",
        "industrial piping on wall",
    ],
    "panel": [
        "decorative wall paneling",
        "wooden wainscoting",
        "flat rectangular panel",
        "access panel on wall",
    ],
    "switch": [
        "small electrical light switch",
        "wall plate with a toggle button",
        "plastic switch next to door",
    ],
    "wall-plug": [
        "electrical power outlet",
        "wall socket with holes",
        "power plug faceplate",
    ],

    # --- Furniture (Seating) ---
    "chair": [
        "a chair for sitting",
        "wooden or plastic chair with legs",
        "dining room chair",
        "office chair with backrest",
    ],
    "sofa": [
        "a large upholstered couch",
        "living room sofa",
        "cushioned seating for multiple people",
        "soft fabric sofa",
    ],
    "bench": [
        "a long wooden bench",
        "seating bench without backrest",
        "flat seating surface",
    ],
    "stool": [
        "a small stool without backrest",
        "round top bar stool",
        "low seating stool",
    ],

    # --- Furniture (Surfaces & Storage) ---
    "table": [
        "a dining table or desk",
        "flat table surface on legs",
        "wooden furniture table",
        "raised flat surface",
    ],
    "desk": [
        "an office workspace desk",
        "computer desk with drawers",
        "wooden working table",
    ],
    "nightstand": [
        "small bedside table cabinet",
        "nightstand with drawers",
        "low table next to bed",
    ],
    "cabinet": [
        "wooden storage cabinet",
        "cupboard with doors",
        "kitchen or office cabinet",
        "storage unit with shelves",
    ],
    "shelf": [
        "horizontal storage shelf",
        "bookshelf on the wall",
        "open shelving unit",
        "rack for holding items",
    ],
    "tv-stand": [
        "low cabinet under the television",
        "media console table",
        "wooden stand for tv",
    ],
    "plant-stand": [
        "tall stand holding a plant pot",
        "pedestal for indoor plant",
    ],
    "desk-organizer": [
        "container for pens and office supplies",
        "plastic desk caddy",
        "organizer box on table",
    ],

    # --- Bedroom / Soft ---
    "bed": [
        "a mattress on a bed frame",
        "bed with pillows and blankets",
        "sleeping furniture",
    ],
    "blanket": [
        "soft fabric blanket",
        "throw blanket on sofa or bed",
        "folded fleece blanket",
    ],
    "comforter": [
        "thick padded bed cover",
        "duvet or quilt on bed",
        "soft fluffy bedding",
    ],
    "pillow": [
        "soft cushion for head",
        "throw pillow on sofa",
        "fluffy bed pillow",
    ],
    "cushion": [
        "seat cushion pad",
        "square fabric cushion",
        "soft padding on chair",
    ],
    "rug": [
        "carpet rug on the floor",
        "patterned floor mat",
        "fabric area rug",
    ],
    "blinds": [
        "horizontal window blinds",
        "slatted window shades",
        "pulled down blinds",
        "window covering slats",
    ],
    "cloth": [
        "a piece of fabric",
        "folded cloth or rag",
        "textile material",
        "draped fabric",
    ],

    # --- Electronics ---
    "tv-screen": [
        "flat screen television",
        "black rectangular TV monitor",
        "large display screen",
    ],
    "monitor": [
        "computer monitor display",
        "desktop screen",
        "black lcd monitor",
    ],
    "camera": [
        "digital camera lens",
        "webcam or security camera",
        "photography device",
    ],
    "tablet": [
        "ipad or android tablet device",
        "flat touchscreen computer",
        "digital slate screen",
    ],
    "clock": [
        "wall clock with hands",
        "digital time display",
        "round clock face",
    ],

    # --- Decor & Small Objects ---
    "picture": [
        "framed painting or photo",
        "art hanging on the wall",
        "picture frame",
    ],
    "sculpture": [
        "decorative art statue",
        "stone or metal sculpture",
        "carved artistic object",
    ],
    "vase": [
        "ceramic flower vase",
        "tall decorative vessel",
        "glass vase",
    ],
    "indoor-plant": [
        "potted house plant",
        "green leaves in a pot",
        "indoor vegetation",
    ],
    "pot": [
        "plant pot or planter",
        "ceramic container for plants",
        "cooking pot",
    ],
    "basket": [
        "woven wicker basket",
        "storage basket container",
        "laundry hamper",
    ],
    "bin": [
        "trash can",
        "waste bin container",
        "recycling bin on floor",
        "garbage basket",
    ],
    "box": [
        "cardboard or plastic box",
        "rectangular storage container",
        "shipping box",
    ],
    "book": [
        "hardcover or paperback book",
        "reading book with spine",
        "stack of books",
    ],
    "bottle": [
        "water bottle",
        "glass or plastic beverage container",
        "flask",
    ],
    "plate": [
        "ceramic dinner plate",
        "round dish for food",
        "flat china plate",
    ],
    "bowl": [
        "deep ceramic bowl",
        "round vessel for soup",
        "kitchen bowl",
    ],
    "candle": [
        "wax candle with wick",
        "decorative candle",
        "glass jar candle",
    ],
    "lamp": [
        "table lamp with shade",
        "standing floor lamp",
        "light fixture",
    ],
    "tissue-paper": [
        "box of facial tissues",
        "white paper tissue",
        "napkin dispenser",
    ],
}

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def validate_prompt_keys(valid_classes: list, mode: str = 'ensemble'):
    """
    Validates keys only if in 'handcrafted' mode.
    """
    if mode != 'handcrafted':
        print(f"[Info] Using mode='{mode}'. Skipping strict dictionary validation.")
        return

    prompts_keys = set(REPLICA_PROMPTS.keys())
    valid_set = set(valid_classes)
    
    # 1. Check for keys in Prompts that are NOT in Dataset
    extras = prompts_keys - valid_set
    if extras:
        error_msg = (
            f"\n[CRITICAL ERROR] Strict Prompt Validation Failed!\n"
            f"The dictionary 'REPLICA_PROMPTS' contains keys that are NOT in the dataset config:\n"
            f"{extras}\n"
            f"Please remove these keys from replica_prompts.py or update eval_info.yaml."
        )
        raise ValueError(error_msg)

    # 2. Check for missing prompts
    missing = valid_set - prompts_keys
    if missing:
        print(f"\n[WARNING] The following classes rely on generic fallbacks (No handcrafted prompt found):")
        print(f"{missing}\n")
    else:
        print("[Info] Strict Prompt Validation Passed: All prompts map to valid classes.")
        
def get_prompts(label: str, mode: str = 'ensemble', llm_prompts: dict = None):
    """
    Returns list of prompts for a given label based on the selected mode.
    
    Args:
        label: Class label
        mode: One of 'ensemble', 'handcrafted', 'llm'
        llm_prompts: Dictionary of LLM-generated prompts (required for mode='llm')
    """
    if mode == "llm":
        # Use LLM-generated prompts
        if llm_prompts is None:
            raise ValueError("llm_prompts dict required when mode='llm'")
        
        if label in llm_prompts:
            return llm_prompts[label]
        
        # Try normalized label
        normalized = label.replace("_", "-").lower()
        if normalized in llm_prompts:
            return llm_prompts[normalized]
        
        # Fallback to handcrafted if LLM prompt not available
        print(f"[Warning] No LLM prompt for '{label}', falling back to handcrafted")
        if label in REPLICA_PROMPTS:
            return REPLICA_PROMPTS[label]
        return [t.format(label.replace("-", " ")) for t in IMAGENET_TEMPLATES]
    
    elif mode == "handcrafted":
        if label in REPLICA_PROMPTS:
            return REPLICA_PROMPTS[label]
        
        # SOTA Fallback: If strict key missing, try to canonicalize
        normalized = label.replace("_", "-").lower()
        if normalized in REPLICA_PROMPTS:
            return REPLICA_PROMPTS[normalized]
            
        # Final Fallback: Use ImageNet templates if handcrafted key is missing
        return [t.format(label.replace("-", " ")) for t in IMAGENET_TEMPLATES]

    elif mode == "ensemble":
        # Strict OpenAI CLIP approach
        clean_label = label.replace("-", " ").replace("_", " ").strip()
        return [t.format(clean_label) for t in IMAGENET_TEMPLATES]

    # Default fallback
    return [f"a photo of a {label}"]