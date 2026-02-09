#!/usr/bin/env python3
"""
Generate ScanNet200 (and ScanNet20) ground truth label text files.

Combines:
  Step 1: ScanNet BenchmarkScripts/ScanNet200/preprocess_scannet200.py
  Step 2: OVO/scripts/scannet_preprocess.py

Into one script that goes directly from raw ScanNet data to GT .txt files.

Two modes:
  A) With TSV (authoritative): Uses scannetv2-labels.combined.tsv for exact mapping
  B) Without TSV (fallback): Uses CLASS_LABELS_200 + fuzzy matching

Usage:
  python generate_scannet200_gt.py --scannet_root <path_to_scannet_data>
  python generate_scannet200_gt.py --scannet_root <path> --tsv_path <path_to_tsv>
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData

# =============================================================================
# ScanNet200 constants (from BenchmarkScripts/ScanNet200/scannet200_constants.py)
# =============================================================================
VALID_CLASS_IDS_200 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22,
    23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65,
    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86,
    87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
    106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131,
    132, 134, 136, 138, 139, 140, 141, 145, 148, 154, 155, 156, 157, 159,
    161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195,
    202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276,
    283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399,
    408, 417, 488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164,
    1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176,
    1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189,
    1190, 1191
)

CLASS_LABELS_200 = (
    'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf',
    'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window',
    'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair',
    'coffee table', 'box', 'refrigerator', 'lamp', 'kitchen cabinet', 'towel',
    'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion',
    'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard',
    'bag', 'backpack', 'toilet paper', 'printer', 'tv stand', 'whiteboard',
    'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave',
    'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench',
    'board', 'washing machine', 'mirror', 'copier', 'basket', 'sofa chair',
    'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
    'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard',
    'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
    'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand',
    'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar',
    'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder',
    'bathroom stall', 'shower wall', 'cup', 'jacket', 'storage bin',
    'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat',
    'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board',
    'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
    'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball',
    'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray',
    'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse',
    'toilet seat cover dispenser', 'furniture', 'cart', 'storage container',
    'scale', 'tissue box', 'light switch', 'crate', 'power outlet',
    'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner',
    'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack', 'broom',
    'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle',
    'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher',
    'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand',
    'projector screen', 'divider', 'laundry detergent', 'bathroom counter',
    'object', 'bathroom vanity', 'closet wall', 'laundry hamper',
    'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell',
    'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
    'coffee kettle', 'structure', 'shower head', 'keyboard piano',
    'case of water bottles', 'coat rack', 'storage organizer', 'folded chair',
    'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant',
    'luggage', 'mattress'
)

VALID_CLASS_IDS_200_SET = set(VALID_CLASS_IDS_200)

# Build name → id mapping from the constants
_NAME_TO_ID = {name.lower(): vid for name, vid in zip(CLASS_LABELS_200, VALID_CLASS_IDS_200)}


def build_mapping_from_tsv(tsv_path: str) -> dict:
    """Build raw_category → fine-grained ID mapping from the TSV file."""
    import pandas as pd
    labels_pd = pd.read_csv(tsv_path, sep='\t', header=0)
    
    raw_to_id = {}
    for _, row in labels_pd.iterrows():
        raw_cat = str(row['raw_category']).strip()
        label_id = int(row['id'])
        if raw_cat and raw_cat != 'nan':
            if raw_cat not in raw_to_id:
                raw_to_id[raw_cat] = label_id
    
    print(f"[TSV] Loaded {len(raw_to_id)} raw_category -> id mappings")
    return raw_to_id


def build_mapping_from_constants() -> dict:
    """
    Build raw_category → fine-grained ID mapping WITHOUT the TSV.
    Uses CLASS_LABELS_200 + common aliases/plurals.
    """
    raw_to_id = {}
    
    # 1. Direct mapping: class name → ID
    for name, vid in zip(CLASS_LABELS_200, VALID_CLASS_IDS_200):
        raw_to_id[name] = vid
        raw_to_id[name.lower()] = vid
    
    # 2. Common plural forms
    for name, vid in zip(CLASS_LABELS_200, VALID_CLASS_IDS_200):
        lower = name.lower()
        # Add plural: "door" → "doors", "book" → "books"
        raw_to_id[lower + 's'] = vid
        raw_to_id[lower + 'es'] = vid
        # Add singular if name is already plural-looking
        if lower.endswith('s') and len(lower) > 2:
            raw_to_id[lower[:-1]] = vid
    
    # 3. Known aliases from ScanNet aggregation files
    # These are common raw_category strings that differ from CLASS_LABELS_200
    aliases = {
        "kitchen cabinets": _NAME_TO_ID.get("kitchen cabinet", 0),
        "kitchen island": _NAME_TO_ID.get("kitchen counter", 0),
        "sofa": _NAME_TO_ID.get("couch", 0),
        "couch pillows": _NAME_TO_ID.get("pillow", 0),
        "throw pillow": _NAME_TO_ID.get("pillow", 0),
        "throw pillows": _NAME_TO_ID.get("pillow", 0),
        "books": _NAME_TO_ID.get("book", 0),
        "papers": _NAME_TO_ID.get("paper", 0),
        "shoes": _NAME_TO_ID.get("shoe", 0),
        "doors": _NAME_TO_ID.get("door", 0),
        "shelves": _NAME_TO_ID.get("shelf", 0),
        "cabinets": _NAME_TO_ID.get("cabinet", 0),
        "tables": _NAME_TO_ID.get("table", 0),
        "chairs": _NAME_TO_ID.get("chair", 0),
        "walls": _NAME_TO_ID.get("wall", 0),
        "windows": _NAME_TO_ID.get("window", 0),
        "desks": _NAME_TO_ID.get("desk", 0),
        "lamps": _NAME_TO_ID.get("lamp", 0),
        "bottles": _NAME_TO_ID.get("bottle", 0),
        "coat": _NAME_TO_ID.get("jacket", 0),
        "coats": _NAME_TO_ID.get("jacket", 0),
        "jackets": _NAME_TO_ID.get("jacket", 0),
        "lamp base": _NAME_TO_ID.get("lamp", 0),
        "piano bench": _NAME_TO_ID.get("bench", 0),
        "shower walls": _NAME_TO_ID.get("shower wall", 0),
        "bathroom stalls": _NAME_TO_ID.get("bathroom stall", 0),
        "closet doors": _NAME_TO_ID.get("closet door", 0),
        "shower doors": _NAME_TO_ID.get("shower door", 0),
        "pillows": _NAME_TO_ID.get("pillow", 0),
        "curtains": _NAME_TO_ID.get("curtain", 0),
        "blankets": _NAME_TO_ID.get("blanket", 0),
        "towels": _NAME_TO_ID.get("towel", 0),
        "plants": _NAME_TO_ID.get("plant", 0),
        "pictures": _NAME_TO_ID.get("picture", 0),
        "monitors": _NAME_TO_ID.get("monitor", 0),
        "bags": _NAME_TO_ID.get("bag", 0),
        "boxes": _NAME_TO_ID.get("box", 0),
        "baskets": _NAME_TO_ID.get("basket", 0),
        "bins": _NAME_TO_ID.get("bin", 0),
        "cups": _NAME_TO_ID.get("cup", 0),
        "plates": _NAME_TO_ID.get("plate", 0),
        "bowls": _NAME_TO_ID.get("bowl", 0),
        "stools": _NAME_TO_ID.get("stool", 0),
        "office chairs": _NAME_TO_ID.get("office chair", 0),
        "counter top": _NAME_TO_ID.get("counter", 0),
        "countertop": _NAME_TO_ID.get("counter", 0),
        "nightstands": _NAME_TO_ID.get("nightstand", 0),
        "dressers": _NAME_TO_ID.get("dresser", 0),
        "wardrobes": _NAME_TO_ID.get("wardrobe", 0),
        "file cabinets": _NAME_TO_ID.get("file cabinet", 0),
        "bookshelves": _NAME_TO_ID.get("bookshelf", 0),
        "bookcase": _NAME_TO_ID.get("bookshelf", 0),
        "bookcases": _NAME_TO_ID.get("bookshelf", 0),
        "bathtubs": _NAME_TO_ID.get("bathtub", 0),
        "toilets": _NAME_TO_ID.get("toilet", 0),
        "sinks": _NAME_TO_ID.get("sink", 0),
        "mirrors": _NAME_TO_ID.get("mirror", 0),
        "stairs": _NAME_TO_ID.get("stairs", 0),
        "staircase": _NAME_TO_ID.get("stairs", 0),
        "cushions": _NAME_TO_ID.get("cushion", 0),
        "speaker": _NAME_TO_ID.get("speaker", 0),
        "speakers": _NAME_TO_ID.get("speaker", 0),
        "ceiling lamp": _NAME_TO_ID.get("ceiling light", 0),
        "light fixture": _NAME_TO_ID.get("light", 0),
        "light fixtures": _NAME_TO_ID.get("light", 0),
        "tv monitor": _NAME_TO_ID.get("tv", 0),
        "television": _NAME_TO_ID.get("tv", 0),
        "recycling bins": _NAME_TO_ID.get("recycling bin", 0),
        "trash": _NAME_TO_ID.get("trash can", 0),
        "garbage": _NAME_TO_ID.get("trash can", 0),
        "garbage can": _NAME_TO_ID.get("trash can", 0),
        "garbage bin": _NAME_TO_ID.get("trash bin", 0),
        "trashcan": _NAME_TO_ID.get("trash can", 0),
        "ottoman": _NAME_TO_ID.get("ottoman", 0),
        "ottomans": _NAME_TO_ID.get("ottoman", 0),
        "range hoods": _NAME_TO_ID.get("range hood", 0),
        "bathroom counter": _NAME_TO_ID.get("bathroom counter", 0),
        "bathroom sink": _NAME_TO_ID.get("sink", 0),
        "kitchen sink": _NAME_TO_ID.get("sink", 0),
        "bulletin boards": _NAME_TO_ID.get("bulletin board", 0),
        "white board": _NAME_TO_ID.get("whiteboard", 0),
        "black board": _NAME_TO_ID.get("blackboard", 0),
        "columns": _NAME_TO_ID.get("column", 0),
        "pillars": _NAME_TO_ID.get("pillar", 0),
        "ladders": _NAME_TO_ID.get("ladder", 0),
        "pipes": _NAME_TO_ID.get("pipe", 0),
        "rails": _NAME_TO_ID.get("rail", 0),
        "racks": _NAME_TO_ID.get("rack", 0),
        "vents": _NAME_TO_ID.get("vent", 0),
        "radiators": _NAME_TO_ID.get("radiator", 0),
        "suitcases": _NAME_TO_ID.get("suitcase", 0),
        "backpacks": _NAME_TO_ID.get("backpack", 0),
        "heater": _NAME_TO_ID.get("radiator", 0),
        "mattresses": _NAME_TO_ID.get("mattress", 0),
        "posters": _NAME_TO_ID.get("poster", 0),
        "potted plants": _NAME_TO_ID.get("potted plant", 0),
        "flower pot": _NAME_TO_ID.get("potted plant", 0),
        "power outlets": _NAME_TO_ID.get("power outlet", 0),
        "light switches": _NAME_TO_ID.get("light switch", 0),
        "tissue boxes": _NAME_TO_ID.get("tissue box", 0),
        "paper towel dispensers": _NAME_TO_ID.get("paper towel dispenser", 0),
        "soap dispensers": _NAME_TO_ID.get("soap dispenser", 0),
        "fire extinguishers": _NAME_TO_ID.get("fire extinguisher", 0),
        "keyboard": _NAME_TO_ID.get("keyboard", 0),
        "keyboards": _NAME_TO_ID.get("keyboard", 0),
        "guitar": _NAME_TO_ID.get("guitar", 0),
        "guitars": _NAME_TO_ID.get("guitar", 0),
        "boardgame": _NAME_TO_ID.get("board", 0),
        "corkboard": _NAME_TO_ID.get("board", 0),
        "footrest": _NAME_TO_ID.get("ottoman", 0),
        "footstool": _NAME_TO_ID.get("stool", 0),
        "bed frame": _NAME_TO_ID.get("bed", 0),
        "bedframe": _NAME_TO_ID.get("bed", 0),
        "beds": _NAME_TO_ID.get("bed", 0),
        "couches": _NAME_TO_ID.get("couch", 0),
        "sofas": _NAME_TO_ID.get("couch", 0),
        "fridge": _NAME_TO_ID.get("refrigerator", 0),
        "mini-fridge": _NAME_TO_ID.get("mini fridge", 0),
        "refrigerators": _NAME_TO_ID.get("refrigerator", 0),
        "closets": _NAME_TO_ID.get("closet", 0),
        "storage": _NAME_TO_ID.get("storage bin", 0),
        "storage bins": _NAME_TO_ID.get("storage bin", 0),
        "storage containers": _NAME_TO_ID.get("storage container", 0),
        "printers": _NAME_TO_ID.get("printer", 0),
        "copiers": _NAME_TO_ID.get("copier", 0),
        "fans": _NAME_TO_ID.get("fan", 0),
        "laptops": _NAME_TO_ID.get("laptop", 0),
        "ovens": _NAME_TO_ID.get("oven", 0),
        "microwaves": _NAME_TO_ID.get("microwave", 0),
        "dishwashers": _NAME_TO_ID.get("dishwasher", 0),
        "washing machines": _NAME_TO_ID.get("washing machine", 0),
        "bicycles": _NAME_TO_ID.get("bicycle", 0),
        "bike": _NAME_TO_ID.get("bicycle", 0),
        "bikes": _NAME_TO_ID.get("bicycle", 0),
        "decoration": _NAME_TO_ID.get("decoration", 0),
        "decorations": _NAME_TO_ID.get("decoration", 0),
        "object": _NAME_TO_ID.get("object", 0),
        "objects": _NAME_TO_ID.get("object", 0),
        "furniture": _NAME_TO_ID.get("furniture", 0),
        "drawer": _NAME_TO_ID.get("dresser", 0),
        "drawers": _NAME_TO_ID.get("dresser", 0),
        "end tables": _NAME_TO_ID.get("end table", 0),
        "dining tables": _NAME_TO_ID.get("dining table", 0),
        "coffee tables": _NAME_TO_ID.get("coffee table", 0),
        "nightstand": _NAME_TO_ID.get("nightstand", 0),
        "armchairs": _NAME_TO_ID.get("armchair", 0),
        "recliner": _NAME_TO_ID.get("armchair", 0),
        "fireplace": _NAME_TO_ID.get("fireplace", 0),
        "fireplaces": _NAME_TO_ID.get("fireplace", 0),
        "mantel": _NAME_TO_ID.get("fireplace", 0),
        "mantle": _NAME_TO_ID.get("fireplace", 0),
        "clocks": _NAME_TO_ID.get("clock", 0),
        "alarm clocks": _NAME_TO_ID.get("alarm clock", 0),
        "signs": _NAME_TO_ID.get("sign", 0),
        "projectors": _NAME_TO_ID.get("projector", 0),
        "candles": _NAME_TO_ID.get("candle", 0),
        "plungers": _NAME_TO_ID.get("plunger", 0),
        "brooms": _NAME_TO_ID.get("broom", 0),
        "mops": _NAME_TO_ID.get("broom", 0),
        "water bottles": _NAME_TO_ID.get("water bottle", 0),
        "shower head": _NAME_TO_ID.get("shower head", 0),
        "shower heads": _NAME_TO_ID.get("shower head", 0),
        "mats": _NAME_TO_ID.get("mat", 0),
        "floor mat": _NAME_TO_ID.get("mat", 0),
        "yoga mat": _NAME_TO_ID.get("mat", 0),
        "bath mat": _NAME_TO_ID.get("mat", 0),
        "rug": _NAME_TO_ID.get("mat", 0),
        "rugs": _NAME_TO_ID.get("mat", 0),
        "carpet": _NAME_TO_ID.get("mat", 0),
        "container": _NAME_TO_ID.get("container", 0),
        "containers": _NAME_TO_ID.get("container", 0),
        "stands": _NAME_TO_ID.get("stand", 0),
        "carts": _NAME_TO_ID.get("cart", 0),
        "seats": _NAME_TO_ID.get("seat", 0),
        "bucket": _NAME_TO_ID.get("bucket", 0),
        "buckets": _NAME_TO_ID.get("bucket", 0),
        "telephone": _NAME_TO_ID.get("telephone", 0),
        "telephones": _NAME_TO_ID.get("telephone", 0),
        "phone": _NAME_TO_ID.get("telephone", 0),
    }
    
    for alias, vid in aliases.items():
        if vid and vid != 0:
            raw_to_id[alias] = vid
    
    # Remove any entries with id=0 (failed lookups from _NAME_TO_ID)
    raw_to_id = {k: v for k, v in raw_to_id.items() if v != 0}
    
    print(f"[Fallback] Built {len(raw_to_id)} raw_category -> id mappings from constants + aliases")
    return raw_to_id


def match_label(raw_label: str, raw_to_id: dict) -> int:
    """
    Try to match a raw aggregation label to a ScanNet200 ID.
    
    Tries:
      1. Exact match
      2. Lowercase match  
      3. Strip trailing 's' (plural)
      4. Last word match (for compound labels like "piano bench" → "bench")
    """
    # 1. Exact
    if raw_label in raw_to_id:
        return raw_to_id[raw_label]
    
    # 2. Lowercase
    lower = raw_label.lower().strip()
    if lower in raw_to_id:
        return raw_to_id[lower]
    
    # 3. Simple depluralize
    if lower.endswith('s') and lower[:-1] in raw_to_id:
        return raw_to_id[lower[:-1]]
    if lower.endswith('es') and lower[:-2] in raw_to_id:
        return raw_to_id[lower[:-2]]
    
    # 4. Direct lookup in CLASS_LABELS_200 (case-insensitive)
    if lower in _NAME_TO_ID:
        return _NAME_TO_ID[lower]
    
    # 5. Last word match (for "piano bench" → "bench")
    words = lower.split()
    if len(words) > 1:
        last_word = words[-1]
        if last_word in _NAME_TO_ID:
            return _NAME_TO_ID[last_word]
        # Try depluralize last word
        if last_word.endswith('s') and last_word[:-1] in _NAME_TO_ID:
            return _NAME_TO_ID[last_word[:-1]]
    
    return 0


def generate_scannet200_labels(
    scene_path: Path, 
    raw_to_id: dict,
    output_dir: Path
) -> bool:
    """
    Generate ScanNet200 fine-grained GT labels for one scene.
    """
    scene_name = scene_path.name
    
    # --- Find required files ---
    mesh_path = scene_path / f"{scene_name}_vh_clean_2.ply"
    segs_path = scene_path / f"{scene_name}_vh_clean_2.0.010000.segs.json"
    
    agg_candidates = [
        scene_path / f"{scene_name}.aggregation.json",
        scene_path / f"{scene_name}_vh_clean.aggregation.json",
    ]
    agg_path = next((p for p in agg_candidates if p.exists()), None)
    
    if not mesh_path.exists():
        print(f"  [SKIP] Missing mesh: {mesh_path.name}")
        return False
    if not segs_path.exists():
        print(f"  [SKIP] Missing segs: {segs_path.name}")
        return False
    if agg_path is None:
        print(f"  [SKIP] Missing aggregation file for {scene_name}")
        return False
    
    # --- 1. Read mesh to get vertex count ---
    plydata = PlyData.read(str(mesh_path))
    n_vertices = len(plydata['vertex'])
    
    # --- 2. Read segment indices ---
    with open(segs_path, 'r') as f:
        segs_data = json.load(f)
    seg_indices = np.array(segs_data['segIndices'])
    
    if len(seg_indices) != n_vertices:
        print(f"  [ERROR] Vertex count mismatch: mesh={n_vertices}, segs={len(seg_indices)}")
        return False
    
    # --- 3. Read aggregation ---
    with open(agg_path, 'r') as f:
        agg_data = json.load(f)
    seg_groups = agg_data['segGroups']
    
    # Build: segment_id → fine-grained label_id
    segment_to_label = {}
    unmapped_labels = set()
    mapped_labels = {}
    
    for group in seg_groups:
        raw_label = group['label']
        segments = group['segments']
        
        label_id = match_label(raw_label, raw_to_id)
        
        # Only keep valid ScanNet200 class IDs
        if label_id not in VALID_CLASS_IDS_200_SET:
            if label_id != 0:
                unmapped_labels.add(f"{raw_label}(id={label_id})")
            elif raw_label.lower() not in ('unannotated', 'remove', ''):
                unmapped_labels.add(raw_label)
            label_id = 0
        else:
            mapped_labels[raw_label] = label_id
        
        for seg_id in segments:
            segment_to_label[seg_id] = label_id
    
    # --- 4. Assign labels to vertices ---
    vertex_labels = np.zeros(n_vertices, dtype=np.int32)
    for vtx_idx in range(n_vertices):
        seg_id = seg_indices[vtx_idx]
        vertex_labels[vtx_idx] = segment_to_label.get(seg_id, 0)
    
    # --- 5. Write to .txt ---
    output_file = output_dir / f"{scene_name}.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(str(int(lbl)) for lbl in vertex_labels))
    
    # Stats
    unique_ids = np.unique(vertex_labels)
    n_labeled = int(np.sum(vertex_labels > 0))
    pct_labeled = 100.0 * n_labeled / n_vertices
    
    print(f"  [OK] {scene_name}: {n_vertices:,} vtx, {len(unique_ids)} classes, "
          f"{pct_labeled:.1f}% labeled")
    if mapped_labels:
        print(f"       Mapped: {dict(list(mapped_labels.items())[:8])}")
    if unmapped_labels:
        print(f"       UNMAPPED (->0): {unmapped_labels}")
    
    return True


def generate_scannet20_labels(scene_path: Path, output_dir: Path) -> bool:
    """Generate ScanNet20 (NYU40) GT labels from labels.ply."""
    scene_name = scene_path.name
    labels_ply = scene_path / f"{scene_name}_vh_clean_2.labels.ply"
    
    if not labels_ply.exists():
        print(f"  [SKIP] Missing labels PLY: {labels_ply.name}")
        return False
    
    plydata = PlyData.read(str(labels_ply))
    gt_labels = np.array(plydata['vertex']['label'])
    
    output_file = output_dir / f"{scene_name}.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(str(int(lbl)) for lbl in gt_labels))
    
    n_vertices = len(gt_labels)
    unique_ids = np.unique(gt_labels)
    print(f"  [OK] {scene_name}: {n_vertices:,} vtx, {len(unique_ids)} unique NYU40 IDs")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate ScanNet200 and ScanNet20 GT label text files"
    )
    parser.add_argument(
        '--scannet_root', required=True,
        help='Path to scannet_data directory containing scans/ folder'
    )
    parser.add_argument(
        '--tsv_path', default=None,
        help='Path to scannetv2-labels.combined.tsv (optional, uses fallback if not provided)'
    )
    parser.add_argument(
        '--scenes', nargs='+', default=None,
        help='Specific scenes to process (default: all scenes in scans/)'
    )
    parser.add_argument(
        '--skip_scannet20', action='store_true',
        help='Skip generating ScanNet20 (NYU40) labels'
    )
    args = parser.parse_args()
    
    scannet_root = Path(args.scannet_root)
    scans_dir = scannet_root / "scans"
    
    if not scans_dir.exists():
        print(f"[ERROR] scans/ directory not found at: {scans_dir}")
        print(f"[HINT] Make sure the path contains a 'scans' subfolder with scene directories.")
        sys.exit(1)
    
    # --- 1. Build label mapping ---
    tsv_path = args.tsv_path
    if tsv_path is None:
        # Search common locations
        candidates = [
            scannet_root / "scannetv2-labels.combined.tsv",
            Path(__file__).parent / "scannetv2-labels.combined.tsv",
            scannet_root.parent / "ScanNet" / "Tasks" / "Benchmark" / "scannetv2-labels.combined.tsv",
        ]
        tsv_path = next((str(p) for p in candidates if p.exists()), None)
    
    if tsv_path and os.path.exists(tsv_path):
        print(f"[Mode] Using TSV label map: {tsv_path}")
        raw_to_id = build_mapping_from_tsv(tsv_path)
    else:
        print(f"[Mode] TSV not found. Using built-in CLASS_LABELS_200 + alias mapping.")
        print(f"       (For best accuracy, provide --tsv_path scannetv2-labels.combined.tsv)")
        raw_to_id = build_mapping_from_constants()
    
    # --- 2. Find scenes ---
    if args.scenes:
        scene_names = args.scenes
    else:
        scene_names = sorted([
            d.name for d in scans_dir.iterdir() 
            if d.is_dir() and d.name.startswith("scene")
        ])
    
    print(f"\n[Config] Processing {len(scene_names)} scenes from: {scans_dir}")
    
    # --- 3. Output directories ---
    scannet200_gt_dir = scannet_root / "scannet200_gt"
    scannet200_gt_dir.mkdir(parents=True, exist_ok=True)
    
    semantic_gt_dir = scannet_root / "semantic_gt"
    semantic_gt_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 4. Generate ScanNet200 labels ---
    print(f"\n{'='*60}")
    print(f"GENERATING SCANNET200 GT LABELS")
    print(f"{'='*60}")
    
    success_200 = 0
    for scene_name in scene_names:
        scene_path = scans_dir / scene_name
        if not scene_path.exists():
            print(f"  [SKIP] Not found: {scene_name}")
            continue
        if generate_scannet200_labels(scene_path, raw_to_id, scannet200_gt_dir):
            success_200 += 1
    
    print(f"\n[Done] ScanNet200: {success_200}/{len(scene_names)} scenes -> {scannet200_gt_dir}")
    
    # --- 5. Generate ScanNet20 labels ---
    if not args.skip_scannet20:
        print(f"\n{'='*60}")
        print(f"GENERATING SCANNET20 (NYU40) GT LABELS")
        print(f"{'='*60}")
        
        success_20 = 0
        for scene_name in scene_names:
            scene_path = scans_dir / scene_name
            if not scene_path.exists():
                continue
            if generate_scannet20_labels(scene_path, semantic_gt_dir):
                success_20 += 1
        
        print(f"\n[Done] ScanNet20: {success_20}/{len(scene_names)} scenes -> {semantic_gt_dir}")
    
    # --- 6. Verification ---
    print(f"\n{'='*60}")
    print(f"VERIFICATION")
    print(f"{'='*60}")
    
    test_scene = scene_names[0] if scene_names else None
    if test_scene:
        txt_200 = scannet200_gt_dir / f"{test_scene}.txt"
        txt_20 = semantic_gt_dir / f"{test_scene}.txt"
        
        if txt_200.exists():
            labels_200 = np.loadtxt(str(txt_200), dtype=int)
            unique_200 = np.unique(labels_200)
            print(f"\n[Verify] {test_scene} ScanNet200:")
            print(f"  Vertices: {len(labels_200):,}")
            print(f"  Unique IDs: {len(unique_200)} (max={unique_200.max()})")
            print(f"  Labeled (>0): {np.sum(labels_200 > 0):,} ({100*np.sum(labels_200>0)/len(labels_200):.1f}%)")
            values, counts = np.unique(labels_200[labels_200 > 0], return_counts=True)
            top_idx = np.argsort(-counts)[:5]
            for i in top_idx:
                vid = int(values[i])
                # Find class name
                try:
                    cls_idx = list(VALID_CLASS_IDS_200).index(vid)
                    cls_name = CLASS_LABELS_200[cls_idx]
                except ValueError:
                    cls_name = "?"
                print(f"    ID {vid:4d} ({cls_name:20s}): {int(counts[i]):,} vertices")
        
        if txt_20.exists():
            labels_20 = np.loadtxt(str(txt_20), dtype=int)
            unique_20 = np.unique(labels_20)
            print(f"\n[Verify] {test_scene} ScanNet20 (NYU40):")
            print(f"  Vertices: {len(labels_20):,}")
            print(f"  Unique IDs: {len(unique_20)} (max={unique_20.max()})")
    
    print(f"\n[Complete] GT files ready for evaluation.")
    print(f"  ScanNet200: {scannet200_gt_dir}")
    print(f"  ScanNet20:  {semantic_gt_dir}")


if __name__ == "__main__":
    main()
