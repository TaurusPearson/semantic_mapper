from typing import Any, Dict, List, Tuple
from sklearn.neighbors import BallTree
from matplotlib.colors import LogNorm
from scipy.spatial import KDTree
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sn 
import pandas as pd
import numpy as np
import torch 
import sys
import time
import os

# =========================================================
# 0. CLASS SPLITS
# =========================================================

# --- Replica 51-class taxonomy ---
# Head: structural elements + large furniture
# Common: mid-sized objects + household furniture
# Tail: small, rare, or infrastructure items

REPLICA_CLASS_SPLITS = {
    "head": [
        "wall", "ceiling", "floor", "chair", "blinds", "sofa", "table",
        "rug", "window", "lamp", "door", "pillow", "bench", "tv-screen",
        "cabinet", "pillar", "blanket",
    ],
    "common": [
        "tv-stand", "cushion", "bin", "vent", "bed", "stool", "picture",
        "indoor-plant", "desk", "comforter", "nightstand", "shelf", "vase",
        "plant-stand", "basket", "plate", "monitor",
    ],
    "tail": [
        "pipe", "panel", "desk-organizer", "wall-plug", "book", "box",
        "clock", "sculpture", "tissue-paper", "camera", "tablet", "pot",
        "bottle", "candle", "bowl", "cloth", "switch",
    ],
}

# --- ScanNet200 200-class taxonomy (official benchmark splits) ---

SCANNET200_CLASS_SPLITS = {
    "head": [
        "tv stand", "curtain", "blinds", "shower curtain", "bookshelf", "tv",
        "kitchen cabinet", "pillow", "lamp", "dresser", "monitor", "object",
        "ceiling", "board", "stove", "closet wall", "couch", "office chair",
        "kitchen counter", "shower", "closet", "doorframe", "sofa chair",
        "mailbox", "nightstand", "washing machine", "picture", "book", "sink",
        "recycling bin", "table", "backpack", "shower wall", "toilet", "copier",
        "counter", "stool", "refrigerator", "window", "file cabinet", "chair",
        "wall", "plant", "coffee table", "stairs", "armchair", "cabinet",
        "bathroom vanity", "bathroom stall", "mirror", "blackboard", "trash can",
        "stair rail", "box", "towel", "door", "clothes", "whiteboard", "bed",
        "floor", "bathtub", "desk", "wardrobe", "clothes dryer", "radiator",
        "shelf",
    ],
    "common": [
        "cushion", "end table", "dining table", "keyboard", "bag",
        "toilet paper", "printer", "blanket", "microwave", "shoe",
        "computer tower", "bottle", "bin", "ottoman", "bench", "basket",
        "fan", "laptop", "person", "paper towel dispenser", "oven", "rack",
        "piano", "suitcase", "rail", "container", "telephone", "stand",
        "light", "laundry basket", "pipe", "seat", "column", "bicycle",
        "ladder", "jacket", "storage bin", "coffee maker", "dishwasher",
        "machine", "mat", "windowsill", "bulletin board", "fireplace",
        "mini fridge", "water cooler", "shower door", "pillar", "ledge",
        "furniture", "cart", "decoration", "closet door", "vacuum cleaner",
        "dish rack", "range hood", "projector screen", "divider",
        "bathroom counter", "laundry hamper", "bathroom stall door",
        "ceiling light", "trash bin", "bathroom cabinet", "structure",
        "storage organizer", "potted plant", "mattress",
    ],
    "tail": [
        "paper", "plate", "soap dispenser", "bucket", "clock", "guitar",
        "toilet paper holder", "speaker", "cup", "paper towel roll", "bar",
        "toaster", "ironing board", "soap dish", "toilet paper dispenser",
        "fire extinguisher", "ball", "hat", "shower curtain rod",
        "paper cutter", "tray", "toaster oven", "mouse",
        "toilet seat cover dispenser", "storage container", "scale",
        "tissue box", "light switch", "crate", "power outlet", "sign",
        "projector", "candle", "plunger", "stuffed animal", "headphones",
        "broom", "guitar case", "dustpan", "hair dryer", "water bottle",
        "handicap bar", "purse", "vent", "shower floor", "water pitcher",
        "bowl", "paper bag", "alarm clock", "music stand",
        "laundry detergent", "dumbbell", "tube", "cd case", "closet rod",
        "coffee kettle", "shower head", "keyboard piano",
        "case of water bottles", "coat rack", "folded chair", "fire alarm",
        "power strip", "calendar", "poster", "luggage",
    ],
}

# =========================================================
# 1. GEOMETRIC MATCHING
# =========================================================

def match_labels_to_vtx(points_3d_labels: torch.Tensor, points_3d: torch.Tensor, mesh_vtx: torch.Tensor, filter_unasigned: bool = True, tree: str ="kd", verbose=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if filter_unasigned:
        assigned_mask = (points_3d_labels >-1).squeeze()
        if verbose:
            print(f"Assigned points {assigned_mask.sum()}, {assigned_mask.float().mean()*100:.1f}")
        points_3d_labels = points_3d_labels[assigned_mask]
        points_3d = points_3d[assigned_mask]
        assert len(points_3d_labels), "All points are unassigned"
    
    # Filter out NaN/Inf values that can occur from bad camera poses
    if isinstance(points_3d, torch.Tensor):
        valid_mask = torch.isfinite(points_3d).all(dim=1)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum().item()
            print(f"[Eval] Warning: Filtering {n_invalid} points with NaN/Inf values")
            points_3d = points_3d[valid_mask]
            points_3d_labels = points_3d_labels[valid_mask]
        points_3d_np = points_3d.cpu().numpy()
    else:
        valid_mask = np.isfinite(points_3d).all(axis=1)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            print(f"[Eval] Warning: Filtering {n_invalid} points with NaN/Inf values")
            points_3d = points_3d[valid_mask]
            points_3d_labels = points_3d_labels[valid_mask]
        points_3d_np = points_3d
    
    if tree =="ball":
        tree = BallTree(points_3d_np)
    else:
        tree = KDTree(points_3d_np)
    distances, indices = tree.query(mesh_vtx, k=5)

    labels = points_3d_labels[indices]
    mesh_labels = torch.mode(labels).values
    
    matched_instances_ids = torch.unique(mesh_labels)
    if not filter_unasigned:
        while len(matched_instances_ids) > 0 and matched_instances_ids[0]<0:
            matched_instances_ids = matched_instances_ids[1:]
        
    n_instances = len(matched_instances_ids)
    instance_idxs = torch.unsqueeze(matched_instances_ids, dim=1)
    mesh_instances_masks = torch.unsqueeze(mesh_labels,dim=0).expand(n_instances,-1) == instance_idxs

    return mesh_labels, mesh_instances_masks, matched_instances_ids

# =========================================================
# 2. SEMANTIC EVALUATION LOGIC
# =========================================================

def eval_semantics(output_path: str, gt_path: str, scenes: List[str], dataset_info: Dict[str, Any], mask_nan: bool = True, ignore_background: bool = False, verbose: bool = True, return_metrics = False) -> Tuple[np.ndarray, np.ndarray]:
    
    num_classes = dataset_info["num_classes"]
    map_to_reduced = dataset_info.get("map_to_reduced", None)
    labels = dataset_info["class_names"] if dataset_info.get("map_to_reduced", None) is None else dataset_info["class_names_reduced"]
    ignore = dataset_info.get("ignore",[]).copy()
    
    if ignore_background:
        if map_to_reduced:
            assert dataset_info.get("background_reduced_ids", None), "To ignore background a list of idxs corresponding to background ids id required!"
            ignore.extend(dataset_info["background_reduced_ids"])
        else:
            assert dataset_info["background_ids"], "To ignore background a list of idxs corresponding to background ids id required!"
            ignore.extend(dataset_info["background_ids"])

    pr_files = []
    gt_files = []
    for scene in scenes:
        pr_files.append(Path(output_path)/ f'{scene}.txt')
        gt_files.append(Path(gt_path) / f'{scene}.txt')
    
    confusion = np.zeros([len(scenes), num_classes, num_classes], dtype=np.ulonglong)

    if verbose:
        print(f'Evaluating {len(pr_files)} scans...')
        print(f'Pred Path: {pr_files[0]}')
        print(f'GT Path:   {gt_files[0]}')

    for i in range(len(pr_files)):
        evaluate_scan(pr_files[i], gt_files[i], confusion[i], map_to_reduced, ignore)
        if verbose:
            sys.stdout.write("\rscans processed: {}".format(i+1))
            sys.stdout.flush()

    # Ensure output_path is Path object
    if isinstance(output_path, str):
        output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- SAVE TEXT REPORT ---
    report_lines = []

    # Per scene:
    for i in range(len(scenes)):
        iou_values, iou_valid_mask, weights_values, acc_values, acc_valid_mask = iou_acc_from_confmat(confusion[i], num_classes, ignore, mask_nan, False, labels)
        if verbose:
            line = f"\nScene: {scenes[i]}"
            print(line)
            report_lines.append(line)
            
            if weights_values[iou_valid_mask].sum() > 0:
                miou = np.mean(iou_values[iou_valid_mask])
                macc = np.mean(acc_values[acc_valid_mask])
                fiou = np.sum(iou_values[iou_valid_mask]*weights_values[iou_valid_mask])/weights_values[iou_valid_mask].sum()
                facc = np.sum(acc_values[acc_valid_mask]*weights_values[acc_valid_mask])/weights_values[acc_valid_mask].sum()
                
                line1 = f'mIoU: \t {miou:.2%}; mAcc: \t {macc:.2%}'
                line2 = f'f-mIoU: \t {fiou:.2%}; f-mAcc: \t {facc:.2%}\n'
                print(line1)
                print(line2)
                report_lines.append(line1)
                report_lines.append(line2)
            else:
                line = 'mIoU: NaN (No valid classes found)'
                print(line)
                report_lines.append(line)

    confusion = confusion.sum(0)
    iou_values, iou_valid_mask, weights_values, acc_values, acc_valid_mask = iou_acc_from_confmat(confusion, num_classes, ignore, mask_nan, verbose, labels)
    
    # Save per-class stats to list for report
    per_class_lines = []
    per_class_lines.append('\n classes \t IoU \t Acc')
    per_class_lines.append('----------------------------')
    count = 0
    for i in range(num_classes):
        if i not in ignore:
            label_name = labels[i]
            val_iou = iou_values[count]
            val_acc = acc_values[count]
            count += 1
            # Show ALL classes, including NaN (display as 0.00%)
            display_iou = 0.0 if np.isnan(val_iou) else val_iou
            display_acc = 0.0 if np.isnan(val_acc) else val_acc
            per_class_lines.append('{0:<14s}: {1:>5.2%}    {2:>6.2%}'.format(label_name, display_iou, display_acc))

    # Add per-class stats to main report
    report_lines.extend(per_class_lines)

    metrics = {
        "iou": round(np.mean(iou_values[iou_valid_mask]),3),
        "acc": round(np.mean(acc_values[acc_valid_mask]),3),
    }
    
    if weights_values[iou_valid_mask].sum() > 0:
        metrics["fiou"] = round(np.sum(iou_values[iou_valid_mask]*weights_values[iou_valid_mask])/weights_values[iou_valid_mask].sum(), 3)
        metrics["facc"] = round(np.sum(acc_values[acc_valid_mask]*weights_values[acc_valid_mask])/weights_values[acc_valid_mask].sum(), 3)
    else:
        metrics["fiou"] = 0.0
        metrics["facc"] = 0.0

    if len(iou_values) >= 50:
        thirds = len(iou_values)//3
        for split, i in [['head',0], ['comm',1], ['tail',2]]:
            idx_i, idx_e = thirds * i,thirds * (i + 1)
            valid_slice = iou_valid_mask[idx_i:idx_e]
            if valid_slice.any():
                metrics[f"iou_{split}"] = round(np.mean(iou_values[idx_i:idx_e][valid_slice]), 3)
                metrics[f"acc_{split}"] = round(np.mean(acc_values[idx_i:idx_e][acc_valid_mask[idx_i:idx_e]]), 3)
            else:
                metrics[f"iou_{split}"] = 0.0
                metrics[f"acc_{split}"] = 0.0

    if verbose:
        final_line = f"\nmIoU: \t {metrics['iou']:.2%}; mAcc: \t {metrics['acc']:.2%}\n "
        print(final_line)
        report_lines.append(final_line)
        
        # --- WRITE REPORT TO FILE ---
        report_path = output_path / "metrics_summary.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        print(f"[IO] Metrics summary saved to {report_path}")

        # Write CSV style stats
        with open(output_path / "statistics.txt", "w") as f:
            f.write(f"label, acc, iou, \n")
            count = 0
            for i in range(len(labels)):
                if i not in ignore:
                    f.write(f"{labels[i]}, {acc_values[count]}, {iou_values[count]}, \n")
                    count +=1

    if verbose and isinstance(output_path, Path):
        plot_metrics(iou_values, acc_values, labels, output_path, ignore)
        plot_confmat(confusion, labels, output_path)
        eval_class_splits(confusion, labels, output_path, ignore)
        
    if return_metrics:
        return metrics, confusion
    return np.mean(iou_values[iou_valid_mask]), confusion

# =========================================================
# 3. EVALUATION HELPERS
# =========================================================

def evaluate_scan(pr_file: str, gt_file: str, confusion: np.ndarray, map_gt_ids: Dict[int, int] | None = None, ignore: List = []) -> None:
    pr_ids = np.array(process_txt(pr_file), dtype=np.int64)
    gt_ids = np.array(process_txt(gt_file)).astype(np.int64)

    # Debug print to check what we actually loaded
    if len(gt_ids) == 0:
        print(f"[Warn] GT File is empty or unreadable: {gt_file}")
    if len(pr_ids) == 0:
        print(f"[Warn] Prediction File is empty: {pr_file}")

    if map_gt_ids is not None:
        assert isinstance(map_gt_ids, dict), "map_gt_ids must be a dict"
        
        def mapper(x): return map_gt_ids.get(x, -1)
        gt_ids = np.vectorize(mapper, otypes=[np.int64])(gt_ids)

    # Sanity checks
    if not pr_ids.shape == gt_ids.shape:
        min_len = min(len(pr_ids), len(gt_ids))
        pr_ids = pr_ids[:min_len]
        gt_ids = gt_ids[:min_len]
        
    update_confmat(confusion, gt_ids, pr_ids, ignore)

def update_confmat(confusion: np.ndarray, gt_ids: List[int], pr_ids: List[int], ignore: List[int]) -> None:
    valid_mask = np.ones_like(gt_ids, dtype=bool)
    for ig in ignore:
        valid_mask &= (gt_ids != ig)
        valid_mask &= (pr_ids != ig)  # Also filter predictions with ignore values
    
    # Additional safety: filter out any negative indices to prevent wrap-around indexing
    valid_mask &= (gt_ids >= 0) & (pr_ids >= 0)
    
    # Also filter out indices that exceed confusion matrix bounds
    num_classes = confusion.shape[0]
    valid_mask &= (gt_ids < num_classes) & (pr_ids < num_classes)
    
    gt_valid = gt_ids[valid_mask]
    pr_valid = pr_ids[valid_mask]
    
    if len(gt_valid) > 0:
        np.add.at(confusion, (gt_valid, pr_valid), 1)

def get_iou(label_id: int, confusion: np.ndarray) -> Tuple[float, float]:
    tp = np.longlong(confusion[label_id, label_id])
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    fp = np.longlong(confusion[:, label_id].sum()) - tp
    denom = float(tp + fp + fn)
    if denom == 0:
        return (float('nan'),float('nan'))
    iou = tp / denom
    acc = tp / max(float(tp + fn), 1e-6)
    return (iou, acc)

def iou_acc_from_confmat(confmat: np.ndarray, num_classes: int, ignore: List[int], mask_nan: bool = True, verbose: bool = False, labels: List[str] = None):
    # NOTE: verbose printing removed here to prevent double printing. 
    # Logic moved to eval_semantics to capture output for file writing.
    list_iou, list_acc, list_weight = [], [], []
    for i in range(num_classes):
        if i not in ignore:
            iou, acc = get_iou(i, confmat)
            list_iou.append(iou)
            list_acc.append(acc)
            list_weight.append(confmat[i].sum()) 

    iou_values = np.array(list_iou)
    acc_values = np.array(list_acc)
    weights_values = np.array(list_weight)

    if mask_nan:
        iou_valid_mask = ~np.isnan(iou_values)
        acc_valid_mask = ~np.isnan(acc_values)
    else:
        iou_valid_mask = np.ones_like(iou_values,dtype=bool)
        acc_valid_mask = np.ones_like(acc_values,dtype=bool)
    return iou_values, iou_valid_mask, weights_values, acc_values, acc_valid_mask

def process_txt(filename: str) -> List[str]:
    if not os.path.exists(filename):
        print(f"[IO] Error: File not found {filename}")
        return []

    try:
        with open(filename, 'r', encoding='utf-8-sig') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]
        return lines
    except Exception as e:
        print(f"[IO] Failed to read {filename}: {e}")
        return []

# =========================================================
# 4. PLOTTING
# =========================================================

def plot_metrics(iou_values: np.ndarray, acc_values: np.ndarray, labels: List, output_path: Path, ignore: List = []) -> None:
    labels = [label for i,label in enumerate(labels) if i not in ignore]
    
    # Filter out NaNs for plotting to avoid errors
    valid_indices = ~np.isnan(iou_values)
    iou_values = iou_values[valid_indices]
    acc_values = acc_values[valid_indices]
    
    # Filter labels to match valid indices length
    # Note: 'iou_values' was computed filtering ignored classes, so we just need to filter NaNs
    labels = [labels[i] for i in range(len(labels)) if valid_indices[i]]

    if len(labels) == 0:
        return

    idx = np.asarray([0.4+i*3 for i in range(len(labels))])
    width = 1.
    ratio = max(10/len(labels), 0.5) # Prevent too small ratio
    
    fig, axs = plt.subplots(figsize=(max(20, len(labels)*0.5), 10))
    axs.margins(x=0.01)
    axs.set_title("IoU and Acc")
    # axs.set_box_aspect(ratio) # Removed to let figsize control aspect
    
    axs.bar(idx, iou_values, width=width, label='IoU')
    axs.bar(idx+width, acc_values, width=width, label='Acc')
    
    axs.set_xticks(idx + width/2)
    axs.set_xticklabels(labels, rotation=85, ha='right')
    axs.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path / "plot_iou_acc.png", dpi=300)
    plt.close()
    print(f"[IO] Plot saved to {output_path / 'plot_iou_acc.png'}")

def plot_confmat(confmat: np.ndarray, labels: List, output_path: Path, save: bool = True) -> None:
    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        confmat_norm = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
    confmat_norm = np.nan_to_num(confmat_norm)

    n = len(labels)

    # For large class counts (e.g. ScanNet200), filter to active classes only
    if n > 60:
        row_sums = confmat.sum(axis=1)
        col_sums = confmat.sum(axis=0)
        active = np.where((row_sums > 0) | (col_sums > 0))[0]
        if len(active) == 0:
            print(f"[Warn] Confusion matrix is empty, skipping plot.")
            return
        active_labels = [labels[i] for i in active]
        confmat_norm = confmat_norm[np.ix_(active, active)]
        n_display = len(active)
        title_suffix = f" ({n_display}/{n} active classes)"
    else:
        active_labels = labels
        n_display = n
        title_suffix = ""

    # Scale figure size based on number of classes, capped for sanity
    fig_size = min(40, max(10, n_display * 0.5))
    dpi = 150 if n_display > 50 else 300
    fig, axs = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    
    axs.set_title(f"Normalized Confusion Matrix{title_suffix}")     
    df_cm = pd.DataFrame(confmat_norm, index=active_labels, columns=active_labels)
    
    sn.heatmap(df_cm, annot=False, ax=axs, xticklabels=True, yticklabels=True, 
               fmt=".2f", cmap="Blues", square=(n_display <= 60))
    
    # Adjust font size based on density
    font_size = 5 if n_display > 80 else (8 if n_display > 20 else 12)
    sn.set(font_scale=1.0)
    axs.tick_params(axis='both', which='major', labelsize=font_size)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    if save:
        plt.savefig(output_path / "confmat.png", dpi=dpi)
        print(f"[IO] Confusion Matrix saved to {output_path / 'confmat.png'}")
    else:
        plt.show()
    plt.close()

def plot_confmat_tail(confmat: np.ndarray, labels: List, iou_values: np.ndarray,
                      output_path: Path, ignore: List[int] = [],
                      iou_threshold: float = 0.05, top_k_confusions: int = 8) -> None:
    """Generate focused confusion matrix and text report for tail classes (low IoU)."""
    num_classes = len(labels)

    # Map full class index → iou_values index (iou_values skips ignored classes)
    iou_idx = 0
    class_iou = {}
    for i in range(num_classes):
        if i not in ignore:
            class_iou[i] = iou_values[iou_idx] if iou_idx < len(iou_values) else float('nan')
            iou_idx += 1

    # Identify tail classes: IoU < threshold and not ignored
    tail_indices = [i for i, iou in class_iou.items()
                    if not np.isnan(iou) and iou < iou_threshold]

    if not tail_indices:
        print(f"[Eval] No tail classes found with IoU < {iou_threshold:.0%}")
        return

    tail_labels = [labels[i] for i in tail_indices]

    # Normalize confusion matrix rows (GT → Pred distribution)
    with np.errstate(divide='ignore', invalid='ignore'):
        confmat_norm = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
    confmat_norm = np.nan_to_num(confmat_norm)

    # Collect all columns that any tail class gets confused with (>1%)
    all_confused_indices = set(tail_indices)
    for ti in tail_indices:
        row = confmat_norm[ti]
        top_preds = np.argsort(row)[::-1][:top_k_confusions]
        for pi in top_preds:
            if row[pi] > 0.01:
                all_confused_indices.add(int(pi))

    display_cols = sorted(all_confused_indices)
    display_col_labels = [labels[i] for i in display_cols]

    # Extract sub-matrix: tail rows × relevant columns
    sub_matrix = confmat_norm[np.ix_(tail_indices, display_cols)]

    # --- Heatmap ---
    fig_h = max(4, len(tail_indices) * 0.6)
    fig_w = max(8, len(display_cols) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    df = pd.DataFrame(sub_matrix, index=tail_labels, columns=display_col_labels)
    sn.heatmap(df, annot=True, fmt=".2f", cmap="OrRd", ax=ax,
               xticklabels=True, yticklabels=True, square=False,
               linewidths=0.5, linecolor='gray',
               cbar_kws={'label': 'Fraction of GT points'})

    ax.set_title(f"Tail Class Confusion (IoU < {iou_threshold:.0%})", fontsize=14)
    ax.set_ylabel("Ground Truth (tail classes)")
    ax.set_xlabel("Predicted As")

    font_size = 9 if len(display_cols) > 15 else 11
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path / "confmat_tail.png", dpi=200)
    plt.close()
    print(f"[IO] Tail confusion matrix saved to {output_path / 'confmat_tail.png'}")

    # --- Text Report ---
    report_path = output_path / "confmat_tail.txt"
    with open(report_path, "w") as f:
        f.write(f"TAIL CLASS CONFUSION REPORT (IoU < {iou_threshold:.0%})\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Tail classes identified: {len(tail_indices)}\n\n")

        for ti in tail_indices:
            name = labels[ti]
            iou = class_iou[ti]
            row_sum = confmat[ti].sum()

            f.write(f"--- {name} (IoU: {iou:.2%}, GT points: {int(row_sum):,}) ---\n")

            if row_sum == 0:
                f.write(f"  NO GT POINTS in evaluation (class absent or fully unmapped)\n\n")
                continue

            # Where do GT points end up? (row distribution)
            row = confmat_norm[ti]
            sorted_preds = np.argsort(row)[::-1]

            f.write(f"  Predicted as:\n")
            for pi in sorted_preds:
                if row[pi] < 0.005:
                    break
                pred_name = labels[pi]
                n_points = int(confmat[ti, pi])
                marker = " <-- CORRECT" if pi == ti else ""
                f.write(f"    {pred_name:20s}: {row[pi]:6.1%} ({n_points:>10,} pts){marker}\n")

            # Reverse: what GT classes get wrongly predicted AS this tail class?
            col = confmat_norm[:, ti]
            sources = [(si, col[si]) for si in np.argsort(col)[::-1]
                       if col[si] > 0.01 and si != ti]

            if sources:
                f.write(f"  Other classes wrongly predicted as '{name}':\n")
                for si, frac in sources[:5]:
                    f.write(f"    {labels[si]:20s}: {frac:6.1%} ({int(confmat[si, ti]):>10,} pts)\n")

            f.write("\n")

    print(f"[IO] Tail confusion report saved to {report_path}")

def eval_class_splits(confmat: np.ndarray, labels: List, output_path: Path,
                      ignore: List[int] = [],
                      splits: Dict[str, List[str]] = None) -> Dict[str, Dict]:
    """
    Compute per-split metrics and generate per-split confusion matrices.
    
    IoU/Acc are computed from the FULL confusion matrix (cross-split FP/FN counted),
    then averaged only over classes in each split.
    Splits are defined by class name strings, resolved to indices at runtime.
    """
    if splits is None:
        # Auto-detect: pick whichever split dict matches more class names
        name_set = set(labels)
        replica_hits = sum(1 for names in REPLICA_CLASS_SPLITS.values() for n in names if n in name_set)
        scannet_hits = sum(1 for names in SCANNET200_CLASS_SPLITS.values() for n in names if n in name_set)
        if scannet_hits > replica_hits:
            splits = SCANNET200_CLASS_SPLITS
            print(f"[Eval] Auto-detected ScanNet200 class splits ({scannet_hits} matches)")
        else:
            splits = REPLICA_CLASS_SPLITS
            print(f"[Eval] Auto-detected Replica class splits ({replica_hits} matches)")

    # Build name -> index lookup
    name_to_idx = {name: i for i, name in enumerate(labels)}

    # Early exit if no split names match the labels
    total_matched = sum(1 for names in splits.values() for n in names if n in name_to_idx)
    if total_matched == 0:
        print(f"[Eval] Skipping class split evaluation — no split names match the {len(labels)} class labels.")
        return {}

    # Compute per-class IoU and Acc from FULL confusion matrix
    all_iou = {}
    all_acc = {}
    all_weight = {}
    for i in range(len(labels)):
        if i not in ignore:
            iou, acc = get_iou(i, confmat)
            all_iou[i] = iou
            all_acc[i] = acc
            all_weight[i] = float(confmat[i].sum())

    # Normalize full confusion matrix for heatmaps
    with np.errstate(divide='ignore', invalid='ignore'):
        confmat_norm = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
    confmat_norm = np.nan_to_num(confmat_norm)

    split_metrics = {}
    report_lines = []
    report_lines.append("CLASS SPLIT EVALUATION REPORT")
    report_lines.append("=" * 70)

    for split_name, split_names in splits.items():
        # Resolve class names to indices
        split_indices = [name_to_idx[n] for n in split_names if n in name_to_idx]
        # Filter to non-ignored classes in this split
        valid_indices = [i for i in split_indices if i not in ignore and i in all_iou]

        if not valid_indices:
            report_lines.append(f"\n--- {split_name.upper()} ({len(split_indices)} classes) ---")
            report_lines.append("  No valid classes in this split.")
            continue

        # Per-class IoU/Acc for this split
        split_iou = np.array([all_iou[i] for i in valid_indices])
        split_acc = np.array([all_acc[i] for i in valid_indices])
        split_wt = np.array([all_weight[i] for i in valid_indices])
        split_labels = [labels[i] for i in valid_indices]

        # Mask NaN (classes with 0 GT points)
        iou_valid = ~np.isnan(split_iou)
        acc_valid = ~np.isnan(split_acc)

        m_iou = float(np.mean(split_iou[iou_valid])) if iou_valid.any() else 0.0
        m_acc = float(np.mean(split_acc[acc_valid])) if acc_valid.any() else 0.0

        wt_sum = split_wt[iou_valid].sum()
        f_iou = float(np.sum(split_iou[iou_valid] * split_wt[iou_valid]) / wt_sum) if wt_sum > 0 else 0.0
        wt_sum_a = split_wt[acc_valid].sum()
        f_acc = float(np.sum(split_acc[acc_valid] * split_wt[acc_valid]) / wt_sum_a) if wt_sum_a > 0 else 0.0

        n_zero = int(np.sum(split_iou[iou_valid] == 0))

        split_metrics[split_name] = {
            "mIoU": round(m_iou, 4),
            "mAcc": round(m_acc, 4),
            "f-mIoU": round(f_iou, 4),
            "f-mAcc": round(f_acc, 4),
            "n_classes": len(valid_indices),
            "n_zero_iou": n_zero,
        }

        # --- Text Report ---
        report_lines.append(f"\n--- {split_name.upper()} ({len(valid_indices)} classes, {n_zero} at 0% IoU) ---")
        report_lines.append(f"  mIoU: {m_iou:.2%}    mAcc: {m_acc:.2%}")
        report_lines.append(f"  f-mIoU: {f_iou:.2%}  f-mAcc: {f_acc:.2%}")
        report_lines.append(f"  {'Class':<20s}  {'IoU':>7s}  {'Acc':>7s}  {'GT pts':>12s}")
        report_lines.append(f"  {'-'*50}")

        for idx, ci in enumerate(valid_indices):
            iou_str = f"{split_iou[idx]:.2%}" if not np.isnan(split_iou[idx]) else "  NaN"
            acc_str = f"{split_acc[idx]:.2%}" if not np.isnan(split_acc[idx]) else "  NaN"
            report_lines.append(f"  {split_labels[idx]:<20s}  {iou_str:>7s}  {acc_str:>7s}  {int(split_wt[idx]):>12,}")

        # --- Per-split Confusion Matrix Heatmap ---
        # Per-scene filtering: only show classes that have GT points in this scene
        active_indices = [i for i in valid_indices if confmat[i].sum() > 0]
        active_names = [labels[i] for i in active_indices]
        
        if not active_indices:
            print(f"[Eval] No GT classes in {split_name} split for this scene (0/{len(valid_indices)} present), skipping heatmap.")
            continue
        
        print(f"[Eval] {split_name.upper()}: {len(active_indices)}/{len(valid_indices)} classes with GT: {active_names[:5]}{'...' if len(active_names) > 5 else ''}")
        
        # Row labels = only active classes (per-scene filtering)
        row_labels = [labels[i] for i in active_indices]

        # Columns: active split classes + top confused-with classes
        all_col_indices = set(active_indices)
        for ci in active_indices:
            row = confmat_norm[ci]
            top_preds = np.argsort(row)[::-1][:10]
            for pi in top_preds:
                if row[pi] > 0.01:
                    all_col_indices.add(int(pi))

        display_cols = sorted(all_col_indices)
        display_col_labels = [labels[i] for i in display_cols]

        sub_matrix = confmat_norm[np.ix_(active_indices, display_cols)]

        n_rows = len(active_indices)
        n_cols = len(display_cols)
        fig_h = min(35, max(5, n_rows * 0.55))
        fig_w = min(40, max(8, n_cols * 0.55))
        split_dpi = 150 if n_cols > 60 else 200
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        cmap = {"head": "Blues", "common": "Greens", "tail": "OrRd"}.get(split_name, "Blues")
        df = pd.DataFrame(sub_matrix, index=row_labels, columns=display_col_labels)

        # Determine max value for color scaling
        vmax_val = float(sub_matrix.max()) if sub_matrix.max() > 0 else 1.0
        
        annot = n_cols <= 25
        sn.heatmap(df, annot=annot, fmt=".2f" if annot else "", cmap=cmap, ax=ax,
                   xticklabels=True, yticklabels=True, square=False,
                   vmin=0, vmax=max(0.01, vmax_val),
                   linewidths=0.3 if n_cols <= 60 else 0,
                   linecolor='gray',
                   cbar_kws={'label': 'Fraction of GT points'})

        # Title shows per-scene active count vs total split size
        n_total_split = len(valid_indices)
        title_note = f" — {n_rows}/{n_total_split} present in scene"
        ax.set_title(f"{split_name.upper()} Class Confusion{title_note}", fontsize=14)
        ax.set_ylabel("Ground Truth")
        ax.set_xlabel("Predicted As")

        font_size = 5 if n_cols > 80 else (7 if n_cols > 40 else (8 if n_cols > 20 else 10))
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        fname = f"confmat_{split_name}.png"
        plt.savefig(output_path / fname, dpi=split_dpi)
        plt.close()
        print(f"[IO] {split_name.upper()} confusion matrix saved to {output_path / fname}")

    # --- Summary comparison ---
    report_lines.append(f"\n{'='*70}")
    report_lines.append(f"{'Split':<10s}  {'mIoU':>8s}  {'mAcc':>8s}  {'f-mIoU':>8s}  {'f-mAcc':>8s}  {'#cls':>5s}  {'#0%':>4s}")
    report_lines.append(f"{'-'*55}")
    for sname, sm in split_metrics.items():
        report_lines.append(
            f"{sname:<10s}  {sm['mIoU']:>7.2%}  {sm['mAcc']:>7.2%}  "
            f"{sm['f-mIoU']:>7.2%}  {sm['f-mAcc']:>7.2%}  {sm['n_classes']:>5d}  {sm['n_zero_iou']:>4d}"
        )

    # Write combined report
    report_path = output_path / "class_splits_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[IO] Class splits report saved to {report_path}")

    return split_metrics

# =========================================================
# 5. PERFORMANCE MONITOR
# =========================================================

class PerformanceMonitor:
    def __init__(self):
        self.timings = {
            "total": [],
            "tracking": [],
            "mapping": [],
            "semantics": [],
            "visualization": []
        }
        self.memory = {
            "vram_allocated": [],
            "vram_reserved": []
        }
        self.gpu_util = []
        
        self.has_nvml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.has_nvml = True
        except:
            pass

        self.current_frame_start = 0
        self.components_start = {}

    def start_frame(self):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        self.current_frame_start = time.time()
        
        if self.has_nvml:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.gpu_util.append(util.gpu)
            except: pass

        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**2 
            mem_res = torch.cuda.memory_reserved() / 1024**2    
            self.memory["vram_allocated"].append(mem_alloc)
            self.memory["vram_reserved"].append(mem_res)

    def end_frame(self):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        dt = time.time() - self.current_frame_start
        self.timings["total"].append(dt)

    def start_component(self, name):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        self.components_start[name] = time.time()

    def end_component(self, name):
        if name in self.components_start:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            dt = time.time() - self.components_start[name]
            self.timings[name].append(dt)

    def report(self):
        N = len(self.timings["total"])
        if N == 0: return

        avg_fps = 1.0 / np.mean(self.timings["total"])
        avg_vram = np.mean(self.memory["vram_allocated"]) if self.memory["vram_allocated"] else 0
        max_vram = np.max(self.memory["vram_allocated"]) if self.memory["vram_allocated"] else 0
        
        print(f"\n=== EFFICIENCY METRICS ===")
        print(f"Total Frames: {N}")
        print(f"Average FPS:  {avg_fps:.2f} Hz")
        print(f"Avg VRAM:     {avg_vram:.0f} MB")
        print(f"Peak VRAM:    {max_vram:.0f} MB")
        
        if self.gpu_util:
            print(f"Avg GPU Util: {np.mean(self.gpu_util):.1f}%")

        print("\n--- Latency Breakdown (ms) ---")
        print(f"{'Component':<15} | {'Mean':<8} | {'Std':<8} | {'% of Frame'}")
        print("-" * 45)
        
        total_mean = np.mean(self.timings["total"])
        
        for comp, times in self.timings.items():
            if comp == "total" or len(times) == 0: continue
            
            mean_ms = np.mean(times) * 1000
            std_ms = np.std(times) * 1000
            amortized_mean = (np.sum(times) / N) 
            pct = (amortized_mean / total_mean) * 100
            
            print(f"{comp:<15} | {mean_ms:<8.1f} | {std_ms:<8.1f} | {pct:.1f}%")
        print("============================\n")