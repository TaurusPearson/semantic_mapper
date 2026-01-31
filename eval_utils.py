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
    
    if tree =="ball":
        tree = BallTree(points_3d)
    else:
        tree = KDTree(points_3d)
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
    for i in range(num_classes):
        if i not in ignore:
            label_name = labels[i]
            val_iou = iou_values[i]
            val_acc = acc_values[i]
            if not np.isnan(val_iou):
                per_class_lines.append('{0:<14s}: {1:>5.2%}    {2:>6.2%}'.format(label_name, val_iou, val_acc))

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
    # Scale figure size based on number of classes
    fig_size = max(10, n * 0.5) 
    fig, axs = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    
    axs.set_title(f"Normalized Confusion Matrix")     
    df_cm = pd.DataFrame(confmat_norm, index = labels, columns = labels)
    
    sn.heatmap(df_cm, annot=False, ax=axs, xticklabels=True, yticklabels=True, 
               fmt=".2f", cmap="Blues", square=True)
    
    # Adjust font size based on density
    font_size = 8 if n > 20 else 12
    sn.set(font_scale=1.0)
    axs.tick_params(axis='both', which='major', labelsize=font_size)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    if save:
        plt.savefig(output_path / "confmat.png", dpi=300)
        print(f"[IO] Confusion Matrix saved to {output_path / 'confmat.png'}")
    else:
        plt.show()
    plt.close()

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