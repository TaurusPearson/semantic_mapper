"""
ScanNet Dataset Utilities

Provides:
1. SensorData - Extract RGB, Depth, Poses from .sens files (Official ScanNet format)
2. ScanNet Dataset class for the pipeline
3. Label mapping utilities for ScanNet20/ScanNet200 benchmarks

Based on official ScanNet SensReader: https://github.com/ScanNet/ScanNet/tree/master/SensReader/python
"""

import os
import struct
import zlib
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import imageio
import png


# ============================================================================
# OFFICIAL SCANNET SENSOR DATA READER (Python 3 compatible)
# ============================================================================

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1: 'unknown', 0: 'raw_ushort', 1: 'zlib_ushort', 2: 'occi_ushort'}


class RGBDFrame:
    """Single RGBD frame from .sens file."""
    
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack('f' * 16, file_handle.read(16 * 4)), 
            dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
            return zlib.decompress(self.depth_data)
        else:
            raise ValueError(f"Unknown depth compression: {compression_type}")

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
            return imageio.imread(self.color_data)
        else:
            raise ValueError(f"Unknown color compression: {compression_type}")


class SensorData:
    """
    Official ScanNet .sens file reader (Python 3 compatible).
    
    Reads ScanNet .sens files and extracts:
    - RGB frames (color/*.jpg)
    - Depth frames (depth/*.png) 
    - Camera poses (pose/*.txt)
    - Intrinsics
    """

    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version, f"Expected version {self.version}, got {version}"
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = f.read(strlen).decode('utf-8')
            self.intrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f'[SensorData] Exporting {len(self.frames) // frame_skip} depth frames to {output_path}')
        for f_idx in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[f_idx].decompress_depth(self.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            if image_size is not None:
                depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
            # Write 16-bit PNG
            out_idx = f_idx // frame_skip
            with open(os.path.join(output_path, f'{out_idx}.png'), 'wb') as f:
                writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16, greyscale=True)
                writer.write(f, depth.tolist())

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f'[SensorData] Exporting {len(self.frames) // frame_skip} color frames to {output_path}')
        for f_idx in range(0, len(self.frames), frame_skip):
            color = self.frames[f_idx].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
            out_idx = f_idx // frame_skip
            imageio.imwrite(os.path.join(output_path, f'{out_idx}.jpg'), color)

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f'[SensorData] Exporting {len(self.frames) // frame_skip} camera poses to {output_path}')
        for f_idx in range(0, len(self.frames), frame_skip):
            out_idx = f_idx // frame_skip
            np.savetxt(os.path.join(output_path, f'{out_idx}.txt'), self.frames[f_idx].camera_to_world)

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f'[SensorData] Exporting camera intrinsics to {output_path}')
        np.savetxt(os.path.join(output_path, 'intrinsic_color.txt'), self.intrinsic_color)
        np.savetxt(os.path.join(output_path, 'extrinsic_color.txt'), self.extrinsic_color)
        np.savetxt(os.path.join(output_path, 'intrinsic_depth.txt'), self.intrinsic_depth)
        np.savetxt(os.path.join(output_path, 'extrinsic_depth.txt'), self.extrinsic_depth)


def extract_scannet_scene(scene_path: str, output_path: Optional[str] = None,
                          max_frames: int = -1, frame_skip: int = 1) -> str:
    """
    Convenience function to extract a ScanNet scene.
    
    Args:
        scene_path: Path to scene directory containing .sens file
        output_path: Where to extract (default: same as scene_path)
        max_frames: Max frames to extract (-1 for all)
        frame_skip: Extract every Nth frame
        
    Returns:
        Path to extracted data
    """
    scene_path = Path(scene_path)
    scene_name = scene_path.name
    
    # Find .sens file
    sens_files = list(scene_path.glob("*.sens"))
    if not sens_files:
        raise FileNotFoundError(f"No .sens file found in {scene_path}")
    
    sens_file = sens_files[0]
    print(f"[ScanNet] Found: {sens_file.name}")
    
    # Output directory
    if output_path is None:
        output_path = scene_path
    
    # Check if already extracted
    if (Path(output_path) / "color").exists():
        n_existing = len(list((Path(output_path) / "color").glob("*.jpg")))
        if n_existing > 0:
            print(f"[ScanNet] Already extracted ({n_existing} frames). Skipping.")
            return str(output_path)
    
    # Extract using official SensorData reader
    print(f"[ScanNet] Loading .sens file (this may take a while for large scenes)...")
    sensor_data = SensorData(str(sens_file))
    print(f"[ScanNet] Loaded {len(sensor_data.frames)} frames, extracting...")
    
    # Export all components
    output_path = Path(output_path)
    sensor_data.export_color_images(str(output_path / "color"), frame_skip=frame_skip)
    sensor_data.export_depth_images(str(output_path / "depth"), frame_skip=frame_skip)
    sensor_data.export_poses(str(output_path / "pose"), frame_skip=frame_skip)
    sensor_data.export_intrinsics(str(output_path / "intrinsic"))
    
    # Save metadata
    import json
    meta = {
        'num_frames': len(sensor_data.frames) // frame_skip,
        'color_width': sensor_data.color_width,
        'color_height': sensor_data.color_height,
        'depth_width': sensor_data.depth_width,
        'depth_height': sensor_data.depth_height,
        'depth_shift': sensor_data.depth_shift,
        'fx': float(sensor_data.intrinsic_depth[0, 0]),
        'fy': float(sensor_data.intrinsic_depth[1, 1]),
        'cx': float(sensor_data.intrinsic_depth[0, 2]),
        'cy': float(sensor_data.intrinsic_depth[1, 2])
    }
    with open(output_path / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"[ScanNet] Extraction complete: {meta['num_frames']} frames")
    print(f"[ScanNet] Intrinsics: fx={meta['fx']:.1f}, fy={meta['fy']:.1f}, cx={meta['cx']:.1f}, cy={meta['cy']:.1f}")
    
    return str(output_path)


# ============================================================================
# SCANNET LABEL MAPPINGS
# ============================================================================

# ScanNet20 benchmark (NYU40 subset)
SCANNET20_VALID_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
SCANNET20_LABELS = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "furniture"
]

# ScanNet200 benchmark
SCANNET200_VALID_IDS = [
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
]

SCANNET200_LABELS = [
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
]

# Color maps
SCANNET20_COLORS = {
    0: [174.0, 199.0, 232.0],  # wall
    1: [152.0, 223.0, 138.0],  # floor
    2: [31.0, 119.0, 180.0],   # cabinet
    3: [255.0, 187.0, 120.0],  # bed
    4: [188.0, 189.0, 34.0],   # chair
    5: [140.0, 86.0, 75.0],    # sofa
    6: [255.0, 152.0, 150.0],  # table
    7: [214.0, 39.0, 40.0],    # door
    8: [197.0, 176.0, 213.0],  # window
    9: [148.0, 103.0, 189.0],  # bookshelf
    10: [196.0, 156.0, 148.0], # picture
    11: [23.0, 190.0, 207.0],  # counter
    12: [247.0, 182.0, 210.0], # desk
    13: [219.0, 219.0, 141.0], # curtain
    14: [255.0, 127.0, 14.0],  # refrigerator
    15: [158.0, 218.0, 229.0], # shower curtain
    16: [44.0, 160.0, 44.0],   # toilet
    17: [112.0, 128.0, 144.0], # sink
    18: [227.0, 119.0, 194.0], # bathtub
    19: [82.0, 84.0, 163.0],   # furniture
    20: [0.0, 0.0, 0.0]        # background/unknown
}


def get_scannet_config(benchmark: str = "scannet20") -> Dict:
    """
    Get configuration for ScanNet benchmark.
    
    Args:
        benchmark: "scannet20" or "scannet200"
        
    Returns:
        Config dict compatible with eval_info.yaml format
    """
    if benchmark == "scannet20":
        valid_ids = SCANNET20_VALID_IDS
        labels = SCANNET20_LABELS
        map_to_reduced = {vid: i for i, vid in enumerate(valid_ids)}
        map_to_reduced[-1] = 20  # Unknown -> background
        background_ids = [0, 1]  # wall, floor
        
        return {
            "dataset": "scannet20",
            "num_classes": 21,
            "valid_class_ids": valid_ids,
            "class_names_reduced": labels + ["background"],
            "map_to_reduced": map_to_reduced,
            "background_reduced_ids": background_ids,
            "ignore": [-1],
            "color_map": SCANNET20_COLORS
        }
    
    elif benchmark == "scannet200":
        valid_ids = SCANNET200_VALID_IDS
        labels = SCANNET200_LABELS
        map_to_reduced = {vid: i for i, vid in enumerate(valid_ids)}
        map_to_reduced[-1] = 200  # Unknown -> background
        background_ids = [0, 2, 35]  # wall, floor, ceiling
        
        return {
            "dataset": "scannet200",
            "num_classes": 201,
            "valid_class_ids": valid_ids,
            "class_names_reduced": labels + ["background"],
            "map_to_reduced": map_to_reduced,
            "background_reduced_ids": background_ids,
            "ignore": [-1]
        }
    
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Use 'scannet20' or 'scannet200'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract ScanNet .sens files")
    parser.add_argument("scene_path", help="Path to scene directory containing .sens file")
    parser.add_argument("--output", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--max-frames", "-n", type=int, default=-1, help="Max frames to extract")
    parser.add_argument("--skip", "-s", type=int, default=1, help="Frame skip interval")
    
    args = parser.parse_args()
    
    extract_scannet_scene(
        args.scene_path,
        output_path=args.output,
        max_frames=args.max_frames,
        frame_skip=args.skip
    )
