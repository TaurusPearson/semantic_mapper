import numpy as np
import torch
from typing import Any, Dict, List, Tuple
import geometry_utils
from typing import Any, Dict, List

class SLAMMapper():
    """This class uses GT camera posses to generate a vanilla point-cloud reconstruction by unprojecting depths"""
    def __init__(self, config: dict, cam_intrinsics: torch.Tensor) -> None:
        self.cam_intrinsics = cam_intrinsics
        self.config = config   
        self.device = config.get("device", "cuda")
        self.max_frame_points = config["mapping"].get("max_frame_points", 1e5)
        
        # Performance optimization: limit total point cloud size (OVO-style memory management)
        # Reduced from 500k to 300k and coarser voxel grid to reduce peak VRAM
        self.max_total_points = config["mapping"].get("max_total_points", 300000)  # 300k max (was 500k)
        self.voxel_size = config["mapping"].get("voxel_size", 0.025)  # 2.5cm voxel grid (was 2cm)

        self.match_distance_th = 0.03 # 3 cm
        self.max_id = 0
        self.estimated_c2ws = dict()  
        self.kfs = {}   
        self.map_updated = False

        self.pcd = torch.empty((0,3), device = self.device)
        self.pcd_ids = torch.empty((0,1), device = self.device,dtype=torch.int32 )
        self.pcd_obj_ids = torch.empty((0,1), device = self.device, dtype=torch.int32)
        self.pcd_colors = torch.empty((0,3), device = self.device, dtype = torch.uint8)
        
        k_size = config["mapping"].get("k_pooling", 3)
        if k_size >1:
            pooling = torch.nn.MaxPool2d(kernel_size=k_size, stride=1, padding = int(k_size/2))
            self.pooling = lambda mask: ~(pooling((~mask[None]).float(), )[0].bool())
        else:
            self.pooling = torch.nn.Identity()
        downscale_ratio = config["mapping"].get("downscale_ratio", 2)
        if downscale_ratio ==1:
            self.downscale = lambda x:x
        else:
            self.downscale = lambda x: x[::downscale_ratio, ::downscale_ratio]

    def track_camera(self, frame_data: List[Any]) -> None:
        frame_id = frame_data[0]
        c2w = frame_data[3]
        if np.isinf(c2w).sum()>0 or np.isnan(c2w).sum()>0:
            return
            
        self.estimated_c2ws[frame_id] = torch.from_numpy(c2w).to(self.device)

    def map(self, frame_data: List[Any], c2w: torch.Tensor) -> None:
        # unproject depth
        image, depth, pose = frame_data[1:4]
        # load depth camera intrinsics
        h = image.shape[0]
        w = image.shape[1]
        # create point cloud
        y, x = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing="ij")
        depth  = torch.from_numpy(depth.astype(np.float32)).to(self.device)
        mask = depth > 0
        
        if self.max_id>0:
            # Check if c2w is valid before any operations requiring inversion
            # Error: "diagonal element 2 is zero" means c2w has a zero on diagonal
            diag = torch.diag(c2w[:3, :3])
            if (diag.abs() < 1e-8).any() or torch.linalg.det(c2w[:3, :3]).abs() < 1e-6:
                print(f"[WARN] Skipping frame with singular c2w matrix (diag={diag.cpu().numpy()})")
                return  # Skip this frame
            
            camera_frustum_corners = geometry_utils.compute_camera_frustum_corners(depth, c2w, self.cam_intrinsics)
            frustum_mask = geometry_utils.compute_frustum_point_ids(self.pcd, camera_frustum_corners, device=self.device)
            
            try:
                c2w_inv = torch.linalg.inv(c2w)
                _, matches = geometry_utils.match_3d_points_to_2d_pixels(depth, c2w_inv, self.pcd[frustum_mask], self.cam_intrinsics, self.match_distance_th)
                mask[matches[:,1], matches[:,0]] = False # Do not project depth on points alredy matched
            except torch._C._LinAlgError as e:
                print(f"[WARN] Skipping frame - c2w inversion failed: {e}")
                return
            mask = self.pooling(mask)

        if mask.sum() == 0:
            return
        
        y, x = self.downscale(y), self.downscale(x)
        depth, mask, image = self.downscale(depth), self.downscale(mask), self.downscale(image)

        x = x[mask]
        y = y[mask]
        depth = depth[mask]
        # convert to 3D
        x_3d = (x - self.cam_intrinsics[0, 2]) * depth / self.cam_intrinsics[0, 0]
        y_3d = (y - self.cam_intrinsics[1, 2]) * depth / self.cam_intrinsics[1, 1]
        z_3d = depth
        
        points = torch.hstack((x_3d.reshape(-1, 1), y_3d.reshape(-1, 1), z_3d.reshape(-1, 1), torch.ones((x_3d.shape[0],1), device=self.device)))
        points = torch.einsum("ij,mj->mi",c2w, points)

        # Add new points
        new_points = points[:,:3]
        new_ids = torch.arange(self.max_id, self.max_id+points.shape[0], device=self.device, dtype=torch.int32).unsqueeze(1)
        new_obj_ids = torch.ones((points.shape[0],1), device=self.device, dtype=torch.int32)*-1
        new_colors = torch.from_numpy(image.astype(np.uint8)).to(self.device)[mask].reshape(-1,3)
        
        self.pcd = torch.vstack((self.pcd, new_points))
        self.pcd_ids = torch.vstack((self.pcd_ids, new_ids))
        self.pcd_obj_ids = torch.vstack((self.pcd_obj_ids, new_obj_ids))
        self.pcd_colors = torch.vstack((self.pcd_colors, new_colors))
        self.max_id += points.shape[0]
        
        # Performance: Downsample if point cloud gets too large
        if self.pcd.shape[0] > self.max_total_points:
            self._downsample_pointcloud()

        # NOTE: Do NOT set map_updated=True here. OVO's vanilla mapper never does.
        # update_map() is expensive O(n²) instance fusion — only meant for SLAM loop closures.
        # self.map_updated = True
            
    def _downsample_pointcloud(self):
        """Voxel-based downsampling to limit point cloud size"""
        # Simple voxel grid downsampling
        voxel_coords = (self.pcd / self.voxel_size).long()
        
        # Get unique voxels (keep first point per voxel)
        _, unique_indices = np.unique(
            voxel_coords.cpu().numpy(), axis=0, return_index=True
        )
        unique_indices = torch.from_numpy(unique_indices).to(self.device)
        
        # Keep only unique points
        self.pcd = self.pcd[unique_indices]
        self.pcd_ids = self.pcd_ids[unique_indices]
        self.pcd_obj_ids = self.pcd_obj_ids[unique_indices]
        self.pcd_colors = self.pcd_colors[unique_indices]
        
        print(f"[Mapper] Downsampled to {self.pcd.shape[0]} points")
            
    def get_c2w(self, frame_id: int) -> torch.Tensor:
        c2w = self.estimated_c2ws.get(frame_id, None)
        if c2w is not None and c2w.device != self.device:
            c2w = c2w.to(self.device)
        return c2w

    def cam_to_cpu(self, frame_id: int) -> None:
        c2w = self.estimated_c2ws.get(frame_id, None)
        if c2w is not None:
            self.estimated_c2ws[frame_id] = c2w.cpu()

    def get_map(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a reference to map tensors."""
        return self.pcd, self.pcd_ids, self.pcd_obj_ids.squeeze()
    
    def get_kfs(self) -> Dict[int, Dict[str, Any]]:
        return self.kfs

    def get_map_dict(self) -> Dict[str, Any]:
        return {
            "xyz": self.pcd.clone().detach().cpu(),
            "obj_ids": self.pcd_obj_ids.clone().detach().cpu(),
            "ids": self.pcd_ids.clone().detach().cpu(),
            "max_id": self.max_id,
            "color": self.pcd_colors.clone().detach().cpu()
        }
    
    def set_map_dict(self, map_dict: Dict[str, Any]) -> None:
        self.pcd = map_dict["xyz"].to(self.device)
        self.pcd_obj_ids = map_dict["obj_ids"].to(self.device)
        self.pcd_ids = map_dict["ids"].to(self.device)
        self.max_id = map_dict["max_id"]
        self.pcd_colors = map_dict["color"].to(self.device)

    def get_cam_dict(self) -> dict[str, Any]:
        out_dict = {}
        for key, item in self.estimated_c2ws.items():
            out_dict[key] = item.cpu().numpy()
        return out_dict

    def set_cam_dict(self, cam_dict: dict[str, Any]) -> None:
        self.estimated_c2ws = {}
        for key, item in cam_dict.items():
            self.estimated_c2ws[int(key)] = torch.from_numpy(item).to(self.device)
    
    def update_pcd_obj_ids(self, pcd_objs_ids: torch.Tensor):
        self.pcd_obj_ids = pcd_objs_ids.unsqueeze(-1)

    def get_pcd_colors(self) -> np.ndarray:
        return self.pcd_colors.cpu().numpy()