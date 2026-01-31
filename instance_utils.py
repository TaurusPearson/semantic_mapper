import open3d as o3d
import torch
import numpy as np

def same_instance(instance1, instance2, points_centroid1, points_centroid2, th_centroid, th_cossim, th_points, allow_geometric_only=True):
    """
    Check if two instances should be fused based on geometric and semantic criteria.
    
    Args:
        instance1, instance2: Instance3D objects
        points_centroid1, points_centroid2: Tuple of (points, centroid) for each instance
        th_centroid: Maximum centroid distance threshold (meters)
        th_cossim: Minimum cosine similarity threshold for CLIP features
        th_points: Distance threshold for point proximity check (meters)
        allow_geometric_only: If True, allows fusion based on strong geometric overlap 
                              even without CLIP features (OVO-style). Default True.
    
    Returns:
        bool: True if instances should be fused
    """
    # CRITICAL: Never fuse objects with different SAM3 semantic classes
    # This prevents wall+panel, chair+floor, etc. from being incorrectly merged
    if hasattr(instance1, 'semantic_class_idx') and hasattr(instance2, 'semantic_class_idx'):
        idx1 = instance1.semantic_class_idx
        idx2 = instance2.semantic_class_idx
        # If both have valid (assigned) classes and they differ -> don't fuse
        if idx1 >= 0 and idx2 >= 0 and idx1 != idx2:
            return False
    
    points1, centroid1 = points_centroid1
    points2, centroid2 = points_centroid2
    
    # 1. Check centroid distance (must always pass)
    distance = ((centroid1 - centroid2) ** 2).sum().sqrt()
    if distance > th_centroid:
        return False
    
    # 2. Compute point cloud proximity (always needed)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1.cpu().numpy())
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2.cpu().numpy())
    dists = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    p_dist = (dists < th_points).astype(float).mean()
    
    # 3. Check CLIP similarity if features are available
    has_clip = instance1.clip_feature is not None and instance2.clip_feature is not None
    cos_sim = 0.0
    
    if has_clip:
        feat1 = instance1.clip_feature[0] if instance1.clip_feature.dim() > 1 else instance1.clip_feature
        feat2 = instance2.clip_feature[0] if instance2.clip_feature.dim() > 1 else instance2.clip_feature
        cos_sim = torch.nn.functional.cosine_similarity(feat1, feat2, dim=0).item()
        
        # Strong semantic match + decent geometry -> fuse
        if cos_sim >= th_cossim and p_dist > 0.3:
            return True
        # Very strong semantic match with weak geometry threshold
        if cos_sim > 0.9 and p_dist > 0.2:
            return True
        # CLIP says different objects -> don't fuse
        if cos_sim < th_cossim:
            return False
    
    # 4. Allow geometric-only fusion with strict thresholds (OVO-style)
    # This enables structural merging when CLIP features are pending/unavailable
    if allow_geometric_only and not has_clip:
        # Very strong geometric evidence: >70% points overlap + close centroids
        if p_dist > 0.7 and distance < th_centroid * 0.5:
            return True
        # Don't fuse without CLIP if geometry is weak
        return False
    
    # 5. Default: require >50% point proximity
    return p_dist > 0.5

def fuse_instances(instance1, instance2, map_data):
    points_3d, points_ids, points_ins_ids = map_data
    
    instance1.add_points_ids(instance2.points_ids)
    for kf in instance2.kfs_ids:
        instance1.add_keyframes(kf)
    for (area, kf_id) in instance2.top_kf:
        instance1.add_top_kf(kf_id, area)
    
    # Preserve semantic_class_idx: prefer valid class over unassigned
    if hasattr(instance1, 'semantic_class_idx') and hasattr(instance2, 'semantic_class_idx'):
        if instance1.semantic_class_idx < 0 and instance2.semantic_class_idx >= 0:
            # instance1 has no class, instance2 does -> take instance2's class
            instance1.semantic_class_idx = instance2.semantic_class_idx
        # If both have valid classes, keep instance1's (larger object typically)
        # If neither has valid class, stays -1
    
    points_ins_ids[points_ins_ids == instance2.id] = instance1.id
    return instance1, points_ins_ids