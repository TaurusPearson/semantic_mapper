from typing import Any, Dict, List
import numpy as np
import torch
import heapq
import torch.nn.functional as F
from collections import deque, defaultdict 

def torch_semantic_clustering(features: torch.Tensor, eps: float = 0.2, min_samples: int = 3):
    """
    A GPU-native 'DBSCAN-like' clustering for cosine similarity.
    1. Computes N x N Cosine Similarity Matrix.
    2. Checks which rows have enough neighbors > threshold.
    3. Returns the centroid of the largest valid cluster.
    """
    # features shape: [N, Dim]
    N = features.shape[0]
    
    # 1. Compute Cosine Similarity Matrix (features are already normalized!)
    # Result: [N, N] matrix where [i,j] is sim score
    sim_matrix = features @ features.T 
    
    # 2. Convert eps (distance) to cosine threshold
    # Distance = 1 - Cosine. So Cosine > 1 - eps
    threshold = 1.0 - eps
    
    # 3. Find Adjacency (Who is close to whom?)
    # neighbors[i, j] = True if i and j are close
    neighbors = sim_matrix > threshold
    
    # 4. Count neighbors for each point (Degree)
    # counts[i] = How many friends does point i have?
    counts = neighbors.sum(dim=1)
    
    # 5. Filter Noise
    # Keep only points that have > min_samples neighbors
    core_mask = counts >= min_samples
    
    if core_mask.sum() == 0:
        # Fallback: If everything is noise, return Mean of everything
        return features.mean(dim=0)
    
    # 6. Find the Dense Cluster
    # We take the point with the MOST neighbors as the "Center"
    # and average all its neighbors.
    best_idx = torch.argmax(counts)
    
    # Get the cluster membership of the 'best' core point
    cluster_mask = neighbors[best_idx]
    
    # Return the average of the cluster
    return features[cluster_mask].mean(dim=0)

class Instance3D:
    """
    3D instance class.

    Args:
        id (int): The unique identifier for the 3D instance.
        kf_id (int, optional): Keyframe ID where the instance has been observed. Defaults to None.
        points_ids (List[int], optional): A list of point IDs associated with the object. Defaults to None.
        mask_area (int, optional): The area of the first mask associated with the object. Defaults to 0.
        n_top_kf (int, optional): The number of top keyframes to keep. Defaults to -1.

    Attributes:
        id (int): The unique identifier for the object.
        clip_feature (None): Placeholder for clip feature.
        clip_feature_kf (None): Placeholder for clip feature keyframe.
        kfs_ids (list): A list of keyframe IDs associated with the object.
        points_ids (list): A list of point IDs associated with the object.
        to_update (bool): Flag indicating if the object descriptor needs to be updated.
        n_top_kf (int): The number of top keyframes to keep. If 0 all keyframes are used.
        top_kf (list): A Heap of top keyframes ordered by their mask area.
        semantic_class_idx (int): The direct semantic class from SAM3 text prompts.
    """
    n_top_kf: int = 0

    def __init__(self, id: int, kf_id: int | None = None, points_ids: List[int] = None, mask_area: int = 0, semantic_class_idx: int = -1):
        self.id = id
        self.clip_feature = None
        self.clip_feature_kf = None
        
        # --- ADDED: Feature History for Clustering ---
        self.feature_history = [] 
        # ---------------------------------------------
        
        # --- ADDED: SAM3 Semantic Class Voting (majority vote across frames) ---
        self.semantic_class_votes = defaultdict(int)  # {class_idx: vote_count}
        self.semantic_class_idx = semantic_class_idx  # Current best class (-1 = unassigned)
        if semantic_class_idx >= 0:
            self.semantic_class_votes[semantic_class_idx] = 1
        # ---------------------------------------------------------------------
        
        # --- ADDED: Label Voting for Stability (OVO-style) ---
        self.label_votes = defaultdict(float)  # {label: cumulative_score}
        self.vote_count = 0
        # -----------------------------------------------------
        
        # --- ADDED: SAM Confidence Tracking ---
        self.sam_confidence = 1.0  # Combined predicted_iou * stability_score
        self.sam_confidence_history = []  # Track per-observation confidence
        # --------------------------------------
        
        self.kfs_ids = []
        self.points_ids = []
        self.top_kf = []
        self.to_update = False
        if kf_id is not None:
            self.update(points_ids, kf_id, mask_area)

        self.reported = False  # Has this object's birth certificate been printed?
        self.birth_stats = {
            "kf_id": kf_id,
            "mask_area": mask_area,
            "n_points": len(points_ids) if points_ids else 0
        }
        
    def update(self, points_ids: List[int], kf_id: int, area: int) -> None:
        """ Add repeated
        Args:
            - points_id (List[int]): ids of points matched with the object.
            - kf_id (int): id of keyframe where the object has been observed.  
            - area (int): area of the segmentation map on current keyframe
        Return:
            - True if current KeyFrame is in the top_k view, False otherwise
        """
        self.add_keyframes(kf_id)
        self.add_points_ids(points_ids)
        self.add_top_kf(kf_id, area)

    def add_points_ids(self, points_ids: List[int]) -> None:
        """Add points_ids to list of points matched with the object. 
        Args:
            - points_id (List[int]): ids of points matched with the object.
        """
        self.points_ids.extend(points_ids)

    def add_keyframes(self, kf_id: int) -> None:
        """If frame  no already in list, add to list of keyframes where the object has been observed.
        Args:
            - keyframe_id (int): id of keyframe where the object has been observed. 
        """
        if kf_id not in self.kfs_ids:  
            self.kfs_ids.append(kf_id)
            # Mark for update whenever we see a new view
            self.to_update = True 

    def add_top_kf(self, kf_id: int, area: int)-> None:
        """ If self.n_top_kf <=0, the self.to_update is set to True but self.top_kf remains empty. Otherwise, if the KF is already on the list of best, update area value. Else, if the area is one of the N biggest, add to list of top keyframes, in both cases self.to_update is set to True.
        Args:
            - keyframe_id (int): id of keyframe where the object has been observed.  
            - area (int): area of the segmentation map on current keyframe
        Return:
            - True if current KeyFrame is in the top_k view, False otherwise
        """        
        idx = self.idx_in_top_kf(kf_id)
        if idx > -1 :
            if area > self.top_kf[idx][0]:
                self.top_kf[idx] = (area, kf_id)
                heapq.heapify(self.top_kf)
                self.to_update=True
        else:
            self._add_top_kf(kf_id, area)
    
    def _add_top_kf(self, kf_id: int, area: int) -> None:
        """If the area is one of the N biggest, add to list of top keyframes
        Args:
            - keyframe_id (int): id of keyframe where the object has been observed. 
            - area (int): area of the segmentation map of the object if keyframe_id 
        """
        if len(self.top_kf) < self.n_top_kf:
            heapq.heappush(self.top_kf,(area, kf_id))
            self.to_update=True
        else:
            removed = heapq.heappushpop(self.top_kf,(area, kf_id))
            if (self.n_top_kf <= 0) or (removed[1] != kf_id):
                self.to_update = True
       
    def idx_in_top_kf(self, kf_id: int) -> int:
        """ If kf_id is in self.top_kf, returns the index. Otherwise return -1.
        Args:
            - kf_id (int): id of keyframe to search.
        Return type:
            - (int)
        """
        for idx, (_, id) in enumerate(self.top_kf):
            if id == kf_id:
                return idx
        return -1        
    
    def is_top_kf(self, kf_id: int) -> bool:
        """ If kf_id is in self.top_kf, returns True. Otherwise False.
        Args:
            - kf_id (int): id of keyframe to search.
        Return type:
            - (bool)
        """
        return self.idx_in_top_kf(kf_id) > -1    

    def add_semantic_class_vote(self, class_idx: int) -> None:
        """Add a vote for a semantic class (SAM3 text-prompted detection).
        Updates the majority class if voting changes the winner.
        
        Args:
            class_idx: The semantic class index detected by SAM3 for this observation.
        """
        if class_idx < 0:
            return
        self.semantic_class_votes[class_idx] += 1
        # Update semantic_class_idx to majority winner
        self.semantic_class_idx = max(self.semantic_class_votes.items(), key=lambda x: x[1])[0]
    
    def get_semantic_class_confidence(self) -> float:
        """Get confidence ratio for the majority semantic class.
        
        Returns:
            float: Ratio of votes for majority class vs total votes (0-1).
        """
        total_votes = sum(self.semantic_class_votes.values())
        if total_votes == 0:
            return 0.0
        return self.semantic_class_votes.get(self.semantic_class_idx, 0) / total_votes

    def update_label_vote(self, label: str, confidence: float = 1.0) -> None:
        """Update label voting for stable classification (OVO-style).
        Args:
            - label (str): The predicted label for this instance.
            - confidence (float): Confidence score for the vote (default 1.0).
        """
        self.label_votes[label] += confidence
        self.vote_count += 1
    
    def get_stable_label(self) -> str:
        """Get the most voted label across all observations.
        Returns:
            - str: The label with highest cumulative votes, or 'Unknown' if no votes.
        """
        if not self.label_votes:
            return "Unknown"
        return max(self.label_votes.items(), key=lambda x: x[1])[0]
    
    def get_label_confidence(self) -> float:
        """Get confidence ratio for the stable label.
        Returns:
            - float: Ratio of votes for top label vs total votes (0-1).
        """
        if self.vote_count == 0:
            return 0.0
        top_label = self.get_stable_label()
        return self.label_votes[top_label] / sum(self.label_votes.values())
    
    def update_sam_confidence(self, predicted_iou: float, stability_score: float) -> None:
        """Update SAM mask confidence for this instance.
        
        Args:
            predicted_iou: SAM's predicted IoU for the mask (0-1)
            stability_score: SAM's stability score for the mask (0-1)
        """
        combined = predicted_iou * stability_score
        self.sam_confidence_history.append(combined)
        # Use running average for stability
        self.sam_confidence = sum(self.sam_confidence_history) / len(self.sam_confidence_history)
    
    def get_combined_confidence(self, clip_confidence: float) -> float:
        """Get combined SAM + CLIP confidence for weighted voting.
        
        Args:
            clip_confidence: CLIP/SigLIP softmax probability (0-1)
        
        Returns:
            Combined confidence score for label voting
        """
        # Weight CLIP confidence by SAM mask quality
        # Poor SAM masks should contribute less to semantic classification
        return clip_confidence * self.sam_confidence
    
    @staticmethod
    def compute_visibility_score(mask_area: int, img_area: int, depth_mean: float = None, depth_std: float = None) -> float:
        """OVO-style visibility scoring for top-K view selection.
        
        Combines mask area with optional depth-based quality metrics.
        
        Args:
            - mask_area: Number of pixels in the mask
            - img_area: Total image area (H * W)
            - depth_mean: Mean depth of masked region (optional)
            - depth_std: Std deviation of depth in masked region (optional)
        
        Returns:
            - float: Visibility score (higher = better view)
        """
        # Normalize area (0-1)
        area_score = mask_area / max(img_area, 1)
        
        if depth_mean is not None and depth_std is not None:
            # Penalize very close or very far objects (optimal ~2m)
            optimal_depth = 2.0  # meters
            depth_score = np.exp(-((depth_mean - optimal_depth) / 1.5) ** 2)
            
            # Penalize high depth variance (indicates occlusion/noise)
            variance_penalty = 1.0 / (1.0 + depth_std)
            
            return area_score * depth_score * variance_penalty
        
        # Fallback to area-only scoring (backwards compatible)
        return area_score    

    def update_clip(self, keyframes_clips: Dict[int, Dict[int, torch.Tensor]], force_update: bool = False) -> None:
        if self.to_update or force_update:
            # 1. Collect features from ALL keyframes (preserve full viewpoint diversity)
            all_features = []
            all_kf_ids = []
            for kf in self.kfs_ids:
                kf_clips = keyframes_clips.get(kf)
                if kf_clips is not None and self.id in kf_clips:
                    feat = kf_clips[self.id]
                    feat = feat / (feat.norm() + 1e-6)
                    all_features.append(feat)
                    all_kf_ids.append(kf)
            
            if not all_features:
                self.to_update = False
                return

            # 2. Decoupled top-K selection: only use top-K features for DBSCAN
            # CLIP was extracted from ALL views, but we cluster on the best subset
            if self.n_top_kf > 0 and self.top_kf and len(all_features) > self.n_top_kf:
                top_kf_ids = {kf for _, kf in heapq.nlargest(self.n_top_kf, self.top_kf)}
                selected = [f for f, kf in zip(all_features, all_kf_ids) if kf in top_kf_ids]
                if len(selected) >= 3:
                    features_stack = torch.stack(selected)
                else:
                    features_stack = torch.stack(all_features)
            else:
                features_stack = torch.stack(all_features)

            # 3. Run GPU Clustering
            if features_stack.shape[0] < 3:
                self.clip_feature = features_stack.mean(dim=0)
            else:
                self.clip_feature = torch_semantic_clustering(features_stack, eps=0.20, min_samples=3)

            # Re-normalize final result
            self.clip_feature = self.clip_feature / (self.clip_feature.norm() + 1e-6)
            
            self.to_update = False

    def export(self, debug_info: bool = False) -> Dict[str, Any]:
        """Export object properties as a dictionary.
        Args:
            debug_info (bool): If True, includes additional debug information (self.kfs_ids, self.points_ids, self.top_kf) in the dictionary.
        Returns:
            dict: A dictionary containing the current state of the Instance.
        """
        obj_dict = {
            f"ins3d_{self.id}_clip_feature": self.clip_feature,
            f"ins3d_{self.id}_clip_feature_kf": self.clip_feature_kf,
        }

        if debug_info:
            obj_dict.update({
            f"ins3d_{self.id}_keyframes_ids":  np.array(self.kfs_ids),
            f"ins3d_{self.id}_points_ids":  np.array(self.points_ids),
            f"ins3d_{self.id}_top_kfs":  np.array(self.top_kf),
            })

        return obj_dict
    
    def restore(self, obj_dict: Dict[str, Any], debug_info: bool) -> None:
        """Restore object properties from a dictionary.
        Args:
            obj_dict (Dict[str, Any]): A dictionary containing the current state of the Instance.
            debug_info (bool): If True, expected additional debug information in the dictionary.
        """
        self.clip_feature = obj_dict[f"ins3d_{self.id}_clip_feature"]
        self.clip_feature_kf = obj_dict.get(f"ins3d_{self.id}_clip_feature_kf", None)
        self.to_update=self.clip_feature is None
        if debug_info:
            self.kfs_ids = obj_dict[f"ins3d_{self.id}_keyframes_ids"].tolist()
            self.points_ids = obj_dict[f"ins3d_{self.id}_points_ids"].tolist()
            if obj_dict.get(f"ins3d_{self.id}_top_kfs", None) is not None:
                self.top_kf=[(area,kf_id) for area, kf_id in obj_dict[f"ins3d_{self.id}_top_kfs"]]

    def old_restore(self, obj_dict: Dict[str, Any], debug_info: bool) -> None:
        """Restore object properties from a dictionary.
        Args:
            obj_dict (Dict[str, Any]): A dictionary containing the current state of the Instance.
            debug_info (bool): If True, expected additional debug information in the dictionary.
        """
        self.clip_feature = torch.tensor(obj_dict[f"default_{self.id}_clip_feature"])
        self.clip_feature_kf = obj_dict.get(f"default_{self.id}_clip_feature_kf", None)
        if debug_info:
            self.kfs_ids = obj_dict[f"default_{self.id}_keyframes_ids"].tolist()
            self.points_ids = obj_dict[f"default_{self.id}_points_ids"].tolist()
            if obj_dict.get(f"default_{self.id}_top_kfs", None) is not None:
                self.top_kf=[(area,kf_id) for area, kf_id in obj_dict[f"default_{self.id}_top_kfs"]]

    def purge_points_ids(self, purge_ids: List[int]) -> None:
        """TODO: We need to define an heurisitc to purge ids that were optimized and do not fall inside the 3D instance. -> As long as we use map's point_ins_ids to classify, and not the instance saved points_ids, it doesn't matter if we don't prune.
        """
        points_ids = self.points_ids.copy()
        for point in points_ids:
            if point in purge_ids:
                self.points_ids.remove(point)