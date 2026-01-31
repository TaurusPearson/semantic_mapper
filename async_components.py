"""
async_components.py

OVO Spec I & III.3: Async PTAM Architecture & Async CLIP Queue

Implements:
- AsyncMapper: Separate mapping thread (back-end) that processes keyframes asynchronously
- AsyncCLIPProcessor: Background process for heavy CLIP feature extraction (uses multiprocessing to bypass GIL)
"""

import threading
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue, Event
import torch
import numpy as np
from queue import Queue, Empty
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import pickle


@dataclass
class KeyframeData:
    """Data packet sent from Tracker to Mapper"""
    frame_id: int
    image: np.ndarray
    depth: np.ndarray
    pose: np.ndarray
    c2w: torch.Tensor
    binary_maps: Optional[torch.Tensor] = None


@dataclass
class CLIPWorkItem:
    """Work item for async CLIP processing"""
    kf_id: int
    image: np.ndarray
    binary_maps: torch.Tensor
    matched_ins_ids: List[int]


class AsyncMapper:
    """
    OVO Spec I: PTAM Back-end Mapper Thread
    
    Runs mapping operations asynchronously. The Tracker (front-end) submits
    keyframes via `submit_keyframe()`, and this thread processes them in
    the background without blocking the tracking loop.
    """
    
    def __init__(self, slam_mapper, semantic_mapper, device: str = "cuda"):
        self.slam_mapper = slam_mapper
        self.semantic_mapper = semantic_mapper
        self.device = device
        
        # Keyframe queue: Tracker -> Mapper
        self.keyframe_queue: Queue[Optional[KeyframeData]] = Queue(maxsize=10)
        
        # Results that need to be synced back
        self.results_lock = threading.Lock()
        self.pending_updates: List[torch.Tensor] = []
        
        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Stats
        self.frames_processed = 0
        self.avg_process_time = 0.0
        
    def start(self):
        """Start the async mapping thread"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._mapping_loop, daemon=True, name="AsyncMapper")
        self._thread.start()
        print("[AsyncMapper] Started background mapping thread")
        
    def stop(self):
        """Stop the mapping thread gracefully"""
        if not self._running:
            return
        self._running = False
        self.keyframe_queue.put(None)  # Poison pill
        if self._thread:
            self._thread.join(timeout=5.0)
        print(f"[AsyncMapper] Stopped. Processed {self.frames_processed} frames")
        
    def submit_keyframe(self, keyframe: KeyframeData, block: bool = False):
        """
        Submit a keyframe for async processing.
        
        Args:
            keyframe: KeyframeData packet from tracker
            block: If True, wait for queue space. If False, drop if full.
        """
        try:
            self.keyframe_queue.put(keyframe, block=block, timeout=0.1)
        except:
            # Queue full - drop frame (tracker is faster than mapper)
            pass
            
    def get_pending_updates(self) -> List[torch.Tensor]:
        """Retrieve processed updates (called by main thread)"""
        with self.results_lock:
            updates = self.pending_updates.copy()
            self.pending_updates.clear()
        return updates
    
    def is_idle(self) -> bool:
        """Check if mapper has finished all pending work"""
        return self.keyframe_queue.empty()
    
    def wait_until_idle(self, timeout: float = 30.0):
        """Block until all pending keyframes are processed"""
        start = time.time()
        while not self.is_idle() and (time.time() - start) < timeout:
            time.sleep(0.05)
            
    def _mapping_loop(self):
        """Main mapping thread loop"""
        while self._running:
            try:
                # Wait for keyframe with timeout
                keyframe = self.keyframe_queue.get(timeout=0.5)
                
                if keyframe is None:
                    break  # Poison pill received
                    
                t0 = time.time()
                self._process_keyframe(keyframe)
                dt = time.time() - t0
                
                # Update stats
                self.frames_processed += 1
                self.avg_process_time = (
                    self.avg_process_time * 0.9 + dt * 0.1
                )
                
            except Empty:
                continue  # Timeout, check if still running
            except Exception as e:
                print(f"[AsyncMapper] Error processing keyframe: {e}")
                import traceback
                traceback.print_exc()
                
    def _process_keyframe(self, kf: KeyframeData):
        """Process a single keyframe (runs in background thread)"""
        frame_data = [kf.frame_id, kf.image, kf.depth, kf.pose]
        
        # 1. Update SLAM map
        self.slam_mapper.map(frame_data, kf.c2w)
        
        # 2. If we have masks, run semantic tracking
        if kf.binary_maps is not None and kf.binary_maps.numel() > 0:
            map_data = self.slam_mapper.get_map()
            
            updated_ids = self.semantic_mapper.track(
                frame_data, 
                kf.c2w, 
                map_data, 
                kf.binary_maps
            )
            
            if updated_ids is not None:
                with self.results_lock:
                    self.pending_updates.append(updated_ids)


class AsyncCLIPProcessor:
    """
    OVO Spec III.3: Asynchronous CLIP Feature Queue
    
    Heavy CLIP feature extraction runs in a background thread.
    The semantic mapper pushes (keyframe, masks) tuples to this queue,
    and features are computed when GPU resources are available.
    """
    
    def __init__(self, clip_generator, device: str = "cuda", max_queue_size: int = 20):
        self.clip_generator = clip_generator
        self.device = device
        
        # Work queue: (kf_id, image, binary_maps, matched_ids)
        self.work_queue: Queue[Optional[CLIPWorkItem]] = Queue(maxsize=max_queue_size)
        
        # Results storage (thread-safe)
        self.results_lock = threading.Lock()
        self.results: Dict[int, Tuple[torch.Tensor, List[int]]] = {}
        
        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Stats
        self.items_processed = 0
        self.avg_process_time = 0.0
        
    def start(self):
        """Start the async CLIP processing thread"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="AsyncCLIP")
        self._thread.start()
        print("[AsyncCLIP] Started background CLIP processing thread")
        
    def stop(self):
        """Stop the CLIP worker gracefully"""
        if not self._running:
            return
        self._running = False
        self.work_queue.put(None)  # Poison pill
        if self._thread:
            self._thread.join(timeout=10.0)
        print(f"[AsyncCLIP] Stopped. Processed {self.items_processed} items")
        
    def submit(self, kf_id: int, image: np.ndarray, binary_maps: torch.Tensor, 
               matched_ins_ids: List[int], block: bool = False) -> bool:
        """
        Submit work for async CLIP processing.
        
        Args:
            kf_id: Keyframe ID
            image: RGB image (numpy array)
            binary_maps: Binary mask tensor [N, H, W]
            matched_ins_ids: List of instance IDs matched to each mask
            block: If True, wait for queue space
            
        Returns:
            True if submitted, False if queue was full
        """
        work = CLIPWorkItem(
            kf_id=kf_id,
            image=image,
            binary_maps=binary_maps.cpu() if binary_maps.is_cuda else binary_maps,
            matched_ins_ids=matched_ins_ids.copy() if isinstance(matched_ins_ids, list) else list(matched_ins_ids)
        )
        
        try:
            self.work_queue.put(work, block=block, timeout=0.1)
            return True
        except:
            return False  # Queue full
            
    def get_results(self, kf_ids: Optional[List[int]] = None) -> Dict[int, Tuple[torch.Tensor, List[int]]]:
        """
        Retrieve computed CLIP features.
        
        Args:
            kf_ids: Specific keyframe IDs to retrieve, or None for all
            
        Returns:
            Dict mapping kf_id -> (clip_embeddings, matched_ins_ids)
        """
        with self.results_lock:
            if kf_ids is None:
                out = self.results.copy()
                self.results.clear()
            else:
                out = {}
                for kf_id in kf_ids:
                    if kf_id in self.results:
                        out[kf_id] = self.results.pop(kf_id)
        return out
    
    def has_pending_work(self) -> bool:
        """Check if there's pending work"""
        return not self.work_queue.empty()
    
    def pending_count(self) -> int:
        """Get approximate count of pending items"""
        return self.work_queue.qsize()
    
    def results_ready(self) -> int:
        """Get count of ready results"""
        with self.results_lock:
            return len(self.results)
            
    def wait_for_completion(self, timeout: float = 60.0):
        """Block until all pending work is done"""
        start = time.time()
        while self.has_pending_work() and (time.time() - start) < timeout:
            time.sleep(0.1)
            
    def _worker_loop(self):
        """Main CLIP worker loop"""
        while self._running:
            try:
                work = self.work_queue.get(timeout=0.5)
                
                if work is None:
                    break  # Poison pill
                    
                t0 = time.time()
                self._process_item(work)
                dt = time.time() - t0
                
                self.items_processed += 1
                self.avg_process_time = self.avg_process_time * 0.9 + dt * 0.1
                
            except Empty:
                continue
            except Exception as e:
                print(f"[AsyncCLIP] Error processing item: {e}")
                import traceback
                traceback.print_exc()
                
    @torch.no_grad()
    def _process_item(self, work: CLIPWorkItem):
        """Process a single CLIP work item"""
        # Move tensors to GPU
        binary_maps = work.binary_maps.to(self.device)
        
        if binary_maps.shape[0] == 0:
            return
            
        # Convert image to tensor format expected by CLIP generator
        if isinstance(work.image, np.ndarray):
            image_tensor = torch.from_numpy(work.image.transpose((2, 0, 1))).to(self.device).float()
        else:
            image_tensor = work.image.to(self.device)
            
        # Extract CLIP features (the heavy computation)
        clip_embeds = self.clip_generator.extract_clip(image_tensor, binary_maps)
        
        # Store results (move back to CPU for storage efficiency)
        with self.results_lock:
            self.results[work.kf_id] = (clip_embeds.cpu(), work.matched_ins_ids)


def _clip_worker_process(work_queue: MPQueue, result_queue: MPQueue, stop_event, 
                          clip_config: dict, device: str):
    """
    Standalone CLIP worker process (runs in separate process to bypass GIL).
    
    Loads its own CLIP model and processes work items from the queue.
    """
    import torch
    from segment_utils import CLIPGenerator
    from unified_semantic_backbone import load_semantic_backbone
    
    print(f"[CLIP Worker] Starting on device {device}...")
    
    # Load CLIP model in this process
    try:
        backbone = load_semantic_backbone(clip_config.get('sem_name', 'siglip'))
        clip_gen = CLIPGenerator(clip_config.get('clip', {}), backbone, device=device)
        print("[CLIP Worker] Model loaded successfully")
    except Exception as e:
        print(f"[CLIP Worker] Failed to load model: {e}")
        return
    
    items_processed = 0
    total_time = 0.0
    
    while not stop_event.is_set():
        try:
            # Get work with timeout
            work = work_queue.get(timeout=0.5)
            
            if work is None:
                break  # Poison pill
                
            kf_id, image_np, binary_maps_np, matched_ins_ids = work
            
            t0 = time.time()
            
            # Convert to tensors
            binary_maps = torch.from_numpy(binary_maps_np).to(device)
            
            if binary_maps.shape[0] == 0:
                continue
                
            image_tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).to(device).float()
            
            # Extract CLIP features
            with torch.no_grad():
                clip_embeds = clip_gen.extract_clip(image_tensor, binary_maps)
            
            # Send results back (as numpy for serialization)
            result_queue.put((kf_id, clip_embeds.cpu().numpy(), matched_ins_ids))
            
            dt = time.time() - t0
            items_processed += 1
            total_time += dt
            
        except Empty:
            continue
        except Exception as e:
            print(f"[CLIP Worker] Error: {e}")
            import traceback
            traceback.print_exc()
    
    avg_time = (total_time / items_processed * 1000) if items_processed > 0 else 0
    print(f"[CLIP Worker] Stopped. Processed {items_processed} items, avg {avg_time:.1f}ms/item")


class MultiprocessingCLIPProcessor:
    """
    OVO Spec III.3: Multiprocessing CLIP Feature Queue
    
    Uses a separate process (not thread) to bypass Python's GIL.
    This enables true parallel execution of CLIP feature extraction.
    """
    
    def __init__(self, clip_config: dict, device: str = "cuda", max_queue_size: int = 20):
        self.clip_config = clip_config
        self.device = device
        
        # Multiprocessing queues
        self.work_queue = MPQueue(maxsize=max_queue_size)
        self.result_queue = MPQueue(maxsize=max_queue_size)
        
        # Control
        self.stop_event = Event()
        self._process: Optional[Process] = None
        
        # Local results cache (for compatibility)
        self.results: Dict[int, Tuple[torch.Tensor, List[int]]] = {}
        
        # Stats
        self.items_processed = 0
        self.avg_process_time = 0.0
        
    def start(self):
        """Start the CLIP worker process"""
        if self._process is not None and self._process.is_alive():
            return
            
        self.stop_event.clear()
        self._process = Process(
            target=_clip_worker_process,
            args=(self.work_queue, self.result_queue, self.stop_event, 
                  self.clip_config, self.device),
            daemon=True
        )
        self._process.start()
        print("[MultiprocessingCLIP] Started worker process")
        
    def stop(self):
        """Stop the worker process gracefully"""
        self.stop_event.set()
        
        # Send poison pill
        try:
            self.work_queue.put(None, timeout=1.0)
        except:
            pass
            
        if self._process:
            self._process.join(timeout=10.0)
            if self._process.is_alive():
                self._process.terminate()
        print(f"[MultiprocessingCLIP] Stopped")
        
    def submit(self, kf_id: int, image: np.ndarray, binary_maps: torch.Tensor,
               matched_ins_ids: List[int], block: bool = False) -> bool:
        """Submit work for async CLIP processing"""
        # Convert to numpy for serialization
        binary_maps_np = binary_maps.cpu().numpy() if binary_maps.is_cuda else binary_maps.numpy()
        
        work = (kf_id, image, binary_maps_np, list(matched_ins_ids))
        
        try:
            self.work_queue.put(work, block=block, timeout=0.1)
            return True
        except:
            return False
            
    def get_results(self, kf_ids: Optional[List[int]] = None) -> Dict[int, Tuple[torch.Tensor, List[int]]]:
        """Retrieve computed CLIP features"""
        # First, collect any new results from the queue
        self._collect_results()
        
        if kf_ids is None:
            out = self.results.copy()
            self.results.clear()
        else:
            out = {}
            for kf_id in kf_ids:
                if kf_id in self.results:
                    out[kf_id] = self.results.pop(kf_id)
        return out
        
    def _collect_results(self):
        """Collect results from worker process"""
        while True:
            try:
                kf_id, clip_embeds_np, matched_ins_ids = self.result_queue.get_nowait()
                clip_embeds = torch.from_numpy(clip_embeds_np)
                self.results[kf_id] = (clip_embeds, matched_ins_ids)
                self.items_processed += 1
            except Empty:
                break
                
    def has_pending_work(self) -> bool:
        return not self.work_queue.empty()
        
    def pending_count(self) -> int:
        return self.work_queue.qsize()
        
    def results_ready(self) -> int:
        self._collect_results()
        return len(self.results)
        
    def wait_for_completion(self, timeout: float = 60.0):
        """Block until all pending work is done"""
        start = time.time()
        while self.has_pending_work() and (time.time() - start) < timeout:
            self._collect_results()
            time.sleep(0.1)
        self._collect_results()


class PTAMController:
    """
    OVO Spec I: Parallel Tracking and Mapping Controller
    
    High-level controller that coordinates:
    - Tracking thread (front-end): Runs at frame rate
    - Mapping thread (back-end): Processes keyframes asynchronously
    - CLIP thread: Extracts semantic features in background
    """
    
    def __init__(self, slam_mapper, semantic_mapper, clip_generator, device: str = "cuda"):
        self.device = device
        
        # Create async components
        self.async_mapper = AsyncMapper(slam_mapper, semantic_mapper, device)
        self.async_clip = AsyncCLIPProcessor(clip_generator, device)
        
        # References
        self.slam_mapper = slam_mapper
        self.semantic_mapper = semantic_mapper
        
        # Keyframe selection criteria
        self.last_kf_id = -1
        self.kf_interval = 10  # Minimum frames between keyframes
        
    def start(self):
        """Start all async threads"""
        self.async_mapper.start()
        self.async_clip.start()
        
    def stop(self):
        """Stop all async threads gracefully"""
        self.async_clip.stop()
        self.async_mapper.stop()
        
    def is_keyframe(self, frame_id: int) -> bool:
        """Determine if current frame should be a keyframe"""
        if frame_id == 0:
            return True
        return (frame_id - self.last_kf_id) >= self.kf_interval
        
    def track_frame(self, frame_id: int, image: np.ndarray, depth: np.ndarray, 
                    pose: np.ndarray, binary_maps: Optional[torch.Tensor] = None):
        """
        Process a frame in the tracking thread (non-blocking).
        
        This is called at frame-rate. If it's a keyframe, work is
        dispatched to the async mapper and CLIP processor.
        """
        c2w = torch.from_numpy(pose).float().to(self.device)
        
        # Always track camera pose
        self.slam_mapper.track_camera([frame_id, image, depth, pose])
        
        # Check if this is a keyframe
        if self.is_keyframe(frame_id):
            self.last_kf_id = frame_id
            
            # Submit to async mapper
            kf_data = KeyframeData(
                frame_id=frame_id,
                image=image,
                depth=depth,
                pose=pose,
                c2w=c2w,
                binary_maps=binary_maps
            )
            self.async_mapper.submit_keyframe(kf_data)
            
        # Sync any pending map updates
        self._sync_updates()
        
    def submit_clip_work(self, kf_id: int, image: np.ndarray, 
                         binary_maps: torch.Tensor, matched_ids: List[int]):
        """Submit CLIP extraction work (called from semantic mapper)"""
        self.async_clip.submit(kf_id, image, binary_maps, matched_ids)
        
    def get_clip_results(self) -> Dict[int, Tuple[torch.Tensor, List[int]]]:
        """Get any completed CLIP results"""
        return self.async_clip.get_results()
        
    def _sync_updates(self):
        """Sync completed mapping work back to main state"""
        updates = self.async_mapper.get_pending_updates()
        for updated_ids in updates:
            self.slam_mapper.update_pcd_obj_ids(updated_ids)
            
    def flush(self, timeout: float = 30.0):
        """Wait for all async work to complete"""
        print("[PTAM] Flushing async queues...")
        self.async_mapper.wait_until_idle(timeout)
        self.async_clip.wait_for_completion(timeout)
        self._sync_updates()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "mapper_frames": self.async_mapper.frames_processed,
            "mapper_avg_time": self.async_mapper.avg_process_time,
            "clip_items": self.async_clip.items_processed,
            "clip_avg_time": self.async_clip.avg_process_time,
            "clip_pending": self.async_clip.pending_count(),
            "clip_ready": self.async_clip.results_ready(),
        }
