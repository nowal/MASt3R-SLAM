"""
Session Keyframes - Single-process adapter for SharedKeyframes

This module provides a SessionKeyframes class that mimics the SharedKeyframes interface
but uses regular Python objects instead of multiprocessing shared memory. This allows
us to use the original FrameTracker without modification while maintaining session isolation.
"""

import logging
import threading
from typing import Optional, List
import torch
import lietorch
from mast3r_slam.frame import Frame
from mast3r_slam.config import config

logger = logging.getLogger("SessionKeyframes")


class SessionKeyframes:
    """
    Single-process version of SharedKeyframes interface
    
    Mimics the SharedKeyframes API but uses regular Python objects instead of
    multiprocessing shared memory. This allows session isolation while maintaining
    compatibility with the original FrameTracker.
    """
    
    def __init__(self, h: int, w: int, buffer: int = 512, dtype=torch.float32, device: str = "cuda"):
        """
        Initialize SessionKeyframes
        
        Args:
            h: Image height
            w: Image width  
            buffer: Maximum number of keyframes to store
            dtype: Tensor data type
            device: Device for tensors
        """
        self.h = h
        self.w = w
        self.buffer = buffer
        self.dtype = dtype
        self.device = device
        
        # Use threading.RLock for thread safety (lighter than multiprocessing)
        self.lock = threading.RLock()
        
        # Store keyframes as a list of Frame objects
        self.keyframes: List[Frame] = []
        self.n_size = 0
        
        # Camera intrinsics (if using calibration)
        self.K = None
        
        logger.info(f"SessionKeyframes initialized: {h}x{w}, buffer={buffer}, device={device}")
    
    def __getitem__(self, idx: int) -> Frame:
        """Get keyframe by index"""
        with self.lock:
            if idx < 0 or idx >= len(self.keyframes):
                raise IndexError(f"Keyframe index {idx} out of range [0, {len(self.keyframes)})")
            return self.keyframes[idx]
    
    def __setitem__(self, idx: int, frame: Frame) -> None:
        """Set keyframe at index"""
        with self.lock:
            # Extend list if necessary
            while len(self.keyframes) <= idx:
                self.keyframes.append(None)
            
            self.keyframes[idx] = frame
            self.n_size = max(idx + 1, self.n_size)
            
            logger.debug(f"Keyframe {idx} set, total keyframes: {self.n_size}")
    
    def __len__(self) -> int:
        """Get number of keyframes"""
        with self.lock:
            return self.n_size
    
    def append(self, frame: Frame) -> None:
        """Add a new keyframe"""
        with self.lock:
            if self.n_size >= self.buffer:
                logger.warning(f"Keyframe buffer full ({self.buffer}), oldest keyframe will be overwritten")
                # For now, just extend the list - could implement circular buffer later
            
            self.keyframes.append(frame)
            self.n_size = len(self.keyframes)
            
            logger.info(f"Keyframe {self.n_size - 1} appended (frame_id: {frame.frame_id}), total: {self.n_size}")
    
    def pop_last(self) -> None:
        """Remove the last keyframe"""
        with self.lock:
            if self.n_size > 0:
                removed_frame = self.keyframes.pop()
                self.n_size = len(self.keyframes)
                logger.info(f"Last keyframe removed (frame_id: {removed_frame.frame_id}), total: {self.n_size}")
            else:
                logger.warning("Attempted to pop from empty keyframes list")
    
    def last_keyframe(self) -> Optional[Frame]:
        """Get the most recent keyframe"""
        with self.lock:
            if self.n_size == 0:
                return None
            return self.keyframes[self.n_size - 1]
    
    def update_T_WCs(self, T_WCs: lietorch.Sim3, idx: torch.Tensor) -> None:
        """
        Update poses for multiple keyframes
        
        Args:
            T_WCs: New poses (lietorch.Sim3 object)
            idx: Indices of keyframes to update
        """
        with self.lock:
            # Log tensor shapes for debugging
            logger.info(f"[TENSOR_DEBUG] update_T_WCs called with T_WCs.data.shape: {T_WCs.data.shape}")
            logger.info(f"[TENSOR_DEBUG] update_T_WCs idx: {idx}")
            
            for i, kf_idx in enumerate(idx):
                kf_idx = int(kf_idx)
                if 0 <= kf_idx < self.n_size:
                    # CRITICAL FIX: Extract pose data correctly to match SharedKeyframes behavior
                    # T_WCs.data has shape [N, 1, 8], so T_WCs.data[i] gives us [1, 8]
                    # NOT T_WCs.data[i:i+1] which would give us [1, 1, 8]
                    pose_data = T_WCs.data[i]  # This gives us [1, 8] from [N, 1, 8]
                    
                    # Log the shape before updating
                    old_shape = self.keyframes[kf_idx].T_WC.data.shape
                    logger.info(f"[TENSOR_DEBUG] Updating keyframe {kf_idx}: old_shape={old_shape}, new_shape={pose_data.shape}")
                    
                    # Create new Sim3 object with the correct shape
                    self.keyframes[kf_idx].T_WC = lietorch.Sim3(pose_data)
                    
                    # Verify the shape after updating
                    new_shape = self.keyframes[kf_idx].T_WC.data.shape
                    if new_shape != torch.Size([1, 8]):
                        logger.error(f"[TENSOR_DEBUG] SHAPE ERROR! Keyframe {kf_idx} pose shape is {new_shape}, expected [1, 8]")
                    else:
                        logger.info(f"[TENSOR_DEBUG] ✓ Keyframe {kf_idx} pose shape correctly maintained: {new_shape}")
                    
                    logger.debug(f"Updated pose for keyframe {kf_idx}")
                else:
                    logger.warning(f"Invalid keyframe index for pose update: {kf_idx}")
    
    def get_dirty_idx(self) -> torch.Tensor:
        """
        Get indices of keyframes that need updates
        
        Note: In the original SharedKeyframes, this tracks which keyframes have been
        modified and need visualization updates. For our implementation, we'll return
        all keyframes as "dirty" since we don't track this state yet.
        """
        with self.lock:
            if self.n_size == 0:
                return torch.tensor([], dtype=torch.long, device=self.device)
            return torch.arange(self.n_size, dtype=torch.long, device=self.device)
    
    def set_intrinsics(self, K: torch.Tensor) -> None:
        """Set camera intrinsics"""
        if not config["use_calib"]:
            logger.warning("Setting intrinsics but use_calib is False")
        
        with self.lock:
            self.K = K.clone()
            logger.info(f"Camera intrinsics set: {K.shape}")
    
    def get_intrinsics(self) -> torch.Tensor:
        """Get camera intrinsics"""
        if not config["use_calib"]:
            raise RuntimeError("Cannot get intrinsics when use_calib is False")
        
        with self.lock:
            if self.K is None:
                raise RuntimeError("Camera intrinsics not set")
            return self.K
    
    def get_stats(self) -> dict:
        """Get statistics about the keyframes"""
        with self.lock:
            stats = {
                "num_keyframes": self.n_size,
                "buffer_size": self.buffer,
                "buffer_usage": f"{self.n_size}/{self.buffer}",
                "device": self.device,
                "has_intrinsics": self.K is not None
            }
            
            if self.n_size > 0:
                last_frame = self.keyframes[-1]
                stats.update({
                    "last_frame_id": last_frame.frame_id,
                    "last_frame_points": last_frame.X_canon.shape[0] if last_frame.X_canon is not None else 0,
                    "last_frame_confidence": float(last_frame.get_average_conf().mean()) if last_frame.C is not None else 0.0
                })
            
            return stats
    
    def clear(self) -> None:
        """Clear all keyframes"""
        with self.lock:
            self.keyframes.clear()
            self.n_size = 0
            logger.info("All keyframes cleared")
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        with self.lock:
            self.clear()
            if self.K is not None:
                del self.K
                self.K = None
            logger.info("SessionKeyframes cleanup completed")
