"""
Frame Saver - Debug utility for saving incoming frames

This module saves incoming frames to disk for visual inspection and debugging.
Helps identify if frames are actually different or if there are issues with
frame processing or frontend transmission.
"""

import os
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger("FrameSaver")


class FrameSaver:
    """
    Utility class for saving frames to disk for debugging
    
    Creates session-specific directories and saves frames with sequential naming
    for easy visual inspection and comparison.
    """
    
    def __init__(self, base_dir: str = "framesChecker"):
        """
        Initialize the frame saver
        
        Args:
            base_dir: Base directory for saving frames
        """
        self.base_dir = Path(base_dir)
        self.session_dirs: Dict[str, Path] = {}
        self.frame_counters: Dict[str, int] = {}
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(exist_ok=True)
        logger.info(f"FrameSaver initialized with base directory: {self.base_dir}")
    
    def setup_session(self, session_id: str) -> Path:
        """
        Set up a new session directory for frame saving
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path to the session directory
        """
        session_dir = self.base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        self.session_dirs[session_id] = session_dir
        self.frame_counters[session_id] = 0
        
        logger.info(f"Created frame saving directory for session {session_id}: {session_dir}")
        return session_dir
    
    def save_frame(self, session_id: str, frame_data: np.ndarray, 
                   frame_type: str = "frame", metadata: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """
        Save a frame to disk
        
        Args:
            session_id: Session identifier
            frame_data: Frame data as numpy array (H, W, C) in BGR format for OpenCV
            frame_type: Type of frame (e.g., "frame", "keyframe", "processed")
            metadata: Optional metadata to include in filename
            
        Returns:
            Path to saved frame or None if failed
        """
        try:
            # Ensure session directory exists
            if session_id not in self.session_dirs:
                self.setup_session(session_id)
            
            session_dir = self.session_dirs[session_id]
            
            # Increment frame counter
            self.frame_counters[session_id] += 1
            frame_num = self.frame_counters[session_id]
            
            # Create filename
            if metadata and 'frame_id' in metadata:
                filename = f"{frame_type}_{metadata['frame_id']:04d}.jpg"
            else:
                filename = f"{frame_type}_{frame_num:04d}.jpg"
            
            filepath = session_dir / filename
            
            # Ensure frame_data is in the right format for OpenCV
            if isinstance(frame_data, torch.Tensor):
                # Convert tensor to numpy
                if frame_data.device.type == 'cuda':
                    frame_data = frame_data.cpu()
                frame_data = frame_data.numpy()
            
            # Handle different input formats
            if frame_data.dtype != np.uint8:
                # Assume normalized [0,1] or [-1,1] range
                if frame_data.min() >= 0 and frame_data.max() <= 1:
                    frame_data = (frame_data * 255).astype(np.uint8)
                elif frame_data.min() >= -1 and frame_data.max() <= 1:
                    frame_data = ((frame_data + 1) * 127.5).astype(np.uint8)
                else:
                    # Normalize to [0, 255]
                    frame_data = ((frame_data - frame_data.min()) / 
                                 (frame_data.max() - frame_data.min()) * 255).astype(np.uint8)
            
            # Handle different shapes
            if len(frame_data.shape) == 4:  # Batch dimension
                frame_data = frame_data[0]
            
            if len(frame_data.shape) == 3:
                if frame_data.shape[0] == 3:  # CHW format
                    frame_data = np.transpose(frame_data, (1, 2, 0))  # Convert to HWC
                
                # Convert RGB to BGR for OpenCV
                if frame_data.shape[2] == 3:
                    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
            
            # Save the frame
            success = cv2.imwrite(str(filepath), frame_data)
            
            if success:
                logger.info(f"Saved {frame_type} for session {session_id}: {filepath}")
                return filepath
            else:
                logger.error(f"Failed to save {frame_type} for session {session_id}: {filepath}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving frame for session {session_id}: {e}", exc_info=True)
            return None
    
    def save_frame_comparison(self, session_id: str, frame1: np.ndarray, frame2: np.ndarray,
                            label1: str = "keyframe", label2: str = "current", 
                            frame_id: Optional[int] = None) -> Optional[Path]:
        """
        Save a side-by-side comparison of two frames
        
        Args:
            session_id: Session identifier
            frame1: First frame (e.g., keyframe)
            frame2: Second frame (e.g., current frame)
            label1: Label for first frame
            label2: Label for second frame
            frame_id: Optional frame ID for filename
            
        Returns:
            Path to saved comparison image or None if failed
        """
        try:
            # Ensure session directory exists
            if session_id not in self.session_dirs:
                self.setup_session(session_id)
            
            session_dir = self.session_dirs[session_id]
            
            # Process both frames to same format
            def process_frame(frame):
                if isinstance(frame, torch.Tensor):
                    if frame.device.type == 'cuda':
                        frame = frame.cpu()
                    frame = frame.numpy()
                
                if frame.dtype != np.uint8:
                    if frame.min() >= 0 and frame.max() <= 1:
                        frame = (frame * 255).astype(np.uint8)
                    elif frame.min() >= -1 and frame.max() <= 1:
                        frame = ((frame + 1) * 127.5).astype(np.uint8)
                    else:
                        frame = ((frame - frame.min()) / 
                               (frame.max() - frame.min()) * 255).astype(np.uint8)
                
                if len(frame.shape) == 4:
                    frame = frame[0]
                if len(frame.shape) == 3 and frame.shape[0] == 3:
                    frame = np.transpose(frame, (1, 2, 0))
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                return frame
            
            frame1_processed = process_frame(frame1)
            frame2_processed = process_frame(frame2)
            
            # Ensure both frames have same height
            h1, w1 = frame1_processed.shape[:2]
            h2, w2 = frame2_processed.shape[:2]
            
            if h1 != h2:
                # Resize to match heights
                target_h = min(h1, h2)
                frame1_processed = cv2.resize(frame1_processed, (int(w1 * target_h / h1), target_h))
                frame2_processed = cv2.resize(frame2_processed, (int(w2 * target_h / h2), target_h))
            
            # Create side-by-side comparison
            comparison = np.hstack([frame1_processed, frame2_processed])
            
            # Create filename
            if frame_id is not None:
                filename = f"comparison_{label1}_vs_{label2}_{frame_id:04d}.jpg"
            else:
                comp_num = self.frame_counters.get(session_id, 0) + 1
                filename = f"comparison_{label1}_vs_{label2}_{comp_num:04d}.jpg"
            
            filepath = session_dir / filename
            
            # Save comparison
            success = cv2.imwrite(str(filepath), comparison)
            
            if success:
                logger.info(f"Saved frame comparison for session {session_id}: {filepath}")
                return filepath
            else:
                logger.error(f"Failed to save frame comparison for session {session_id}: {filepath}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving frame comparison for session {session_id}: {e}", exc_info=True)
            return None
    
    def cleanup_session(self, session_id: str):
        """
        Clean up session data (but keep saved frames on disk)
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.session_dirs:
            del self.session_dirs[session_id]
        if session_id in self.frame_counters:
            del self.frame_counters[session_id]
        
        logger.info(f"Cleaned up frame saver data for session {session_id}")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session statistics
        """
        if session_id not in self.session_dirs:
            return {"error": "Session not found"}
        
        session_dir = self.session_dirs[session_id]
        frame_count = self.frame_counters.get(session_id, 0)
        
        # Count actual files in directory
        saved_files = list(session_dir.glob("*.jpg"))
        
        return {
            "session_id": session_id,
            "session_dir": str(session_dir),
            "frame_counter": frame_count,
            "saved_files": len(saved_files),
            "files": [f.name for f in saved_files]
        }


# Global frame saver instance
frame_saver = FrameSaver()
