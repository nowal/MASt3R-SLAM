"""
Frame Tracker Wrapper - Integration with original FrameTracker

This module provides a wrapper around the original mast3r_slam.tracker.FrameTracker
to integrate it with our FastAPI session-based architecture. It handles the interface
between our SessionKeyframes and the original SharedKeyframes-expecting FrameTracker.
"""

import logging
import time
from typing import Dict, Any, Tuple, Optional
import torch
import lietorch
from mast3r_slam.frame import Frame, Mode
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.config import config
from .session_keyframes import SessionKeyframes
from .debug_frame_tracker import create_debug_frame_tracker

logger = logging.getLogger("FrameTrackerWrapper")


class FrameTrackerWrapper:
    """
    Wrapper around the original FrameTracker for FastAPI integration
    
    This class adapts the original FrameTracker to work with our session-based
    architecture while maintaining compatibility with the original implementation.
    """
    
    def __init__(self, model, device: str = "cuda:0"):
        """
        Initialize the frame tracker wrapper
        
        Args:
            model: MASt3R model for tracking
            device: Device for computations
        """
        self.model = model
        self.device = device
        self.tracking_stats = {
            "frames_tracked": 0,
            "keyframes_added": 0,
            "tracking_failures": 0,
            "total_tracking_time": 0.0,
            "last_tracking_time": None
        }
        
        logger.info(f"FrameTrackerWrapper initialized with device: {device}")
    
    def create_tracker_for_session(self, session_keyframes: SessionKeyframes) -> FrameTracker:
        """
        Create a FrameTracker instance for a session
        
        Args:
            session_keyframes: SessionKeyframes instance for the session
            
        Returns:
            FrameTracker instance configured for the session
        """
        # Use the debug tracker to get detailed keyframe selection logging
        tracker = create_debug_frame_tracker(self.model, session_keyframes, self.device)
        
        logger.info(f"Created DebugFrameTracker for session with {len(session_keyframes)} existing keyframes")
        return tracker
    
    def track_frame(self, session_id: str, frame: Frame, session_keyframes: SessionKeyframes, 
                   frame_tracker: FrameTracker) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Track a frame using the original FrameTracker
        
        Args:
            session_id: Session identifier for logging
            frame: Frame to track
            session_keyframes: SessionKeyframes for the session
            frame_tracker: FrameTracker instance for the session
            
        Returns:
            Tuple of (add_new_kf, try_reloc, tracking_info)
            - add_new_kf: Whether to add frame as new keyframe
            - try_reloc: Whether to trigger relocalization
            - tracking_info: Dictionary with tracking details and timing
        """
        start_time = time.time()
        self.tracking_stats["frames_tracked"] += 1
        
        logger.info(f"[TRACKING] Starting frame {frame.frame_id} for session {session_id}")
        
        try:
            # Check if we have any keyframes to track against
            if len(session_keyframes) == 0:
                logger.warning(f"[TRACKING] No keyframes available for tracking frame {frame.frame_id}")
                return False, True, {
                    "error": "no_keyframes",
                    "frame_id": frame.frame_id,
                    "session_id": session_id,
                    "duration": time.time() - start_time
                }
            
            # Add debug logging before tracking
            logger.info(f"[TRACKING_DEBUG] Frame {frame.frame_id}: About to call FrameTracker.track()")
            logger.info(f"[TRACKING_DEBUG] Frame {frame.frame_id}: Keyframes available: {len(session_keyframes)}")
            logger.info(f"[TRACKING_DEBUG] Frame {frame.frame_id}: Config match_frac_thresh: {frame_tracker.cfg.get('match_frac_thresh', 'NOT_SET')}")
            
            # Use the original FrameTracker.track() method
            add_new_kf, match_info, try_reloc = frame_tracker.track(frame)
            
            # Add detailed debug logging after tracking
            logger.info(f"[TRACKING_DEBUG] Frame {frame.frame_id}: FrameTracker returned add_new_kf={add_new_kf}, try_reloc={try_reloc}")
            
            end_time = time.time()
            tracking_duration = end_time - start_time
            self.tracking_stats["total_tracking_time"] += tracking_duration
            self.tracking_stats["last_tracking_time"] = start_time
            
            # Extract match information for logging and analysis
            tracking_info = {
                "frame_id": frame.frame_id,
                "session_id": session_id,
                "add_new_kf": add_new_kf,
                "try_reloc": try_reloc,
                "duration": tracking_duration,
                "start_time": start_time,
                "end_time": end_time,
                "num_keyframes_before": len(session_keyframes),
                "last_keyframe_id": session_keyframes.last_keyframe().frame_id if session_keyframes.last_keyframe() else None
            }
            
            # Add match information if available
            if match_info and len(match_info) >= 6:
                # match_info contains: [keyframe.X_canon, keyframe.get_average_conf(), 
                #                      frame.X_canon, frame.get_average_conf(), Qkf, Qff]
                kf_points = match_info[0]
                kf_conf = match_info[1]
                frame_points = match_info[2] 
                frame_conf = match_info[3]
                
                tracking_info.update({
                    "keyframe_points": kf_points.shape[0] if kf_points is not None else 0,
                    "frame_points": frame_points.shape[0] if frame_points is not None else 0,
                    "keyframe_avg_conf": float(kf_conf.mean()) if kf_conf is not None else 0.0,
                    "frame_avg_conf": float(frame_conf.mean()) if frame_conf is not None else 0.0
                })
                
                logger.info(f"[TRACKING_DEBUG] Frame {frame.frame_id}: Match info available with {len(match_info)} elements")
            else:
                logger.warning(f"[TRACKING_DEBUG] Frame {frame.frame_id}: No match info or insufficient elements: {len(match_info) if match_info else 0}")
            
            if add_new_kf:
                self.tracking_stats["keyframes_added"] += 1
                logger.info(
                    f"[TRACKING] SUCCESS - New keyframe needed for frame {frame.frame_id} "
                    f"(session {session_id}) in {tracking_duration:.3f}s"
                )
            elif try_reloc:
                self.tracking_stats["tracking_failures"] += 1
                logger.warning(
                    f"[TRACKING] FAILURE - Relocalization needed for frame {frame.frame_id} "
                    f"(session {session_id}) in {tracking_duration:.3f}s"
                )
            else:
                logger.info(
                    f"[TRACKING] SUCCESS - Frame {frame.frame_id} tracked successfully "
                    f"(session {session_id}) in {tracking_duration:.3f}s"
                )
            
            return add_new_kf, try_reloc, tracking_info
            
        except Exception as e:
            end_time = time.time()
            tracking_duration = end_time - start_time
            self.tracking_stats["tracking_failures"] += 1
            
            logger.error(
                f"[TRACKING] ERROR - Exception during tracking frame {frame.frame_id} "
                f"(session {session_id}): {e}", exc_info=True)
            
            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "frame_id": frame.frame_id,
                "session_id": session_id,
                "duration": tracking_duration,
                "start_time": start_time,
                "end_time": end_time
            }
            
            # Tracking failure should trigger relocalization
            return False, True, error_info
    
    def get_tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration from the loaded config"""
        return config.get("tracking", {})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        avg_tracking_time = (
            self.tracking_stats["total_tracking_time"] / self.tracking_stats["frames_tracked"]
            if self.tracking_stats["frames_tracked"] > 0 else 0.0
        )
        
        success_rate = (
            (self.tracking_stats["frames_tracked"] - self.tracking_stats["tracking_failures"]) / 
            self.tracking_stats["frames_tracked"]
            if self.tracking_stats["frames_tracked"] > 0 else 0.0
        )
        
        keyframe_rate = (
            self.tracking_stats["keyframes_added"] / self.tracking_stats["frames_tracked"]
            if self.tracking_stats["frames_tracked"] > 0 else 0.0
        )
        
        return {
            **self.tracking_stats,
            "average_tracking_time": avg_tracking_time,
            "success_rate": success_rate,
            "keyframe_rate": keyframe_rate,
            "tracking_config": self.get_tracking_config(),
            "device": self.device
        }
    
    def reset_stats(self) -> None:
        """Reset tracking statistics"""
        logger.info("Resetting frame tracking statistics")
        self.tracking_stats = {
            "frames_tracked": 0,
            "keyframes_added": 0,
            "tracking_failures": 0,
            "total_tracking_time": 0.0,
            "last_tracking_time": None
        }


def create_frame_tracker_wrapper(model, device: str = "cuda:0") -> FrameTrackerWrapper:
    """
    Factory function to create a FrameTrackerWrapper
    
    Args:
        model: MASt3R model for tracking
        device: Device for computations
        
    Returns:
        FrameTrackerWrapper instance
    """
    return FrameTrackerWrapper(model, device)


def validate_tracking_requirements(session_keyframes: SessionKeyframes, frame: Frame) -> Tuple[bool, str]:
    """
    Validate that tracking requirements are met
    
    Args:
        session_keyframes: SessionKeyframes for the session
        frame: Frame to be tracked
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(session_keyframes) == 0:
        return False, "No keyframes available for tracking"
    
    last_keyframe = session_keyframes.last_keyframe()
    if last_keyframe is None:
        return False, "Last keyframe is None"
    
    if last_keyframe.X_canon is None or last_keyframe.C is None:
        return False, "Last keyframe missing point cloud data"
    
    if frame.img is None:
        return False, "Frame missing image data"
    
    # Check device compatibility
    if hasattr(frame.img, 'device') and hasattr(last_keyframe.X_canon, 'device'):
        if frame.img.device != last_keyframe.X_canon.device:
            return False, f"Device mismatch: frame on {frame.img.device}, keyframe on {last_keyframe.X_canon.device}"
    
    return True, ""


# Helper function to extract match fraction from tracking results
def extract_match_fraction_from_tracker_output(match_info) -> Optional[float]:
    """
    Extract match fraction from FrameTracker output for analysis
    
    This is useful for logging and determining relocalization triggers.
    Note: The original FrameTracker doesn't directly return match fraction,
    so this is an approximation based on available information.
    
    Args:
        match_info: Match information from FrameTracker.track()
        
    Returns:
        Estimated match fraction or None if not available
    """
    if not match_info or len(match_info) < 6:
        return None
    
    try:
        # match_info contains confidence information we can use to estimate match quality
        kf_conf = match_info[1]  # keyframe confidence
        frame_conf = match_info[3]  # frame confidence
        
        if kf_conf is not None and frame_conf is not None:
            # Simple heuristic: use average confidence as proxy for match quality
            avg_conf = (kf_conf.mean() + frame_conf.mean()) / 2.0
            return float(avg_conf)
    except Exception as e:
        logger.debug(f"Could not extract match fraction: {e}")
    
    return None
