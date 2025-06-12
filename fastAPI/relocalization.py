"""
Real Relocalization Implementation for MASt3R-SLAM FastAPI System

This module provides real relocalization functionality that recovers from tracking failures
by finding similar keyframes in the database and performing strict geometric validation.
Follows the exact logic from main.py relocalization() function.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional
import torch
import lietorch

from mast3r_slam.config import config
from mast3r_slam.frame import Frame, Mode
from mast3r_slam.mast3r_utils import load_retriever, mast3r_inference_mono
from mast3r_slam.global_opt import FactorGraph

logger = logging.getLogger("Relocalization")


class RealRelocalizer:
    """
    Real relocalization for recovering from tracking failures
    
    This class implements the exact relocalization logic from main.py, including:
    - Retrieval database queries for similar keyframes
    - Strict geometric validation with is_reloc=True
    - Temporary keyframe addition/removal during validation
    - 5-second timeout with user guidance messages
    """
    
    def __init__(self, model, device="cuda"):
        """
        Initialize the real relocalizer
        
        Args:
            model: MASt3R model for feature matching and relocalization
            device: Device to run relocalization on (default: "cuda")
        """
        self.model = model
        self.device = device
        self.relocalization_count = 0
        self.total_relocalization_time = 0.0
        self.last_relocalization_time = None
        
        # Session-specific tracking
        self.retrieval_databases = {}  # session_id -> retrieval_database
        self.reloc_start_times = {}    # session_id -> start_timestamp
        self.guidance_sent = {}        # session_id -> bool (sent once)
        self.relocalization_attempts = []  # Track all attempts for analysis
        
        logger.info(f"RealRelocalizer initialized on device {device}")
    
    async def relocalize_frame_async(self, session_data, frame: Frame, reason: str = "tracking_failure") -> Dict[str, Any]:
        """
        Attempt real relocalization following main.py logic exactly
        
        Args:
            session_data: Session data containing keyframes and SLAM state
            frame: Frame that triggered relocalization
            reason: Reason for relocalization (e.g., "tracking_failure", "relocalization_mode")
            
        Returns:
            Dictionary with relocalization results and new mode
        """
        start_time = time.time()
        self.relocalization_count += 1
        session_id = getattr(session_data, 'session_id', 'unknown')
        
        logger.info(f"[RELOC] Starting real relocalization #{self.relocalization_count} for session {session_id}, frame {frame.frame_id}, reason: {reason}")
        
        try:
            # Track relocalization session timing
            await self._track_relocalization_timing(session_data, session_id, start_time)
            
            # Get shared retrieval database for this session
            retrieval_database = await self._get_retrieval_database(session_data, session_id)
            if retrieval_database is None:
                return self._create_error_result("failed_to_create_retrieval_database", start_time, session_id, frame.frame_id)
            
            # Step 1: Generate 3D points for the frame (required for relocalization)
            logger.info(f"[RELOC] Generating 3D points for frame {frame.frame_id}")
            try:
                X_init, C_init = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(X_init, C_init)
                logger.info(f"[RELOC] Generated 3D points for frame {frame.frame_id}: {X_init.shape}")
            except Exception as e:
                logger.error(f"[RELOC] Failed to generate 3D points for frame {frame.frame_id}: {e}")
                return self._create_error_result(f"3d_point_generation_failed: {str(e)}", start_time, session_id, frame.frame_id)
            
            # Step 2: Query retrieval database for similar keyframes (add_after_query=False initially)
            logger.info(f"[RELOC] Querying retrieval database for similar keyframes")
            try:
                retrieval_inds = retrieval_database.update(
                    frame,
                    add_after_query=False,  # Don't add to database yet
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"]
                )
                logger.info(f"[RELOC] Database query returned {len(retrieval_inds) if retrieval_inds else 0} candidate keyframes: {retrieval_inds}")
            except Exception as e:
                logger.error(f"[RELOC] Database query failed: {e}")
                return self._create_error_result(f"database_query_failed: {str(e)}", start_time, session_id, frame.frame_id)
            
            # Step 3: Attempt relocalization if we have candidates
            successful_relocalization = False
            if retrieval_inds:
                successful_relocalization = await self._attempt_geometric_validation(
                    session_data, frame, retrieval_inds, retrieval_database
                )
            else:
                logger.info(f"[RELOC] No candidate keyframes found for frame {frame.frame_id}")
            
            # Calculate timing
            end_time = time.time()
            duration = end_time - start_time
            self.total_relocalization_time += duration
            
            # Create result
            result = self._create_relocalization_result(
                successful_relocalization, session_id, frame.frame_id, reason,
                start_time, end_time, duration, retrieval_inds
            )
            
            # Handle success/failure
            if successful_relocalization:
                # Clear relocalization tracking on success
                self.reloc_start_times.pop(session_id, None)
                self.guidance_sent.pop(session_id, None)
                
                logger.info(
                    f"[RELOC] SUCCESS #{self.relocalization_count} for session {session_id}, "
                    f"frame {frame.frame_id} in {duration:.3f}s - returning to TRACKING mode"
                )
                result["new_mode"] = Mode.TRACKING
            else:
                logger.info(
                    f"[RELOC] FAILED #{self.relocalization_count} for session {session_id}, "
                    f"frame {frame.frame_id} in {duration:.3f}s - staying in RELOC mode"
                )
                result["new_mode"] = Mode.RELOC
            
            # Track attempt for analysis
            self.relocalization_attempts.append({
                "timestamp": start_time,
                "session_id": session_id,
                "frame_id": frame.frame_id,
                "reason": reason,
                "success": successful_relocalization,
                "duration": duration,
                "candidates_found": len(retrieval_inds) if retrieval_inds else 0
            })
            
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"[RELOC] Exception during relocalization for frame {frame.frame_id}: {e}", exc_info=True)
            return self._create_error_result(f"relocalization_exception: {str(e)}", start_time, session_id, frame.frame_id)
    
    async def _track_relocalization_timing(self, session_data, session_id: str, current_time: float):
        """Track relocalization timing and send guidance message if needed"""
        # Track when relocalization started for this session
        if session_id not in self.reloc_start_times:
            self.reloc_start_times[session_id] = current_time
            self.guidance_sent[session_id] = False
            logger.info(f"[RELOC] Started relocalization timer for session {session_id}")
        
        # Check timeout for guidance message (send once after 5 seconds)
        reloc_start_time = self.reloc_start_times[session_id]
        time_in_reloc = current_time - reloc_start_time
        timeout_seconds = config["reloc"]["timeout_seconds"]
        
        if time_in_reloc > timeout_seconds and not self.guidance_sent[session_id]:
            await self._send_relocalization_guidance(session_data, session_id, time_in_reloc)
            self.guidance_sent[session_id] = True
    
    async def _send_relocalization_guidance(self, session_data, session_id: str, time_elapsed: float):
        """Send guidance message to user via debug logging"""
        guidance_message = "Couldn't match up new footage, please go back to previously captured area"
        
        logger.warning(f"[RELOC_GUIDANCE] Session {session_id}: {guidance_message} (elapsed: {time_elapsed:.1f}s)")
        
        # Send via connection manager if available
        try:
            from .connection_manager import connection_manager
            await connection_manager.send_debug_message(
                session_id, "warning", guidance_message
            )
        except Exception as e:
            logger.debug(f"Could not send guidance via connection manager: {e}")
    
    async def _get_retrieval_database(self, session_data, session_id: str):
        """Get shared retrieval database from session data"""
        if hasattr(session_data, 'retrieval_database') and session_data.retrieval_database is not None:
            logger.info(f"[RELOC] Using shared retrieval database for session {session_id}")
            return session_data.retrieval_database
        else:
            logger.error(f"[RELOC] No shared retrieval database found for session {session_id}")
            return None
    
    async def _attempt_geometric_validation(self, session_data, frame: Frame, retrieval_inds, retrieval_database) -> bool:
        """
        Attempt geometric validation following main.py logic exactly
        
        This is the core relocalization logic from main.py:
        1. Temporarily add frame to keyframes
        2. Try to add factors with strict validation (is_reloc=True)
        3. If successful: add to database permanently, set pose, optimize
        4. If failed: remove frame from keyframes
        """
        session_id = getattr(session_data, 'session_id', 'unknown')
        
        # We need keyframes and factor graph for validation
        if not hasattr(session_data, 'keyframes') or session_data.keyframes is None:
            logger.error(f"[RELOC] No keyframes available for session {session_id}")
            return False
        
        if not hasattr(session_data, 'global_optimizer') or session_data.global_optimizer is None:
            logger.error(f"[RELOC] No global optimizer available for session {session_id}")
            return False
        
        if session_data.global_optimizer.factor_graph is None:
            logger.error(f"[RELOC] No factor graph available for session {session_id}")
            return False
        
        keyframes = session_data.keyframes
        factor_graph = session_data.global_optimizer.factor_graph
        
        logger.info(f"[RELOC] Attempting geometric validation against keyframes {retrieval_inds}")
        
        # Step 3a: Analyze matches with each candidate keyframe before validation
        await self._analyze_candidate_matches(session_data, frame, retrieval_inds)
        
        # Step 3b: Temporarily add frame to keyframes (following main.py exactly)
        try:
            keyframes.append(frame)
            n_kf = len(keyframes)
            logger.info(f"[RELOC] Temporarily added frame {frame.frame_id} as keyframe {n_kf-1} (total: {n_kf})")
        except Exception as e:
            logger.error(f"[RELOC] Failed to add frame to keyframes: {e}")
            return False
        
        # Step 3c: Try strict geometric validation with factor graph
        try:
            kf_idx = list(retrieval_inds)  # Convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)  # Current frame index repeated
            
            logger.info(f"[RELOC] Attempting factor graph validation: frame_idx={frame_idx}, kf_idx={kf_idx}")
            logger.info(f"[RELOC] Using strict validation with min_match_frac={config['reloc']['min_match_frac']}, is_reloc={config['reloc']['strict']}")
            
            # This is the key validation step from main.py
            factors_added = factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],  # 0.3 (stricter than tracking)
                is_reloc=config["reloc"]["strict"]   # True (strict validation)
            )
            
            if factors_added:
                # SUCCESS: Validation passed
                logger.info(f"[RELOC] Geometric validation PASSED for frame {frame.frame_id}")
                
                # Step 3c: Add frame to retrieval database permanently
                try:
                    retrieval_database.update(
                        frame,
                        add_after_query=True,  # Now add permanently
                        k=config["retrieval"]["k"],
                        min_thresh=config["retrieval"]["min_thresh"]
                    )
                    logger.info(f"[RELOC] Added frame {frame.frame_id} to retrieval database permanently")
                except Exception as e:
                    logger.warning(f"[RELOC] Failed to add frame to database permanently: {e}")
                
                # Step 3d: Initialize pose from matched keyframe (main.py logic)
                try:
                    keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
                    logger.info(f"[RELOC] Initialized pose for frame {frame.frame_id} from keyframe {kf_idx[0]}")
                except Exception as e:
                    logger.warning(f"[RELOC] Failed to initialize pose: {e}")
                
                # Step 3e: Trigger optimization (like main.py)
                try:
                    if config.get("use_calib", False):
                        factor_graph.solve_GN_calib()
                        logger.info(f"[RELOC] Performed calibrated optimization")
                    else:
                        factor_graph.solve_GN_rays()
                        logger.info(f"[RELOC] Performed ray-based optimization")
                except Exception as e:
                    logger.warning(f"[RELOC] Optimization failed: {e}")
                
                return True
                
            else:
                # FAILURE: Validation failed
                logger.info(f"[RELOC] Geometric validation FAILED for frame {frame.frame_id}")
                
                # Step 3f: Remove frame from keyframes (main.py logic)
                try:
                    keyframes.pop_last()
                    logger.info(f"[RELOC] Removed frame {frame.frame_id} from keyframes after validation failure")
                except Exception as e:
                    logger.error(f"[RELOC] Failed to remove frame from keyframes: {e}")
                
                return False
                
        except Exception as e:
            logger.error(f"[RELOC] Exception during geometric validation: {e}")
            
            # Clean up: try to remove frame if it was added
            try:
                if len(keyframes) > 0 and keyframes[-1].frame_id == frame.frame_id:
                    keyframes.pop_last()
                    logger.info(f"[RELOC] Cleaned up frame {frame.frame_id} after exception")
            except Exception as cleanup_error:
                logger.error(f"[RELOC] Failed to cleanup frame after exception: {cleanup_error}")
            
            return False
    
    async def _analyze_candidate_matches(self, session_data, frame: Frame, retrieval_inds):
        """
        Analyze matches between current frame and each candidate keyframe
        
        This provides detailed logging to understand why geometric validation might fail
        """
        session_id = getattr(session_data, 'session_id', 'unknown')
        keyframes = session_data.keyframes
        min_match_frac = config["reloc"]["min_match_frac"]
        
        logger.info(f"[RELOC_DETAIL] Analyzing {len(retrieval_inds)} candidate keyframes for frame {frame.frame_id}")
        logger.info(f"[RELOC_DETAIL] Required minimum match fraction: {min_match_frac:.3f} ({min_match_frac*100:.1f}%)")
        
        try:
            # Import the MASt3R model utilities for feature matching
            from mast3r_slam.mast3r_utils import mast3r_inference_mono
            
            for i, candidate_kf_idx in enumerate(retrieval_inds):
                if candidate_kf_idx >= len(keyframes):
                    logger.warning(f"[RELOC_DETAIL] Candidate {i+1}: Invalid keyframe index {candidate_kf_idx} (only {len(keyframes)} keyframes)")
                    continue
                
                candidate_keyframe = keyframes[candidate_kf_idx]
                
                try:
                    # Perform feature matching between current frame and candidate keyframe
                    # This mimics what the factor graph will do internally
                    logger.info(f"[RELOC_DETAIL] Candidate {i+1}: Analyzing keyframe {candidate_kf_idx} (frame_id: {candidate_keyframe.frame_id})")
                    
                    # Use MASt3R to get matches between the two frames
                    # Note: This is a simplified analysis - the actual factor graph does more complex validation
                    matches_info = await self._get_frame_matches(frame, candidate_keyframe)
                    
                    if matches_info:
                        valid_matches = matches_info.get('valid_matches', 0)
                        total_points = matches_info.get('total_points', 1)
                        match_fraction = valid_matches / total_points if total_points > 0 else 0.0
                        
                        # Determine if this candidate would likely pass the match threshold
                        threshold_status = "PASS" if match_fraction >= min_match_frac else "FAIL"
                        threshold_color = "✅" if match_fraction >= min_match_frac else "❌"
                        
                        logger.info(f"[RELOC_DETAIL] Candidate {i+1}: {threshold_color} {threshold_status}")
                        logger.info(f"[RELOC_DETAIL]   - Valid matches: {valid_matches:,} / {total_points:,} ({match_fraction:.3f} = {match_fraction*100:.1f}%)")
                        logger.info(f"[RELOC_DETAIL]   - Required: {min_match_frac:.3f} ({min_match_frac*100:.1f}%)")
                        
                        if match_fraction >= min_match_frac:
                            logger.info(f"[RELOC_DETAIL]   - Prediction: Should pass match threshold, may still fail geometric validation")
                        else:
                            deficit = min_match_frac - match_fraction
                            logger.info(f"[RELOC_DETAIL]   - Prediction: Will fail match threshold (deficit: {deficit:.3f} = {deficit*100:.1f}%)")
                    else:
                        logger.warning(f"[RELOC_DETAIL] Candidate {i+1}: ❌ FAIL - Could not analyze matches")
                        logger.warning(f"[RELOC_DETAIL]   - Prediction: Will fail due to match analysis error")
                        
                except Exception as e:
                    logger.warning(f"[RELOC_DETAIL] Candidate {i+1}: ❌ FAIL - Analysis error: {str(e)}")
                    logger.warning(f"[RELOC_DETAIL]   - Prediction: Will fail due to analysis exception")
            
            logger.info(f"[RELOC_DETAIL] Completed analysis of all {len(retrieval_inds)} candidates")
            
        except Exception as e:
            logger.error(f"[RELOC_DETAIL] Failed to analyze candidate matches: {e}")
    
    async def _get_frame_matches(self, frame1: Frame, frame2: Frame):
        """
        Get match statistics between two frames
        
        This is a simplified version of what the factor graph does internally
        """
        try:
            # Check if both frames have the required data
            if not hasattr(frame1, 'X_canon') or frame1.X_canon is None:
                return None
            if not hasattr(frame2, 'X_canon') or frame2.X_canon is None:
                return None
            
            # Get 3D points from both frames
            X1 = frame1.X_canon  # 3D points from frame 1
            X2 = frame2.X_canon  # 3D points from frame 2
            
            if X1 is None or X2 is None:
                return None
            
            # Simple distance-based matching (this is a rough approximation)
            # The actual factor graph uses much more sophisticated matching
            total_points = X1.shape[0] if len(X1.shape) > 1 else 1
            
            # For now, we'll estimate based on the confidence values if available
            if hasattr(frame1, 'C') and frame1.C is not None and hasattr(frame2, 'C') and frame2.C is not None:
                # Use confidence values as a proxy for match quality
                C1 = frame1.C.flatten() if len(frame1.C.shape) > 1 else frame1.C
                C2 = frame2.C.flatten() if len(frame2.C.shape) > 1 else frame2.C
                
                # Estimate valid matches based on confidence thresholds
                # This is a rough approximation - actual matching is much more complex
                conf_thresh = config.get("local_opt", {}).get("C_conf", 0.0)
                valid1 = (C1 > conf_thresh).sum().item() if hasattr(C1, 'sum') else 0
                valid2 = (C2 > conf_thresh).sum().item() if hasattr(C2, 'sum') else 0
                
                # Estimate overlap (very rough approximation)
                estimated_valid_matches = min(valid1, valid2) // 2  # Conservative estimate
                
                return {
                    'valid_matches': estimated_valid_matches,
                    'total_points': total_points,
                    'frame1_valid': valid1,
                    'frame2_valid': valid2
                }
            else:
                # Fallback: assume some percentage of points are valid matches
                estimated_valid_matches = total_points // 4  # Very conservative estimate
                return {
                    'valid_matches': estimated_valid_matches,
                    'total_points': total_points
                }
                
        except Exception as e:
            logger.debug(f"Error in _get_frame_matches: {e}")
            return None
    
    def _create_relocalization_result(self, success: bool, session_id: str, frame_id: int, reason: str,
                                    start_time: float, end_time: float, duration: float, retrieval_inds) -> Dict[str, Any]:
        """Create relocalization result dictionary"""
        return {
            "session_id": session_id,
            "frame_id": frame_id,
            "reason": reason,
            "relocalization_count": self.relocalization_count,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "success": success,
            "status": "relocalization_completed",
            "candidates_found": len(retrieval_inds) if retrieval_inds else 0,
            "candidate_keyframes": list(retrieval_inds) if retrieval_inds else []
        }
    
    def _create_error_result(self, error_msg: str, start_time: float, session_id: str, frame_id: int) -> Dict[str, Any]:
        """Create error result dictionary"""
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "session_id": session_id,
            "frame_id": frame_id,
            "relocalization_count": self.relocalization_count,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "success": False,
            "status": "relocalization_failed",
            "error": error_msg,
            "candidates_found": 0,
            "candidate_keyframes": [],
            "new_mode": Mode.RELOC
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get relocalization statistics"""
        avg_duration = (
            self.total_relocalization_time / self.relocalization_count 
            if self.relocalization_count > 0 else 0.0
        )
        
        # Analyze attempts
        success_count = sum(1 for attempt in self.relocalization_attempts if attempt["success"])
        success_rate = success_count / len(self.relocalization_attempts) if self.relocalization_attempts else 0.0
        
        # Analyze by reason
        reason_stats = {}
        for attempt in self.relocalization_attempts:
            reason = attempt["reason"]
            if reason not in reason_stats:
                reason_stats[reason] = {"total": 0, "success": 0}
            reason_stats[reason]["total"] += 1
            if attempt["success"]:
                reason_stats[reason]["success"] += 1
        
        # Calculate success rates by reason
        for reason in reason_stats:
            stats = reason_stats[reason]
            stats["success_rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return {
            "relocalization_count": self.relocalization_count,
            "total_relocalization_time": self.total_relocalization_time,
            "average_duration": avg_duration,
            "last_relocalization_time": self.last_relocalization_time,
            "success_count": success_count,
            "success_rate": success_rate,
            "reason_stats": reason_stats,
            "active_sessions": len(self.reloc_start_times),
            "sessions_with_guidance_sent": len([s for s in self.guidance_sent.values() if s]),
            "status": "real_relocalization_active",
            "device": self.device
        }
    
    def reset_stats(self) -> None:
        """Reset relocalization statistics"""
        logger.info("Resetting relocalization statistics")
        self.relocalization_count = 0
        self.total_relocalization_time = 0.0
        self.last_relocalization_time = None
        self.relocalization_attempts.clear()
        # Note: Don't reset session tracking (reloc_start_times, guidance_sent) as they're for active sessions


# Global instance for the real relocalization
real_relocalization = RealRelocalizer(None, "cuda")  # Model will be set during session setup


# Compatibility functions for existing code
async def relocalize_frame_async(session_data, frame: Frame, reason: str = "tracking_failure") -> Dict[str, Any]:
    """
    Asynchronous function to attempt relocalization
    
    This function is called from the data receiver and performs
    real relocalization following main.py logic.
    
    Args:
        session_data: Session data containing keyframes and SLAM state
        frame: Frame that triggered relocalization
        reason: Reason for relocalization
        
    Returns:
        Dictionary with relocalization results
    """
    # Use the session's relocalizer if available, otherwise use global instance
    if hasattr(session_data, 'relocalizer') and session_data.relocalizer is not None:
        return await session_data.relocalizer.relocalize_frame_async(session_data, frame, reason)
    else:
        logger.warning("No session relocalizer found, using global instance")
        return await real_relocalization.relocalize_frame_async(session_data, frame, reason)


def get_relocalization_stats() -> Dict[str, Any]:
    """Get relocalization statistics"""
    return real_relocalization.get_stats()


def reset_relocalization_stats() -> None:
    """Reset relocalization statistics"""
    real_relocalization.reset_stats()


# Legacy compatibility (these functions are no longer used but kept for compatibility)
def attempt_relocalization(session_id: str, frame: Frame, reason: str = "tracking_failure") -> Dict[str, Any]:
    """Legacy synchronous relocalization (not used in real implementation)"""
    logger.warning("Legacy attempt_relocalization called - this should not happen in real implementation")
    return {
        "success": False,
        "error": "legacy_function_called",
        "session_id": session_id,
        "frame_id": frame.frame_id,
        "reason": reason
    }


async def attempt_relocalization_async(session_id: str, frame: Frame, reason: str = "tracking_failure") -> Dict[str, Any]:
    """Legacy async relocalization (not used in real implementation)"""
    logger.warning("Legacy attempt_relocalization_async called - this should not happen in real implementation")
    return await attempt_relocalization(session_id, frame, reason)


def add_relocalization_task(background_tasks, session_id: str, frame: Frame, reason: str = "tracking_failure"):
    """Legacy background task function (not used in real implementation)"""
    logger.warning("Legacy add_relocalization_task called - this should not happen in real implementation")
    # In real implementation, relocalization is called directly from data_receiver


def should_trigger_relocalization(match_fraction: float, min_match_frac: float) -> bool:
    """
    Determine if relocalization should be triggered based on tracking quality
    
    This mimics the logic from the original FrameTracker.track() method.
    
    Args:
        match_fraction: Fraction of valid matches found
        min_match_frac: Minimum required match fraction
        
    Returns:
        True if relocalization should be triggered
    """
    should_relocalize = match_fraction < min_match_frac
    
    if should_relocalize:
        logger.info(f"[RELOC] Trigger condition met: match_fraction {match_fraction:.3f} < min_match_frac {min_match_frac:.3f}")
    
    return should_relocalize
