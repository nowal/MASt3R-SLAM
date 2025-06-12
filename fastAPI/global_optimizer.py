"""
Real Global Optimizer - Bundle Adjustment and Loop Closure Detection

This module provides real global optimization using FactorGraph for bundle adjustment
and RetrievalDatabase for loop closure detection. It replaces the dummy implementation
with actual SLAM backend functionality.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional
import torch
import lietorch

from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.mast3r_utils import load_retriever
from mast3r_slam.config import config
from mast3r_slam.frame import Frame

logger = logging.getLogger("GlobalOptimizer")


def copy_frame_safely(original_frame: Frame) -> Frame:
    """
    Safely copy a Frame object while preserving exact tensor structures
    
    This function manually copies each field to avoid the tensor shape corruption
    that occurs with copy.deepcopy() on lietorch.Sim3 objects.
    
    Args:
        original_frame: Frame object to copy
        
    Returns:
        New Frame object with identical data but preserved tensor shapes
    """
    # Log original tensor shape for debugging
    original_pose_shape = original_frame.T_WC.data.shape
    logger.info(f"[TENSOR_DEBUG] Original frame {original_frame.frame_id} pose shape: {original_pose_shape}")
    
    # Create new frame with manually copied fields to preserve tensor structures
    copied_frame = Frame(
        frame_id=original_frame.frame_id,
        img=original_frame.img.clone(),
        img_shape=original_frame.img_shape.clone(),
        img_true_shape=original_frame.img_true_shape.clone(),
        uimg=original_frame.uimg.clone(),
        T_WC=lietorch.Sim3(original_frame.T_WC.data.clone()),  # Key fix: preserve [1, 8] shape
    )
    
    # Log copied tensor shape for debugging
    copied_pose_shape = copied_frame.T_WC.data.shape
    logger.info(f"[TENSOR_DEBUG] Copied frame {copied_frame.frame_id} pose shape: {copied_pose_shape}")
    
    # Verify shapes match
    if original_pose_shape != copied_pose_shape:
        logger.error(f"[TENSOR_DEBUG] SHAPE MISMATCH! Original: {original_pose_shape}, Copied: {copied_pose_shape}")
    else:
        logger.info(f"[TENSOR_DEBUG] ✓ Shape preserved for frame {copied_frame.frame_id}")
    
    # Copy optional fields if they exist
    if original_frame.X_canon is not None:
        copied_frame.X_canon = original_frame.X_canon.clone()
    if original_frame.C is not None:
        copied_frame.C = original_frame.C.clone()
    if original_frame.feat is not None:
        copied_frame.feat = original_frame.feat.clone()
    if original_frame.pos is not None:
        copied_frame.pos = original_frame.pos.clone()
    if original_frame.K is not None:
        copied_frame.K = original_frame.K.clone()
    
    # Copy scalar fields
    copied_frame.N = original_frame.N
    copied_frame.N_updates = original_frame.N_updates
    
    return copied_frame


class RealGlobalOptimizer:
    """
    Real global optimizer with bundle adjustment and loop closure detection
    
    This class provides actual SLAM backend functionality including:
    - Factor graph construction and optimization
    - Loop closure detection via retrieval database
    - Camera pose refinement through bundle adjustment
    """
    
    def __init__(self, model, device="cuda"):
        """
        Initialize the real global optimizer for synchronous operation
        
        Args:
            model: MASt3R model for feature matching and optimization
            device: Device to run optimization on (default: "cuda")
        """
        self.model = model
        self.device = device
        self.optimization_count = 0
        self.total_optimization_time = 0.0
        self.last_optimization_time = None
        
        # Factor graph will be created when first keyframe is processed
        self.factor_graph = None
        
        logger.info(f"RealGlobalOptimizer initialized on device {device} (synchronous mode)")
    
    async def optimize_keyframe_sync(self, session_data, keyframe_idx: int) -> Dict[str, Any]:
        """
        Perform synchronous global optimization using shared keyframes database
        
        Args:
            session_data: Session data containing shared keyframes and SLAM state
            keyframe_idx: Index of the keyframe that triggered optimization
            
        Returns:
            Dictionary with optimization results and timing
        """
        start_time = time.time()
        self.optimization_count += 1
        
        logger.info(f"[GLOBAL_OPT] Starting synchronous optimization #{self.optimization_count} for keyframe {keyframe_idx}")
        
        try:
            # Use shared keyframes directly (no separate optimization keyframes)
            keyframes = session_data.keyframes
            retrieval_database = session_data.retrieval_database
            
            if keyframes is None:
                logger.error("No keyframes available for optimization")
                return self._create_error_result("no_keyframes_available", start_time)
            
            if keyframe_idx >= len(keyframes):
                logger.error(f"Invalid keyframe index {keyframe_idx}, only {len(keyframes)} keyframes available")
                return self._create_error_result("invalid_keyframe_index", start_time)
            
            # Initialize factor graph if this is the first optimization
            if self.factor_graph is None:
                self._initialize_factor_graph_sync(session_data)
            
            # Get the frame that triggered optimization
            frame = keyframes[keyframe_idx]
            
            # Build list of keyframes to optimize against
            kf_idx = []
            
            # Add previous consecutive keyframes (local window)
            n_consec = min(config["local_opt"].get("window_size", 1), keyframe_idx)
            for j in range(min(n_consec, keyframe_idx)):
                kf_idx.append(keyframe_idx - 1 - j)
            
            # Query retrieval database for loop closures
            retrieval_inds = []
            if retrieval_database is not None:
                try:
                    retrieval_inds = retrieval_database.update(
                        frame,
                        add_after_query=True,
                        k=config["retrieval"]["k"],
                        min_thresh=config["retrieval"]["min_thresh"]
                    )
                    kf_idx += retrieval_inds
                    
                    # Log loop closures
                    lc_inds = set(retrieval_inds)
                    lc_inds.discard(keyframe_idx - 1)  # Remove consecutive frame
                    if len(lc_inds) > 0:
                        logger.info(f"[LOOP_CLOSURE] Detected loop closures for keyframe {keyframe_idx}: {lc_inds}")
                        
                except Exception as e:
                    logger.warning(f"Retrieval database query failed: {e}")
            
            # Remove duplicates and current keyframe
            kf_idx = list(set(kf_idx))
            if keyframe_idx in kf_idx:
                kf_idx.remove(keyframe_idx)
            
            # Add factors (constraints) between keyframes
            factors_added = False
            if kf_idx:
                try:
                    frame_idx = [keyframe_idx] * len(kf_idx)
                    factors_added = self.factor_graph.add_factors(
                        kf_idx, 
                        frame_idx, 
                        config["local_opt"]["min_match_frac"]
                    )
                    
                    if factors_added:
                        logger.info(f"[FACTORS] Added constraints between keyframe {keyframe_idx} and keyframes {kf_idx}")
                    else:
                        logger.warning(f"[FACTORS] Failed to add sufficient constraints for keyframe {keyframe_idx}")
                        
                except Exception as e:
                    logger.error(f"Failed to add factors: {e}")
                    return self._create_error_result(f"factor_addition_failed: {str(e)}", start_time)
            
            # Perform bundle adjustment if we have constraints
            if factors_added:
                try:
                    optimization_start = time.time()
                    
                    # Choose optimization method based on calibration
                    if config.get("use_calib", False):
                        self.factor_graph.solve_GN_calib()
                        method = "solve_GN_calib"
                    else:
                        self.factor_graph.solve_GN_rays()
                        method = "solve_GN_rays"
                    
                    optimization_time = time.time() - optimization_start
                    logger.info(f"[BUNDLE_ADJUSTMENT] Completed {method} in {optimization_time:.3f}s")
                    
                except Exception as e:
                    logger.error(f"Bundle adjustment failed: {e}")
                    return self._create_error_result(f"bundle_adjustment_failed: {str(e)}", start_time)
            else:
                logger.info(f"[SKIP] No factors added for keyframe {keyframe_idx}, skipping bundle adjustment")
            
            # Calculate timing
            end_time = time.time()
            duration = end_time - start_time
            self.total_optimization_time += duration
            
            # Calculate time since last optimization
            time_since_last = None
            if self.last_optimization_time is not None:
                time_since_last = start_time - self.last_optimization_time
            self.last_optimization_time = start_time
            
            # Create success result
            result = {
                "session_id": getattr(session_data, 'session_id', 'unknown'),
                "keyframe_idx": keyframe_idx,
                "optimization_count": self.optimization_count,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "time_since_last": time_since_last,
                "status": "optimization_completed",
                "factors_added": factors_added,
                "keyframes_optimized_against": kf_idx,
                "loop_closures_detected": len(retrieval_inds) if retrieval_inds else 0,
                "total_keyframes": len(keyframes)
            }
            
            logger.info(
                f"[GLOBAL_OPT] Completed synchronous #{self.optimization_count} for keyframe {keyframe_idx} "
                f"in {duration:.3f}s (factors: {factors_added}, loop_closures: {len(retrieval_inds) if retrieval_inds else 0})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Synchronous global optimization failed for keyframe {keyframe_idx}: {e}", exc_info=True)
            return self._create_error_result(f"optimization_failed: {str(e)}", start_time)
    
    async def optimize_keyframe(self, session_data, keyframe_idx: int) -> Dict[str, Any]:
        """
        Perform real global optimization for a new keyframe using separate optimization keyframes
        
        Args:
            session_data: Session data containing keyframes and SLAM state
            keyframe_idx: Index of the keyframe that triggered optimization
            
        Returns:
            Dictionary with optimization results and timing
        """
        start_time = time.time()
        self.optimization_count += 1
        
        logger.info(f"[GLOBAL_OPT] Starting real optimization #{self.optimization_count} for keyframe {keyframe_idx}")
        
        try:
            # Initialize optimization keyframes if this is the first optimization
            if session_data.optimization_keyframes is None:
                logger.info(f"[GLOBAL_OPT] Initializing optimization keyframes from tracking keyframes")
                # Import here to avoid circular imports
                from .session_keyframes import SessionKeyframes
                import copy
                
                # Create new SessionKeyframes container (avoids RLock pickle issue)
                session_data.optimization_keyframes = SessionKeyframes(
                    h=session_data.keyframes.h,
                    w=session_data.keyframes.w,
                    buffer=session_data.keyframes.buffer,
                    dtype=session_data.keyframes.dtype,
                    device=session_data.keyframes.device
                )
                
                # Copy intrinsics if they exist
                if session_data.keyframes.K is not None:
                    session_data.optimization_keyframes.K = session_data.keyframes.K.clone()
                    logger.info(f"[GLOBAL_OPT] Copied camera intrinsics to optimization keyframes")
                
                # Safely copy each Frame individually (avoids RLock and tensor shape issues)
                for i in range(len(session_data.keyframes)):
                    original_frame = session_data.keyframes[i]
                    copied_frame = copy_frame_safely(original_frame)
                    session_data.optimization_keyframes.append(copied_frame)
                
                logger.info(f"[GLOBAL_OPT] Manually copied {len(session_data.optimization_keyframes)} keyframes for optimization")
            else:
                # Add the new keyframe to optimization set
                if keyframe_idx >= len(session_data.keyframes):
                    logger.error(f"Invalid keyframe index {keyframe_idx}, only {len(session_data.keyframes)} keyframes available")
                    return self._create_error_result("invalid_keyframe_index", start_time)
                
                new_keyframe = session_data.keyframes[keyframe_idx]
                copied_keyframe = copy_frame_safely(new_keyframe)
                session_data.optimization_keyframes.append(copied_keyframe)
                logger.info(f"[GLOBAL_OPT] Added keyframe {keyframe_idx} to optimization set (now {len(session_data.optimization_keyframes)} total)")
            
            # Initialize factor graph if this is the first optimization
            if self.factor_graph is None:
                self._initialize_factor_graph(session_data)
            
            # Get the frame that triggered optimization from optimization keyframes
            if keyframe_idx >= len(session_data.optimization_keyframes):
                logger.error(f"Invalid keyframe index {keyframe_idx}, only {len(session_data.optimization_keyframes)} optimization keyframes available")
                return self._create_error_result("invalid_optimization_keyframe_index", start_time)
            
            frame = session_data.optimization_keyframes[keyframe_idx]
            
            # Build list of keyframes to optimize against
            kf_idx = []
            
            # Add previous consecutive keyframes (local window)
            n_consec = min(config["local_opt"].get("window_size", 1), keyframe_idx)
            for j in range(min(n_consec, keyframe_idx)):
                kf_idx.append(keyframe_idx - 1 - j)
            
            # Query retrieval database for loop closures
            retrieval_inds = []
            if self.retrieval_database is not None:
                try:
                    retrieval_inds = self.retrieval_database.update(
                        frame,
                        add_after_query=True,
                        k=config["retrieval"]["k"],
                        min_thresh=config["retrieval"]["min_thresh"]
                    )
                    kf_idx += retrieval_inds
                    
                    # Log loop closures
                    lc_inds = set(retrieval_inds)
                    lc_inds.discard(keyframe_idx - 1)  # Remove consecutive frame
                    if len(lc_inds) > 0:
                        logger.info(f"[LOOP_CLOSURE] Detected loop closures for keyframe {keyframe_idx}: {lc_inds}")
                        
                except Exception as e:
                    logger.warning(f"Retrieval database query failed: {e}")
            
            # Remove duplicates and current keyframe
            kf_idx = list(set(kf_idx))
            if keyframe_idx in kf_idx:
                kf_idx.remove(keyframe_idx)
            
            # Add factors (constraints) between keyframes
            factors_added = False
            if kf_idx:
                try:
                    frame_idx = [keyframe_idx] * len(kf_idx)
                    factors_added = self.factor_graph.add_factors(
                        kf_idx, 
                        frame_idx, 
                        config["local_opt"]["min_match_frac"]
                    )
                    
                    if factors_added:
                        logger.info(f"[FACTORS] Added constraints between keyframe {keyframe_idx} and keyframes {kf_idx}")
                    else:
                        logger.warning(f"[FACTORS] Failed to add sufficient constraints for keyframe {keyframe_idx}")
                        
                except Exception as e:
                    logger.error(f"Failed to add factors: {e}")
                    return self._create_error_result(f"factor_addition_failed: {str(e)}", start_time)
            
            # Perform bundle adjustment if we have constraints
            if factors_added:
                try:
                    optimization_start = time.time()
                    
                    # Choose optimization method based on calibration
                    if config.get("use_calib", False):
                        self.factor_graph.solve_GN_calib()
                        method = "solve_GN_calib"
                    else:
                        self.factor_graph.solve_GN_rays()
                        method = "solve_GN_rays"
                    
                    optimization_time = time.time() - optimization_start
                    logger.info(f"[BUNDLE_ADJUSTMENT] Completed {method} in {optimization_time:.3f}s")
                    
                except Exception as e:
                    logger.error(f"Bundle adjustment failed: {e}")
                    return self._create_error_result(f"bundle_adjustment_failed: {str(e)}", start_time)
            else:
                logger.info(f"[SKIP] No factors added for keyframe {keyframe_idx}, skipping bundle adjustment")
            
            # Calculate timing
            end_time = time.time()
            duration = end_time - start_time
            self.total_optimization_time += duration
            
            # Calculate time since last optimization
            time_since_last = None
            if self.last_optimization_time is not None:
                time_since_last = start_time - self.last_optimization_time
            self.last_optimization_time = start_time
            
            # Create success result
            result = {
                "session_id": getattr(session_data, 'session_id', 'unknown'),
                "keyframe_idx": keyframe_idx,
                "optimization_count": self.optimization_count,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "time_since_last": time_since_last,
                "status": "optimization_completed",
                "factors_added": factors_added,
                "keyframes_optimized_against": kf_idx,
                "loop_closures_detected": len(retrieval_inds) if retrieval_inds else 0,
                "total_keyframes": len(session_data.keyframes)
            }
            
            logger.info(
                f"[GLOBAL_OPT] Completed #{self.optimization_count} for keyframe {keyframe_idx} "
                f"in {duration:.3f}s (factors: {factors_added}, loop_closures: {len(retrieval_inds) if retrieval_inds else 0})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Global optimization failed for keyframe {keyframe_idx}: {e}", exc_info=True)
            return self._create_error_result(f"optimization_failed: {str(e)}", start_time)
    
    def _initialize_factor_graph(self, session_data):
        """Initialize the factor graph for this session using optimization keyframes"""
        try:
            # Get camera intrinsics if available
            K = getattr(session_data, 'K', None)
            
            # Log tensor shapes of all keyframes before passing to FactorGraph
            logger.info(f"[TENSOR_DEBUG] Initializing FactorGraph with {len(session_data.optimization_keyframes)} keyframes")
            for i in range(len(session_data.optimization_keyframes)):
                frame = session_data.optimization_keyframes[i]
                pose_shape = frame.T_WC.data.shape
                pose_data_sample = frame.T_WC.data.flatten()[:3]  # Show first 3 values
                logger.info(f"[TENSOR_DEBUG] FactorGraph keyframe {i} (frame_id={frame.frame_id}) pose shape: {pose_shape}, data sample: {pose_data_sample}")
            
            # Create factor graph using optimization keyframes
            self.factor_graph = FactorGraph(
                self.model, 
                session_data.optimization_keyframes, 
                K=K, 
                device=self.device
            )
            
            logger.info("Factor graph initialized successfully with optimization keyframes")
            
        except Exception as e:
            logger.error(f"Failed to initialize factor graph: {e}")
            raise e
    
    def _initialize_factor_graph_sync(self, session_data):
        """Initialize the factor graph for synchronous operation using shared keyframes"""
        try:
            # Get camera intrinsics if available
            K = getattr(session_data, 'K', None)
            
            # Log tensor shapes of all keyframes before passing to FactorGraph
            logger.info(f"[TENSOR_DEBUG] Initializing FactorGraph with {len(session_data.keyframes)} shared keyframes")
            for i in range(len(session_data.keyframes)):
                frame = session_data.keyframes[i]
                pose_shape = frame.T_WC.data.shape
                pose_data_sample = frame.T_WC.data.flatten()[:3]  # Show first 3 values
                logger.info(f"[TENSOR_DEBUG] FactorGraph keyframe {i} (frame_id={frame.frame_id}) pose shape: {pose_shape}, data sample: {pose_data_sample}")
            
            # Create factor graph using shared keyframes directly
            self.factor_graph = FactorGraph(
                self.model, 
                session_data.keyframes, 
                K=K, 
                device=self.device
            )
            
            logger.info("Factor graph initialized successfully with shared keyframes")
            
        except Exception as e:
            logger.error(f"Failed to initialize factor graph: {e}")
            raise e
    
    def _create_error_result(self, error_msg: str, start_time: float) -> Dict[str, Any]:
        """Create an error result dictionary"""
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "keyframe_idx": -1,
            "optimization_count": self.optimization_count,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "status": "optimization_failed",
            "error": error_msg,
            "factors_added": False,
            "keyframes_optimized_against": [],
            "loop_closures_detected": 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get global optimization statistics"""
        avg_duration = (
            self.total_optimization_time / self.optimization_count 
            if self.optimization_count > 0 else 0.0
        )
        
        return {
            "optimization_count": self.optimization_count,
            "total_optimization_time": self.total_optimization_time,
            "average_duration": avg_duration,
            "last_optimization_time": self.last_optimization_time,
            "status": "real_optimization_active",
            "has_retrieval_database": self.retrieval_database is not None,
            "has_factor_graph": self.factor_graph is not None
        }
    
    def reset_stats(self) -> None:
        """Reset optimization statistics"""
        logger.info("Resetting global optimization statistics")
        self.optimization_count = 0
        self.total_optimization_time = 0.0
        self.last_optimization_time = None


# Async wrapper functions for compatibility with existing code
async def trigger_global_optimization_async(session_data, keyframe_idx: int) -> Dict[str, Any]:
    """
    Asynchronous function to trigger real global optimization
    
    Args:
        session_data: Session data containing global optimizer and keyframes
        keyframe_idx: Index of the keyframe that triggered optimization
        
    Returns:
        Dictionary with optimization results
    """
    from datetime import datetime
    
    # Print start message to verify asyncio task is running
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] 🔧 BUNDLE_ADJUSTMENT: Starting async optimization for keyframe {keyframe_idx}")
    
    if not hasattr(session_data, 'global_optimizer') or session_data.global_optimizer is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] ❌ BUNDLE_ADJUSTMENT: No global optimizer found for keyframe {keyframe_idx}")
        logger.error("No global optimizer found in session data")
        return {
            "status": "optimization_failed",
            "error": "no_global_optimizer_in_session",
            "keyframe_idx": keyframe_idx
        }
    
    try:
        result = await session_data.global_optimizer.optimize_keyframe(session_data, keyframe_idx)
        
        # Print completion message
        if result.get("status") == "optimization_completed":
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            duration = result.get("duration", 0)
            factors_added = result.get("factors_added", False)
            loop_closures = result.get("loop_closures_detected", 0)
            
            print(f"[{timestamp}] ✅ BUNDLE_ADJUSTMENT: Completed async keyframe {keyframe_idx} in {duration:.3f}s (factors: {factors_added}, loop_closures: {loop_closures})")
        else:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            error = result.get("error", "unknown_error")
            print(f"[{timestamp}] ❌ BUNDLE_ADJUSTMENT: Failed async keyframe {keyframe_idx} - {error}")
        
        return result
        
    except Exception as e:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] ❌ BUNDLE_ADJUSTMENT: Exception in async keyframe {keyframe_idx} - {str(e)}")
        logger.error(f"Async global optimization failed: {e}")
        return {
            "status": "optimization_failed",
            "error": f"async_optimization_failed: {str(e)}",
            "keyframe_idx": keyframe_idx
        }


def get_global_optimization_stats(session_data=None) -> Dict[str, Any]:
    """Get global optimization statistics"""
    if session_data and hasattr(session_data, 'global_optimizer') and session_data.global_optimizer:
        return session_data.global_optimizer.get_stats()
    else:
        return {
            "status": "no_optimizer_available",
            "optimization_count": 0
        }


def reset_global_optimization_stats(session_data=None) -> None:
    """Reset global optimization statistics"""
    if session_data and hasattr(session_data, 'global_optimizer') and session_data.global_optimizer:
        session_data.global_optimizer.reset_stats()


# Legacy compatibility functions (for background tasks if needed)
def trigger_global_optimization(session_data, keyframe_idx: int, session_id: str = None, connection_manager = None) -> Dict[str, Any]:
    """
    Synchronous wrapper for global optimization (for background tasks)
    
    Note: This runs the async function in a new event loop.
    Prefer using trigger_global_optimization_async directly.
    """
    from datetime import datetime
    
    # Debug print to verify background task is running
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] 🔧 BUNDLE_ADJUSTMENT: Starting optimization for keyframe {keyframe_idx}")
    
    try:
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            trigger_global_optimization_async(session_data, keyframe_idx)
        )
        loop.close()
        
        # Print completion message directly to terminal
        if result.get("status") == "optimization_completed":
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            duration = result.get("duration", 0)
            factors_added = result.get("factors_added", False)
            loop_closures = result.get("loop_closures_detected", 0)
            
            print(f"[{timestamp}] ✅ BUNDLE_ADJUSTMENT: Completed keyframe {keyframe_idx} in {duration:.3f}s (factors: {factors_added}, loop_closures: {loop_closures})")
        else:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            error = result.get("error", "unknown_error")
            print(f"[{timestamp}] ❌ BUNDLE_ADJUSTMENT: Failed keyframe {keyframe_idx} - {error}")
        
        return result
    except Exception as e:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] ❌ BUNDLE_ADJUSTMENT: Exception for keyframe {keyframe_idx} - {str(e)}")
        logger.error(f"Synchronous global optimization failed: {e}")
        
        return {
            "status": "optimization_failed",
            "error": f"sync_wrapper_failed: {str(e)}",
            "keyframe_idx": keyframe_idx
        }


def add_global_optimization_task(background_tasks, session_id: str, keyframe_idx: int, session_data, connection_manager = None):
    """
    Add global optimization to FastAPI background tasks with completion callbacks
    
    Args:
        background_tasks: FastAPI BackgroundTasks instance
        session_id: Session identifier for logging and callbacks
        keyframe_idx: Index of the keyframe that triggered optimization
        session_data: Session data containing global optimizer and keyframes
        connection_manager: For sending completion messages to frontend
    """
    logger.info(f"[GLOBAL_OPT] Queuing real optimization for session {session_id}, keyframe {keyframe_idx}")
    background_tasks.add_task(trigger_global_optimization, session_data, keyframe_idx, session_id, connection_manager)
