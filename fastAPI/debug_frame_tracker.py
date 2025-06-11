"""
Debug Frame Tracker - Enhanced logging for keyframe selection debugging

This module provides a debug wrapper around the original FrameTracker that logs
the exact keyframe selection calculations to help diagnose why keyframes aren't
being added.
"""

import logging
import torch
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.frame import Frame
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric
from mast3r_slam.config import config

logger = logging.getLogger("DebugFrameTracker")


class DebugFrameTracker(FrameTracker):
    """
    Enhanced FrameTracker with detailed keyframe selection logging
    
    This class inherits from the original FrameTracker and adds comprehensive
    logging to the keyframe selection logic to help debug why keyframes aren't
    being added when they should be.
    """
    
    def __init__(self, model, frames, device):
        super().__init__(model, frames, device)
        logger.info(f"DebugFrameTracker initialized with config: {self.cfg}")
    
    def track(self, frame: Frame):
        """
        Enhanced track method with detailed keyframe selection logging
        """
        keyframe = self.keyframes.last_keyframe()
        
        logger.info(f"[DEBUG_TRACK] Frame {frame.frame_id}: Starting tracking against keyframe {keyframe.frame_id}")
        logger.info(f"[DEBUG_TRACK] Frame {frame.frame_id}: Config match_frac_thresh = {self.cfg['match_frac_thresh']}")
        logger.info(f"[DEBUG_TRACK] Frame {frame.frame_id}: Config min_match_frac = {self.cfg['min_match_frac']}")

        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        )
        # Save idx for next
        self.idx_f2k = idx_f2k.clone()

        # Get rid of batch dim
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # Update keyframe pointmap after registration (need pose)
        frame.update_pointmap(Xff, Cff)

        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K
        else:
            K = None

        # Get poses and point correspondneces and confidences
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # Get valid
        # Use canonical confidence average
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf = valid_match_k & valid_Q

        match_frac = valid_opt.sum() / valid_opt.numel()
        
        # Enhanced logging for match fraction check
        logger.info(f"[DEBUG_TRACK] Frame {frame.frame_id}: Initial match_frac = {match_frac:.6f}")
        logger.info(f"[DEBUG_TRACK] Frame {frame.frame_id}: valid_opt.sum() = {valid_opt.sum()}")
        logger.info(f"[DEBUG_TRACK] Frame {frame.frame_id}: valid_opt.numel() = {valid_opt.numel()}")
        logger.info(f"[DEBUG_TRACK] Frame {frame.frame_id}: Checking if {match_frac:.6f} < {self.cfg['min_match_frac']}")
        
        if match_frac < self.cfg["min_match_frac"]:
            logger.warning(f"[DEBUG_TRACK] Frame {frame.frame_id}: SKIPPED - match_frac {match_frac:.6f} < min_match_frac {self.cfg['min_match_frac']}")
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True

        try:
            # Track
            if not use_calib:
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                )
        except Exception as e:
            logger.error(f"[DEBUG_TRACK] Frame {frame.frame_id}: Cholesky failed: {e}")
            print(f"Cholesky failed {frame.frame_id}")
            return False, [], True

        frame.T_WC = T_WCf

        # Use pose to transform points to update keyframe
        Xkk = T_CkCf.act(Xkf)
        keyframe.update_pointmap(Xkk, Ckf)
        # write back the fitered pointmap
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # ===== ENHANCED KEYFRAME SELECTION LOGGING =====
        logger.info(f"[DEBUG_KEYFRAME_SELECTION] Frame {frame.frame_id}: Starting keyframe selection logic")
        
        # Keyframe selection - with detailed logging
        n_valid = valid_kf.sum()
        total_points = valid_kf.numel()
        match_frac_k = n_valid / total_points
        
        logger.info(f"[DEBUG_KEYFRAME_SELECTION] Frame {frame.frame_id}: n_valid = {n_valid}")
        logger.info(f"[DEBUG_KEYFRAME_SELECTION] Frame {frame.frame_id}: total_points = {total_points}")
        logger.info(f"[DEBUG_KEYFRAME_SELECTION] Frame {frame.frame_id}: match_frac_k = {match_frac_k:.6f}")
        
        # Calculate unique fraction
        unique_indices = torch.unique(idx_f2k[valid_match_k[:, 0]])
        unique_count = unique_indices.shape[0]
        unique_frac_f = unique_count / total_points
        
        logger.info(f"[DEBUG_KEYFRAME_SELECTION] Frame {frame.frame_id}: unique_indices.shape[0] = {unique_count}")
        logger.info(f"[DEBUG_KEYFRAME_SELECTION] Frame {frame.frame_id}: unique_frac_f = {unique_frac_f:.6f}")
        
        # The critical decision
        min_frac = min(match_frac_k, unique_frac_f)
        threshold = self.cfg["match_frac_thresh"]
        new_kf = min_frac < threshold
        
        logger.info(f"[DEBUG_KEYFRAME_SELECTION] Frame {frame.frame_id}: min(match_frac_k, unique_frac_f) = min({match_frac_k:.6f}, {unique_frac_f:.6f}) = {min_frac:.6f}")
        logger.info(f"[DEBUG_KEYFRAME_SELECTION] Frame {frame.frame_id}: threshold = {threshold}")
        logger.info(f"[DEBUG_KEYFRAME_SELECTION] Frame {frame.frame_id}: new_kf = {min_frac:.6f} < {threshold} = {new_kf}")
        
        if new_kf:
            logger.info(f"[DEBUG_KEYFRAME_SELECTION] Frame {frame.frame_id}: ✅ NEW KEYFRAME NEEDED!")
        else:
            logger.info(f"[DEBUG_KEYFRAME_SELECTION] Frame {frame.frame_id}: ❌ No new keyframe needed")
            logger.info(f"[DEBUG_KEYFRAME_SELECTION] Frame {frame.frame_id}: Match quality too high: {min_frac:.6f} >= {threshold}")

        # Rest idx if new keyframe
        if new_kf:
            self.reset_idx_f2k()

        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )


def create_debug_frame_tracker(model, keyframes, device):
    """
    Factory function to create a DebugFrameTracker
    
    Args:
        model: MASt3R model for tracking
        keyframes: Keyframes object (SessionKeyframes or SharedKeyframes)
        device: Device for computations
        
    Returns:
        DebugFrameTracker instance
    """
    return DebugFrameTracker(model, keyframes, device)
