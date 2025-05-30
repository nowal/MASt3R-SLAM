# slam_process_runner.py
import datetime
from pathlib import Path
import sys
import time
import cv2 # type: ignore
import lietorch # type: ignore
import torch
import yaml
import numpy as np
import logging
import queue as std_queue # For type hinting if mp.Queue is not directly hintable
import multiprocessing as mp

# SLAM-specific imports (ensure these are in your Python path)
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.config import load_config, config as global_slam_config, set_global_config
import mast3r_slam.evaluate as eval
# Import resize_img to determine resized dimensions
from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame, Frame # Import Frame for type hints
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization # Keep for conditional viz
import os
import shutil

# Configure logging for this process
logger = logging.getLogger("SLAM_Process")
if not logger.hasHandlers(): # Avoid adding multiple handlers if this module is reloaded
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(process)d %(message)s", # Added process ID
        datefmt="%Y-%m-%d %H:%M:%S"
    )

os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def relocalization(frame: Frame, keyframes: SharedKeyframes, factor_graph: FactorGraph, retrieval_database):
    with keyframes.lock:
        kf_idx_to_match = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=global_slam_config["retrieval"]["k"],
            min_thresh=global_slam_config["retrieval"]["min_thresh"],
        )
        kf_idx_to_match.extend(list(retrieval_inds))
        successful_loop_closure = False

        if kf_idx_to_match:
            keyframes.append(frame) # Tentatively add current frame
            n_kf = len(keyframes)
            current_frame_temp_idx = n_kf - 1
            
            valid_kf_indices_to_match = [idx for idx in kf_idx_to_match if idx < current_frame_temp_idx]

            if not valid_kf_indices_to_match:
                logger.info(f"Relocalization: No valid prior keyframes from retrieval {kf_idx_to_match} for temp KF {current_frame_temp_idx}.")
                keyframes.pop_last() 
                return False

            logger.info(f"RELOCALIZING frame (temp idx {current_frame_temp_idx}) against keyframes {valid_kf_indices_to_match}")
            current_frame_indices_for_factors = [current_frame_temp_idx] * len(valid_kf_indices_to_match)

            if factor_graph.add_factors(
                current_frame_indices_for_factors, 
                valid_kf_indices_to_match,  
                global_slam_config["reloc"]["min_match_frac"],
                is_reloc=global_slam_config["reloc"]["strict"],
            ):
                retrieval_database.update( 
                    frame,
                    add_after_query=True,
                    k=global_slam_config["retrieval"]["k"],
                    min_thresh=global_slam_config["retrieval"]["min_thresh"],
                )
                logger.info("Success! Relocalized")
                successful_loop_closure = True
                if valid_kf_indices_to_match and valid_kf_indices_to_match[0] < len(keyframes.T_WC):
                    keyframes.T_WC[current_frame_temp_idx] = keyframes.T_WC[valid_kf_indices_to_match[0]].clone()
                else:
                    logger.warning(f"Reloc: Could not clone T_WC from index {valid_kf_indices_to_match[0] if valid_kf_indices_to_match else 'N/A'}. Pose not updated post-reloc.")
            else:
                keyframes.pop_last() 
                logger.info("Failed to relocalize (factor graph could not add factors).")
        else:
            logger.info("No retrieval candidates for relocalization.")

        if successful_loop_closure:
            if global_slam_config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg_dict, model, states: SharedStates, keyframes: SharedKeyframes, K_matrix):
    set_global_config(cfg_dict)
    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K_matrix, device)
    retrieval_database = load_retriever(model)

    while True:
        mode = states.get_mode()
        if mode == Mode.TERMINATED:
            logger.info("Backend: Termination signal received.")
            break
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue

        if mode == Mode.RELOC:
            frame_to_reloc = states.get_frame()
            if frame_to_reloc is not None:
                success = relocalization(frame_to_reloc, keyframes, factor_graph, retrieval_database)
                if success:
                    states.set_mode(Mode.TRACKING)
            else:
                logger.warning("Backend: Tried to relocalize but no current frame in states.")
            states.dequeue_reloc()
            continue

        idx_to_optimize = -1
        with states.lock:
            if states.global_optimizer_tasks:
                idx_to_optimize = states.global_optimizer_tasks[0]

        if idx_to_optimize == -1:
            time.sleep(0.01)
            continue
        
        if not (0 <= idx_to_optimize < len(keyframes)):
            logger.warning(f"Backend: Invalid keyframe index {idx_to_optimize} for optimization. Max KF index: {len(keyframes)-1}. Discarding task.")
            with states.lock:
                if states.global_optimizer_tasks and states.global_optimizer_tasks[0] == idx_to_optimize:
                    states.global_optimizer_tasks.pop(0)
            time.sleep(0.01)
            continue

        kf_for_opt = keyframes[idx_to_optimize]
        if kf_for_opt is None:
            logger.warning(f"Backend: Keyframe at index {idx_to_optimize} is None. Skipping optimization task.")
            with states.lock:
                if states.global_optimizer_tasks and states.global_optimizer_tasks[0] == idx_to_optimize:
                     states.global_optimizer_tasks.pop(0)
            time.sleep(0.01)
            continue
            
        kf_indices_to_link = []
        n_consec = global_slam_config.get("backend_n_consecutive_links", 1)
        for j in range(min(n_consec, idx_to_optimize)):
            kf_indices_to_link.append(idx_to_optimize - 1 - j)

        retrieval_inds_backend = retrieval_database.update(
            kf_for_opt,
            add_after_query=True, 
            k=global_slam_config["retrieval"]["k"],
            min_thresh=global_slam_config["retrieval"].get("min_thresh_backend", global_slam_config["retrieval"]["min_thresh"]),
        )
        kf_indices_to_link.extend(list(retrieval_inds_backend))

        unique_indices_to_link = set(kf_indices_to_link)
        unique_indices_to_link.discard(idx_to_optimize)
        valid_indices_to_link = [idx for idx in unique_indices_to_link if 0 <= idx < len(keyframes) and idx != idx_to_optimize]


        if valid_indices_to_link:
            current_kf_repeated_indices = [idx_to_optimize] * len(valid_indices_to_link)
            factor_graph.add_factors(
                current_kf_repeated_indices, valid_indices_to_link,
                global_slam_config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if global_slam_config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if states.global_optimizer_tasks and states.global_optimizer_tasks[0] == idx_to_optimize:
                states.global_optimizer_tasks.pop(0)
            else:
                logger.warning(f"Backend: Mismatch popping optimization task. Expected {idx_to_optimize}, found {states.global_optimizer_tasks[0] if states.global_optimizer_tasks else 'empty'}. List: {list(states.global_optimizer_tasks)}")
    logger.info("Backend: Process finished.")


def run_slam_from_queue_entrypoint(
    frame_input_queue: mp.Queue,
    results_output_queue: mp.Queue,
    session_id: str,
    slam_config_file_path: str,
    slam_process_device: str,
    slam_output_base_dir: str,
    other_slam_params: dict
):
    logger.info(f"[{session_id}] SLAM process starting. PID: {os.getpid()}. Device: {slam_process_device}")
    logger.info(f"[{session_id}] Config file: {slam_config_file_path}")
    logger.info(f"[{session_id}] Other params: {other_slam_params}")

    backend_process = None 
    viz_process = None     

    try:
        load_config(slam_config_file_path)
        cfg_dict_for_backend = global_slam_config.copy()
        set_global_config(global_slam_config)

        h_original = other_slam_params.get("expected_frame_height")
        w_original = other_slam_params.get("expected_frame_width")
        if h_original is None or w_original is None:
            logger.error(f"[{session_id}] Critical: Original frame dimensions (h,w) not provided.")
            results_output_queue.put({"type": "error", "session_id": session_id, "message": "SLAM init failed: Missing frame dimensions."})
            return
        
        model_input_img_size_cfg = global_slam_config.get("model_input_size", 512)
        # Create a dummy image with original HWC format (uint8 is common for images before float conversion)
        dummy_orig_img_for_resize = np.zeros((int(h_original), int(w_original), 3), dtype=np.uint8) 
        
        logger.info(f"[{session_id}] Determining resized dimensions from original {h_original}x{w_original} using model_input_size_cfg: {model_input_img_size_cfg}")
        # resize_img expects an image that could be float [0,1] or uint8 [0,255].
        # The dtype of dummy_orig_img_for_resize (uint8) is fine for resize_img to determine output shape.
        resized_info_dict = resize_img(dummy_orig_img_for_resize, model_input_img_size_cfg)
        
        # Frame.uimg is (H_resized, W_resized, C)
        # Frame.img (model input) is (C, H_resized, W_resized)
        # SharedKeyframes/States constructors expect (h, w) for their internal buffer allocations.
        # These h, w should correspond to the actual tensor dimensions they will store.
        h_resized, w_resized = resized_info_dict["unnormalized_img"].shape[:2]
        logger.info(f"[{session_id}] Resized dimensions for shared memory: H={h_resized}, W={w_resized}")
        
        no_viz = other_slam_params.get("no_viz", True)
        save_3d = other_slam_params.get("enable_splatt3r_feature_if_configured", False)
        if save_3d and not global_slam_config.get("splatt3r", {}).get("enabled", False):
            logger.info(f"[{session_id}] Splatt3r requested by client but not enabled in SLAM config '{slam_config_file_path}'. Disabling.")
            save_3d = False
            
        output_dir_for_splatt3r = Path(other_slam_params.get("splatt3r_output_dir_base", "screenshots_splatt3r_rt"))
        save_frequency = other_slam_params.get("save_frequency", 5)
        voxel_size_recon = other_slam_params.get("voxel_size", 0.01)
        max_points_recon = other_slam_params.get("max_points", 1000000)
        conf_threshold_recon = other_slam_params.get("conf_threshold", 1.5)
        save_reconstruction_on_exit = other_slam_params.get("save_reconstruction_on_exit", True)
        save_trajectory_on_exit = other_slam_params.get("save_trajectory_on_exit", True)

        current_mp_start_method = mp.get_start_method(allow_none=True)
        if current_mp_start_method != 'spawn':
            logger.warning(f"[{session_id}] MP start method is '{current_mp_start_method}', attempting to set to 'spawn'.")
            try:
                mp.set_start_method("spawn", force=True)
                logger.info(f"[{session_id}] MP start method successfully set to 'spawn'.")
            except RuntimeError as e_mp:
                logger.error(f"[{session_id}] Failed to set MP start method to 'spawn': {e_mp}. CUDA in subprocesses might fail if not already 'spawn'.")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)
        device = torch.device(slam_process_device)

        manager = mp.Manager()
        main2viz_q = new_queue(manager, no_viz)
        viz2main_q = new_queue(manager, no_viz)
        
        # Initialize SharedKeyframes and SharedStates with the RESIZED dimensions
        keyframes = SharedKeyframes(manager, h_resized, w_resized, device=device) 
        states = SharedStates(manager, h_resized, w_resized, device=device)       
        logger.info(f"[{session_id}] SharedKeyframes and SharedStates initialized with H={h_resized}, W={w_resized}.")

        splatt3r_client = None
        if save_3d:
            logger.info(f"[{session_id}] 3D reconstructions (Splatt3r) enabled. Output base: {output_dir_for_splatt3r}")
            try:
                from splatt3r_client import Splatt3rClient # type: ignore
                splatt3r_client = Splatt3rClient()
                logger.info(f"[{session_id}] Splatt3r client initialized.")
            except Exception as e_spl_client:
                logger.warning(f"[{session_id}] Failed to initialize Splatt3r client: {e_spl_client}. Disabling 3D saving for this session.")
                save_3d = False
        
        if not no_viz:
            viz_process = mp.Process(
                target=run_visualization,
                args=(global_slam_config.copy(), states, keyframes, main2viz_q, viz2main_q),
            )
            viz_process.start()
            logger.info(f"[{session_id}] Visualization process started (PID: {viz_process.pid}).")

        model = load_mast3r(device=device)
        model.share_memory()

        K_matrix = None # For camera intrinsics
        if global_slam_config["use_calib"]:
            # If K_matrix is supposed to be loaded/created based on config, that logic would go here.
            # Since user said no_calib, this path is less critical but we ensure config["use_calib"] is respected.
            if K_matrix is None: # If no K_matrix was formed (e.g. not provided by params and not loaded from file)
                logger.warning(f"[{session_id}] 'use_calib' is true in SLAM config, but no K_matrix formed. Forcing 'use_calib' to False for this session.")
                global_slam_config["use_calib"] = False # Override for this session's logic
            
            if K_matrix is not None and global_slam_config["use_calib"]: # Check again after potential override
                 keyframes.set_intrinsics(K_matrix)
        
        slam_session_output_dir = Path(slam_output_base_dir) / session_id
        slam_session_output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_trajectory_on_exit or save_reconstruction_on_exit:
            traj_file = slam_session_output_dir / f"{session_id}.txt"
            recon_file = slam_session_output_dir / f"{session_id}.ply"
            if traj_file.exists(): traj_file.unlink(missing_ok=True)
            if recon_file.exists(): recon_file.unlink(missing_ok=True)

        tracker = FrameTracker(model, keyframes, device)
        last_viz_msg = WindowMsg()

        backend_process = mp.Process(target=run_backend, args=(cfg_dict_for_backend, model, states, keyframes, K_matrix))
        backend_process.start()
        logger.info(f"[{session_id}] Backend optimizer process started (PID: {backend_process.pid}).")

        frame_idx_counter = 0
        fps_timer = time.time()
        
        retrieval_database_splatt3r = load_retriever(model) if save_3d else None
        added_to_database_splatt3r = set()
        splatt3r_frame_output_dir = output_dir_for_splatt3r / session_id / "frame_output"
        if save_3d and not splatt3r_frame_output_dir.exists():
            splatt3r_frame_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{session_id}] Entering main processing loop.")
        while True:
            current_mode = states.get_mode()
            add_new_kf_flag = False # Reset for each frame iteration

            if not no_viz and viz_process is not None:
                viz_msg = try_get_msg(viz2main_q)
                last_viz_msg = viz_msg if viz_msg is not None else last_viz_msg
                if last_viz_msg.is_terminated:
                    logger.info(f"[{session_id}] Termination signal from visualization window.")
                    states.set_mode(Mode.TERMINATED)
                if last_viz_msg.is_paused and not last_viz_msg.next: states.pause()
                if not last_viz_msg.is_paused: states.unpause()
            
            if current_mode == Mode.TERMINATED:
                logger.info(f"[{session_id}] Mode is TERMINATED. Exiting main loop.")
                break
            if states.is_paused():
                time.sleep(0.01)
                continue

            try:
                queued_item = frame_input_queue.get(timeout=0.1)
                if queued_item is None:
                    logger.info(f"[{session_id}] Received termination sentinel (None) from input queue.")
                    states.set_mode(Mode.TERMINATED)
                    break
                timestamp_ms, img_numpy_from_queue = queued_item
            except (std_queue.Empty, mp.queues.Empty):
                if states.get_mode() == Mode.TERMINATED: break # Check again after timeout
                time.sleep(0.005)
                continue
            
            T_WC_initial_guess = (
                lietorch.Sim3.Identity(1, device=device)
                if frame_idx_counter == 0 or states.get_frame() is None 
                else states.get_frame().T_WC
            )
            
            # model_input_img_size_cfg is already defined (e.g., 512)
            current_slam_frame_obj = create_frame(
                frame_idx_counter,
                img_numpy_from_queue,
                T_WC_initial_guess,
                img_size=model_input_img_size_cfg, # Integer model input size for resize_img
                device=device
            )

            if current_mode == Mode.INIT:
                X_init, C_init = mast3r_inference_mono(model, current_slam_frame_obj)
                current_slam_frame_obj.update_pointmap(X_init, C_init)
                keyframes.append(current_slam_frame_obj)
                states.queue_global_optimization(len(keyframes) - 1)
                states.set_mode(Mode.TRACKING)
                states.set_frame(current_slam_frame_obj)
                results_output_queue.put({"type": "status", "session_id": session_id, "mode": "INITIALIZED", "keyframe_count": len(keyframes)})
                add_new_kf_flag = True

            elif current_mode == Mode.TRACKING:
                add_new_kf_flag, match_info, try_reloc = tracker.track(current_slam_frame_obj)
                if try_reloc:
                    states.set_mode(Mode.RELOC)
                    results_output_queue.put({"type": "status", "session_id": session_id, "mode": "RELOCALIZING"})
                states.set_frame(current_slam_frame_obj)
                if current_slam_frame_obj.T_WC is not None:
                    try:
                        pose_np = current_slam_frame_obj.T_WC.matrix().cpu().numpy().tolist()
                        results_output_queue.put({"type": "pose_update", "session_id": session_id, "pose_matrix_wc": pose_np, "frame_idx": frame_idx_counter})
                    except Exception as e_pose: logger.warning(f"[{session_id}] Could not get pose matrix for frame {frame_idx_counter}: {e_pose}")
                
                if add_new_kf_flag: # If tracker decided to add this frame
                    keyframes.append(current_slam_frame_obj)

            elif current_mode == Mode.RELOC:
                # In RELOC mode, relocalization() inside run_backend handles adding the keyframe if successful.
                # The main loop here prepares the frame and queues the relocalization attempt.
                X, C = mast3r_inference_mono(model, current_slam_frame_obj)
                current_slam_frame_obj.update_pointmap(X, C)
                states.set_frame(current_slam_frame_obj) # Set for backend to pick up
                states.queue_reloc()
                # add_new_kf_flag remains False here; backend's relocalization handles KF addition

            else: 
                logger.error(f"[{session_id}] Invalid mode: {current_mode}. Terminating.")
                states.set_mode(Mode.TERMINATED)
                break

            if add_new_kf_flag: # True if KF added in INIT or by TRACKER
                if len(keyframes) > 0:
                    current_kf_idx_in_list = len(keyframes) - 1
                    states.queue_global_optimization(current_kf_idx_in_list)
                    results_output_queue.put({"type": "new_keyframe", "session_id": session_id, "keyframe_id": current_kf_idx_in_list, "keyframe_count": len(keyframes)})
                    
                    if save_3d and splatt3r_client is not None and retrieval_database_splatt3r is not None:
                        newly_added_keyframe = keyframes[current_kf_idx_in_list]
                        if current_kf_idx_in_list not in added_to_database_splatt3r:
                            retrieval_database_splatt3r.update(
                                newly_added_keyframe, add_after_query=True, k=5, min_thresh=0.0
                            )
                            added_to_database_splatt3r.add(current_kf_idx_in_list)
                            logger.info(f"[{session_id}] Added keyframe {current_kf_idx_in_list} to Splatt3r retrieval DB.")

                        if len(keyframes) > 0 and len(keyframes) % save_frequency == 0 :
                            match_idx_splatt3r = -1 
                            try:
                                if current_kf_idx_in_list > 0:
                                    match_idx_splatt3r = current_kf_idx_in_list -1 
                                    retrieved_indices = retrieval_database_splatt3r.update(
                                        newly_added_keyframe, add_after_query=False, k=current_kf_idx_in_list, min_thresh=0.0075
                                    )
                                    valid_retrieved = [idx for idx in retrieved_indices if idx != current_kf_idx_in_list and 0 <= idx < current_kf_idx_in_list]
                                    if valid_retrieved:
                                        match_idx_splatt3r = valid_retrieved[-1]
                                        logger.info(f"[{session_id}] Splatt3r: Matched KF {current_kf_idx_in_list} with KF {match_idx_splatt3r}.")
                                    else:
                                         logger.info(f"[{session_id}] Splatt3r: No suitable match for KF {current_kf_idx_in_list}, using previous KF {match_idx_splatt3r if match_idx_splatt3r >=0 else 'N/A'}.")
                                else: 
                                    logger.info(f"[{session_id}] Splatt3r: Not enough keyframes to form a pair for KF {current_kf_idx_in_list}.")
                                    # Continue to next part of the loop, don't skip FPS logging
                                    
                                if match_idx_splatt3r >= 0 : # Proceed only if a valid match_idx was found/set
                                    splatt3r_output_ply = splatt3r_frame_output_dir / f"kf_{current_kf_idx_in_list}_{match_idx_splatt3r}.ply"
                                    kf_current_for_splat = keyframes[current_kf_idx_in_list]
                                    kf_match_for_splat = keyframes[match_idx_splatt3r]
                                    current_img_splat_np = (kf_current_for_splat.uimg.cpu().numpy() * 255).astype('uint8')
                                    match_img_splat_np = (kf_match_for_splat.uimg.cpu().numpy() * 255).astype('uint8')
                                    temp_curr_img_path = f"temp_current_{session_id}_{current_kf_idx_in_list}.jpg"
                                    temp_match_img_path = f"temp_match_{session_id}_{match_idx_splatt3r}.jpg"
                                    cv2.imwrite(temp_curr_img_path, cv2.cvtColor(current_img_splat_np, cv2.COLOR_RGB2BGR))
                                    cv2.imwrite(temp_match_img_path, cv2.cvtColor(match_img_splat_np, cv2.COLOR_RGB2BGR))
                                    splatt3r_client.process_images(temp_curr_img_path, temp_match_img_path, str(splatt3r_output_ply))
                                    os.remove(temp_curr_img_path); os.remove(temp_match_img_path)
                                    logger.info(f"[{session_id}] Saved Splatt3r PLY to {splatt3r_output_ply}")
                                    results_output_queue.put({"type": "splatt3r_saved", "session_id": session_id, "path": str(splatt3r_output_ply)})
                                else:
                                    if current_kf_idx_in_list > 0: # Only log if a match was attempted
                                     logger.info(f"[{session_id}] Splatt3r: Invalid or no match index ({match_idx_splatt3r}), skipping save for KF {current_kf_idx_in_list}.")
                            except IndexError: 
                                logger.warning(f"[{session_id}] Splatt3r: Index error accessing keyframes for pairing (current_kf_idx: {current_kf_idx_in_list}). Skipping this pair.")
                            except Exception as e_splatt3r_save:
                                logger.error(f"[{session_id}] Splatt3r: Failed to save pair for KF {current_kf_idx_in_list}: {e_splatt3r_save}", exc_info=True)
                else:
                    logger.warning(f"[{session_id}] add_new_kf_flag was true, but keyframes list is empty. This is unexpected.")

            frame_idx_counter += 1
            if frame_idx_counter > 0 and frame_idx_counter % 30 == 0:
                current_time_fps = time.time()
                time_diff_fps = current_time_fps - fps_timer
                fps = 30.0 / time_diff_fps if time_diff_fps > 1e-9 else 0 # Avoid division by zero or tiny number
                fps_timer = current_time_fps
                logger.info(f"[{session_id}] Processed {frame_idx_counter} frames. Approx FPS: {fps:.2f}")
                results_output_queue.put({"type": "fps_update", "session_id": session_id, "fps": fps, "frames_processed": frame_idx_counter})

        logger.info(f"[{session_id}] Exited main processing loop. Processed {frame_idx_counter} frames.")
        results_output_queue.put({"type": "status", "session_id": session_id, "mode": "TERMINATING"})

        if len(keyframes) > 0:
            keyframe_timestamps_for_eval = [kf.frame_id for kf in keyframes.get_all_keyframes() if kf is not None and hasattr(kf, 'frame_id')]
            if not keyframe_timestamps_for_eval:
                 logger.warning(f"[{session_id}] No valid frame_ids found in keyframes for trajectory saving.")
            else:
                logger.info(f"[{session_id}] Using frame_id for trajectory timestamps (count: {len(keyframe_timestamps_for_eval)}). For accurate evaluation, modify Frame class to store real timestamps.")

            if save_trajectory_on_exit and keyframe_timestamps_for_eval:
                eval.save_traj(slam_session_output_dir, f"{session_id}.txt", keyframe_timestamps_for_eval, keyframes)
                logger.info(f"[{session_id}] Saved final trajectory to {slam_session_output_dir / f'{session_id}.txt'}")
                results_output_queue.put({"type": "trajectory_saved", "session_id": session_id, "path": str(slam_session_output_dir / f'{session_id}.txt')})

            if save_reconstruction_on_exit:
                eval.save_reconstruction(
                    slam_session_output_dir, f"{session_id}.ply", keyframes,
                    C_conf_threshold=conf_threshold_recon,
                    voxel_sz=voxel_size_recon, max_pts=max_points_recon
                )
                logger.info(f"[{session_id}] Saved final reconstruction to {slam_session_output_dir / f'{session_id}.ply'}")
                results_output_queue.put({"type": "reconstruction_saved", "session_id": session_id, "path": str(slam_session_output_dir / f'{session_id}.ply')})
        else:
            logger.info(f"[{session_id}] No keyframes generated, skipping final save of trajectory/reconstruction.")

        if splatt3r_client is not None and hasattr(splatt3r_client, 'stop_server') and callable(splatt3r_client.stop_server):
            try:
                logger.info(f"[{session_id}] Attempting to stop Splatt3r client/server component if applicable.")
                # splatt3r_client.stop_server() 
            except Exception as e_spl_stop:
                logger.warning(f"[{session_id}] Error stopping Splatt3r client: {e_spl_stop}")


    except Exception as e_main_slam:
        logger.error(f"[{session_id}] Unhandled exception in SLAM process main logic: {e_main_slam}", exc_info=True)
        results_output_queue.put({"type": "error", "session_id": session_id, "message": f"SLAM process critical error: {str(e_main_slam)}"})
    finally:
        logger.info(f"[{session_id}] SLAM process entering final cleanup. Ensuring child processes are joined.")
        if 'states' in locals() and states is not None :
            states.set_mode(Mode.TERMINATED) # Signal children to terminate

        if backend_process is not None: # Check if it was ever assigned
            if backend_process.is_alive():
                logger.info(f"[{session_id}] Waiting for backend process (PID: {backend_process.pid}) to terminate...")
                backend_process.join(timeout=10) # Give backend more time
                if backend_process.is_alive():
                    logger.warning(f"[{session_id}] Backend process (PID: {backend_process.pid}) did not terminate gracefully, forcing.")
                    backend_process.terminate(); backend_process.join(timeout=3)
            logger.info(f"[{session_id}] Backend process (PID: {backend_process.pid if backend_process.pid else 'N/A'}) cleanup attempt complete.")
        
        if viz_process is not None: # Check if it was ever assigned
            if viz_process.is_alive():
                logger.info(f"[{session_id}] Waiting for visualization process (PID: {viz_process.pid}) to terminate...")
                viz_process.join(timeout=5)
                if viz_process.is_alive():
                    logger.warning(f"[{session_id}] Visualization process (PID: {viz_process.pid}) did not terminate gracefully, forcing.")
                    viz_process.terminate(); viz_process.join(timeout=3)
            logger.info(f"[{session_id}] Visualization process (PID: {viz_process.pid if viz_process.pid else 'N/A'}) cleanup attempt complete.")
        
        logger.info(f"[{session_id}] SLAM process (PID: {os.getpid()}) cleanup complete. Exiting.")
        try:
            results_output_queue.put({"type": "status", "session_id": session_id, "mode": "SHUTDOWN_COMPLETE"}, timeout=1.0)
        except (std_queue.Full, mp.queues.Full): # Handle both types of queue full exceptions
            logger.warning(f"[{session_id}] Result queue full during final shutdown message. Client may not receive SHUTDOWN_COMPLETE.")
        except Exception as e_q_final: # Catch any other exception during the final put
            logger.error(f"[{session_id}] Error putting final SHUTDOWN_COMPLETE to queue: {e_q_final}")