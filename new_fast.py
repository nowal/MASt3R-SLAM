# new_fast.py
import os
import multiprocessing as mp

# Set multiprocessing start method to 'spawn' as early as possible.
# This is crucial for CUDA compatibility in subprocesses.
try:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method("spawn", force=True)
    print(f"INFO_NEW_FAST_TOP: MP start method successfully set/confirmed to: {mp.get_start_method()}")
except RuntimeError as e:
    print(f"WARNING_NEW_FAST_TOP: Could not programmatically set MP start method to 'spawn': {e}. Current method: {mp.get_start_method(allow_none=True)}. Ensure 'spawn' is used if CUDA is involved in subprocesses (e.g., via environment variable PYTHON_MULTIPROCESSING_START_METHOD=spawn).")


import asyncio
import uuid
import logging
import json
import base64
import cv2 # type: ignore
import numpy as np
import time
import torch # For type hints and checking cuda availability
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Deque
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FastAPI_SLAM_Manager")

active_slam_sessions: Dict[str, Dict[str, Any]] = {}
sessions_lock = mp.Lock() # Using mp.Lock for inter-process potential, though FastAPI itself is async

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SLAM Backend (FastAPI) starting up...")
    logger.info(f"FastAPI Lifespan: Effective multiprocessing start method: {mp.get_start_method(allow_none=True)}")
    yield
    logger.info("SLAM Backend (FastAPI) shutting down...")
    session_ids_to_cleanup = []
    with sessions_lock: # Ensure thread-safety if multiple admin calls could trigger cleanup
        session_ids_to_cleanup = list(active_slam_sessions.keys())

    for session_id in session_ids_to_cleanup:
        logger.info(f"Initiating shutdown for SLAM process of session: {session_id}")
        session_data = None
        with sessions_lock:
            session_data = active_slam_sessions.pop(session_id, None)

        if session_data and session_data.get("process") and session_data["process"].is_alive():
            try:
                logger.info(f"Sending termination sentinel to SLAM process for session: {session_id}")
                # Ensure queue exists and is not broken before putting
                if session_data.get("frame_queue"):
                    session_data["frame_queue"].put(None, timeout=2) # Sentinel for SLAM process
                # Wait for process to finish
                session_data["process"].join(timeout=10) # Increased timeout
                if session_data["process"].is_alive():
                    logger.warning(f"SLAM process {session_id} (PID {session_data['process'].pid}) did not terminate gracefully. Forcing.")
                    session_data["process"].terminate() # Force terminate
                    session_data["process"].join(timeout=5) # Wait for termination
                    if session_data["process"].is_alive():
                        logger.error(f"SLAM process {session_id} (PID {session_data['process'].pid}) failed to terminate even after force.")
                    else:
                        logger.info(f"SLAM process {session_id} (PID {session_data['process'].pid}) forcibly terminated.")
                else:
                    logger.info(f"SLAM process {session_id} (PID {session_data['process'].pid}) terminated gracefully.")
            except (mp.queues.Full, queue.Full): # Handle potential queue full during shutdown
                logger.warning(f"Frame queue full for session {session_id} during shutdown sentinel send. Forcing termination.")
                if session_data["process"].is_alive():
                    session_data["process"].terminate()
                    session_data["process"].join(timeout=5)
            except Exception as e:
                logger.error(f"Error during shutdown of SLAM process {session_id}: {e}", exc_info=True)
                # Ensure termination if any error occurs
                if session_data.get("process") and session_data["process"].is_alive():
                    session_data["process"].terminate()
                    session_data["process"].join(timeout=5)
            finally:
                # Clean up queues
                try:
                    if session_data.get("frame_queue"):
                        session_data["frame_queue"].close()
                        session_data["frame_queue"].join_thread()
                    if session_data.get("result_queue"):
                        session_data["result_queue"].close()
                        session_data["result_queue"].join_thread()
                except Exception as e_q:
                    logger.error(f"Error closing queues for session {session_id}: {e_q}", exc_info=True)
        elif session_data and session_data.get("process"):
            logger.info(f"SLAM process for session {session_id} was already finished or not alive.")
        else:
            logger.warning(f"Session {session_id} data or process not found during shutdown or process already dead.")
    logger.info("All SLAM sessions cleaned up.")


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost", "http://localhost:3000", "http://localhost:8000",
    "http://localhost:5173", "https://localhost", "https://api.takeshapehome.com"
] # Add your frontend origin if different
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def image_payload_to_numpy(frame_data_base64: str) -> Optional[np.ndarray]:
    """
    Convert base64 image data (from data URL) to a NumPy array (H, W, C) with RGB order, float32, range [0,1].
    """
    try:
        # Remove "data:image/jpeg;base64," or similar prefix if present
        if "," in frame_data_base64:
            frame_data_base64 = frame_data_base64.split(",", 1)[1]

        img_data = base64.b64decode(frame_data_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Decodes to BGR by default

        if img_bgr is None:
            logger.error("cv2.imdecode failed. Input data might not be a valid image.")
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        
        # Normalize to [0, 1] and ensure float32, C-contiguous
        img_normalized = img_rgb.astype(np.float32) / 255.0
        return np.ascontiguousarray(img_normalized)

    except binascii.Error as e_b64: # More specific error for base64 issues
        logger.error(f"Base64 decoding error: {e_b64}. Input data: {frame_data_base64[:100]}...")
        return None
    except Exception as e:
        logger.error(f"image_payload_to_numpy error: {e}", exc_info=True)
        return None


@app.post("/connect")
async def handle_connect():
    """Generates a unique session ID for the client to use for the WebSocket connection."""
    session_id = str(uuid.uuid4())
    logger.info(f"Generated session ID: {session_id} for new connection request.")
    return {"sessionId": session_id, "message": f"Connect WebSocket to /ws/{session_id} and send initial frame to start SLAM."}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WebSocket accepted for session: {session_id}")

    slam_process: Optional[mp.Process] = None
    frame_input_queue: Optional[mp.Queue] = None
    results_output_queue: Optional[mp.Queue] = None
    forwarder_task: Optional[asyncio.Task] = None
    
    first_frame_data_tuple: Optional[Tuple[float, np.ndarray]] = None
    h: Optional[int] = None
    w: Optional[int] = None

    try:
        # 1. Receive the FIRST frame to determine dimensions
        logger.info(f"[{session_id}] Waiting for the first frame to determine dimensions...")
        initial_raw_data = await websocket.receive_text()
        try:
            initial_message = json.loads(initial_raw_data)
        except json.JSONDecodeError:
            logger.warning(f"[{session_id}] First message is not valid JSON: {initial_raw_data[:100]}...")
            await websocket.send_json({"type": "error", "message": "Initial message must be valid JSON."})
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA); return

        if initial_message.get("type") == "FRAME":
            payload = initial_message.get("payload")
            timestamp_ms = initial_message.get("timestamp")
            if not (payload and isinstance(payload, str) and timestamp_ms and isinstance(timestamp_ms, (int, float))):
                logger.warning(f"[{session_id}] Invalid initial FRAME format: {initial_message}")
                await websocket.send_json({"type": "error", "message": "Invalid initial FRAME format."})
                await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA); return

            img_array_for_slam = image_payload_to_numpy(payload)
            if img_array_for_slam is not None and img_array_for_slam.ndim == 3: # Check if it's a color image
                h, w = img_array_for_slam.shape[:2]
                first_frame_data_tuple = (float(timestamp_ms), img_array_for_slam)
                logger.info(f"[{session_id}] First frame received. Dimensions: H={h}, W={w}. Timestamp: {timestamp_ms}")
            else:
                logger.warning(f"[{session_id}] Failed to convert first frame or invalid dimensions. Shape: {img_array_for_slam.shape if img_array_for_slam is not None else 'None'}")
                await websocket.send_json({"type": "error", "message": "Invalid initial image data or format."})
                await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA); return
        else:
            logger.warning(f"[{session_id}] First message was not of type FRAME: {initial_message.get('type')}")
            await websocket.send_json({"type": "error", "message": "First message must be of type FRAME."})
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA); return

        # 2. Initialize SLAM process now that we have h, w
        slam_config_file_path = "config/base.yaml" # Make sure this path is correct
        slam_process_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        slam_output_base_dir = "slam_outputs_rt" # Base directory for saving trajectories, reconstructions

        if not Path(slam_config_file_path).exists():
            logger.error(f"MASt3R SLAM configuration file not found at: {Path(slam_config_file_path).resolve()}.")
            await websocket.send_json({"type": "error", "message": "Server configuration error: SLAM config missing."})
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR); return

        frame_input_queue = mp.Queue(maxsize=60) # Frames from FastAPI to SLAM process
        results_output_queue = mp.Queue(maxsize=120) # Results from SLAM process to FastAPI

        try:
            from slam_process_runner import run_slam_from_queue_entrypoint # Ensure this is in PYTHONPATH
        except ImportError:
            logger.critical("ImportError: Cannot import 'run_slam_from_queue_entrypoint' from 'slam_process_runner'.", exc_info=True)
            await websocket.send_json({"type": "error", "message": "Server critical error: SLAM module missing."})
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR); return
        
        # Parameters for the SLAM process, now including dynamic h, w
        other_slam_params_for_subprocess = {
            "expected_frame_height": h,
            "expected_frame_width": w,
            "no_viz": True, # As requested
            "use_calib_from_params": False, # As requested, rely on config file's use_calib and K if any
            # Add other parameters from modified_main.py's argparse if needed, with defaults:
            "save_frequency": 5, # For Splatt3r if enabled in SLAM process
            "voxel_size": 0.01, # For final reconstruction
            "max_points": 1000000, # For final reconstruction
            "conf_threshold": 1.5, # For final reconstruction
            "normal_downsample": 1.0, # For final reconstruction
            # Booleans for controlling what SLAM process saves on exit:
            "save_reconstruction_on_exit": True,
            "save_trajectory_on_exit": True,
            "enable_splatt3r_feature_if_configured": True, # Let slam_process_runner check its own config
            "splatt3r_output_dir_base": "screenshots_splatt3r_rt" # Base for splatt3r outputs
        }

        slam_process_args = (
            frame_input_queue, results_output_queue, session_id,
            slam_config_file_path, slam_process_device, slam_output_base_dir,
            other_slam_params_for_subprocess
        )
        slam_process = mp.Process(target=run_slam_from_queue_entrypoint, args=slam_process_args)
        
        with sessions_lock:
            if session_id in active_slam_sessions: # Should ideally not happen if /connect is used once
                logger.warning(f"Session ID {session_id} collision. This should not happen if IDs are unique.");
                await websocket.send_json({"type": "error", "message": "Session ID collision."});
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION); return
            slam_process.start()
            logger.info(f"Started SLAM PID {slam_process.pid} for session {session_id} with H={h}, W={w} on {slam_process_device}.")
            active_slam_sessions[session_id] = {
                "process": slam_process, 
                "frame_queue": frame_input_queue, 
                "result_queue": results_output_queue, 
                "websocket": websocket, # Store websocket for potential direct messages if needed (careful with async)
                "start_time": time.time(),
                "dimensions": (h,w)
            }
        
        # Send the first frame to the SLAM process
        if first_frame_data_tuple and frame_input_queue:
            try:
                frame_input_queue.put(first_frame_data_tuple, timeout=1.0)
                logger.info(f"[{session_id}] Queued the first frame for SLAM process.")
            except (mp.queues.Full, queue.Full):
                 logger.error(f"[{session_id}] Frame input queue full when trying to send the first frame. SLAM process might be stuck or slow to start.")
                 await websocket.send_json({"type": "error", "message": "Server busy, cannot process initial frame."})
                 # Consider a more robust cleanup here
                 await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER); return


        # 3. Task to forward results from SLAM process to WebSocket client
        async def forward_slam_results_to_client():
            logger.info(f"[{session_id}] SLAM result forwarder task started.")
            try:
                while True:
                    if not slam_process.is_alive() and (results_output_queue is None or results_output_queue.empty()):
                        logger.info(f"[{session_id}] SLAM process ended and result queue empty. Stopping forwarder.")
                        break
                    try:
                        if results_output_queue:
                            slam_output_message = results_output_queue.get(timeout=0.1) # Non-blocking with timeout
                            if slam_output_message:
                                await websocket.send_json(slam_output_message)
                        else: # Should not happen if setup is correct
                            await asyncio.sleep(0.1)
                    except (mp.queues.Empty, queue.Empty):
                        await asyncio.sleep(0.01) # Wait a bit if queue is empty
                    except (WebSocketDisconnect, asyncio.CancelledError):
                        logger.info(f"[{session_id}] WebSocket disconnected or forwarder task cancelled. Stopping forwarder.")
                        break
                    except Exception as e_fwd:
                        logger.error(f"[{session_id}] Error in SLAM result forwarder: {e_fwd}", exc_info=True)
                        await asyncio.sleep(0.1) # Avoid busy loop on error
            except asyncio.CancelledError:
                logger.info(f"[{session_id}] SLAM result forwarder task was cancelled externally.")
            finally:
                logger.info(f"[{session_id}] SLAM result forwarder task finished.")

        forwarder_task = asyncio.create_task(forward_slam_results_to_client())

        # 4. Main loop for receiving subsequent frames
        frames_sent_to_slam_count = 1 # Already sent one frame
        while True:
            raw_data = await websocket.receive_text()
            try:
                message_from_client = json.loads(raw_data)
            except json.JSONDecodeError:
                logger.warning(f"[{session_id}] Non-JSON msg: {raw_data[:100]}...");
                await websocket.send_json({"type": "error", "message": "Invalid JSON."});
                continue

            msg_type = message_from_client.get("type")
            if msg_type == "FRAME":
                payload = message_from_client.get("payload")
                timestamp_ms = message_from_client.get("timestamp")
                if not (payload and isinstance(payload, str) and timestamp_ms and isinstance(timestamp_ms, (int, float))):
                    logger.warning(f"[{session_id}] Invalid FRAME: {message_from_client}");
                    await websocket.send_json({"type": "error", "message": "Invalid FRAME format."});
                    continue
                
                img_array_for_slam = image_payload_to_numpy(payload)
                if img_array_for_slam is not None:
                    try:
                        # Optional: Check if dimensions match the first frame, if strictness is needed
                        # current_h, current_w = img_array_for_slam.shape[:2]
                        # if current_h != h or current_w != w:
                        #     logger.warning(f"[{session_id}] Frame dimensions changed mid-stream! Expected {h}x{w}, got {current_h}x{current_w}. Skipping frame.")
                        #     await websocket.send_json({"type": "warning", "message": "Frame dimensions changed, please maintain consistent stream."})
                        #     continue
                            
                        frame_data_tuple: Tuple[float, np.ndarray] = (float(timestamp_ms), img_array_for_slam)
                        if frame_input_queue:
                            frame_input_queue.put(frame_data_tuple, timeout=0.5) # Timeout to prevent indefinite blocking
                            frames_sent_to_slam_count += 1
                            if frames_sent_to_slam_count % 60 == 0: # Log less frequently
                                logger.info(f"[{session_id}] Sent frame {frames_sent_to_slam_count} (ts: {timestamp_ms}) to SLAM.")
                        else: # Should not happen
                             logger.error(f"[{session_id}] Frame input queue is None. Cannot send frame.")
                    except (mp.queues.Full, queue.Full):
                        logger.warning(f"[{session_id}] Frame input queue to SLAM is full. Frame (ts: {timestamp_ms}) dropped.");
                        await websocket.send_json({"type": "warning", "message": "Server busy, frame dropped."})
                    except Exception as e_put:
                        logger.error(f"[{session_id}] Error queueing frame for SLAM: {e_put}", exc_info=True);
                        await websocket.send_json({"type": "error", "message": f"Server error queueing: {str(e_put)}"})
                else:
                    logger.warning(f"[{session_id}] Failed to convert frame (ts: {timestamp_ms}) to NumPy.");
                    await websocket.send_json({"type": "error", "message": "Invalid image data."})
            else:
                logger.warning(f"[{session_id}] Unknown msg type from client: '{msg_type}'");
                await websocket.send_json({"type": "error", "message": f"Unknown type: '{msg_type}'."})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}.")
    except asyncio.CancelledError: # If the websocket_endpoint task itself is cancelled
        logger.info(f"WebSocket task for session {session_id} cancelled.")
    except Exception as e_ws: # Catch-all for the WebSocket endpoint logic
        logger.error(f"[{session_id}] Unhandled WebSocket error: {e_ws}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "session_id": session_id, "message": f"Unhandled server error: {str(e_ws)}"})
        except Exception: pass # Ignore if can't send error back
    finally:
        logger.info(f"Cleaning up WebSocket session: {session_id}")
        
        if forwarder_task and not forwarder_task.done():
            logger.info(f"[{session_id}] Cancelling SLAM result forwarder task.")
            forwarder_task.cancel()
            try: await forwarder_task # Wait for cancellation to complete
            except asyncio.CancelledError: logger.info(f"[{session_id}] Forwarder task cancelled during cleanup.")
            except Exception as e_task_cancel: logger.error(f"[{session_id}] Error awaiting forwarder task cancellation: {e_task_cancel}", exc_info=True)
        
        session_to_cleanup_data = None
        with sessions_lock:
            session_to_cleanup_data = active_slam_sessions.pop(session_id, None)
        
        if session_to_cleanup_data:
            p_slam = session_to_cleanup_data.get("process")
            q_in_slam = session_to_cleanup_data.get("frame_queue")
            q_out_slam = session_to_cleanup_data.get("result_queue")
            
            logger.info(f"Terminating SLAM process (PID {p_slam.pid if p_slam and p_slam.pid else 'N/A'}) for session {session_id}.")
            if p_slam and p_slam.is_alive():
                try:
                    if q_in_slam: q_in_slam.put(None, timeout=1.0) # Send termination sentinel
                    p_slam.join(timeout=5) # Wait for SLAM process to finish
                    if p_slam.is_alive():
                        logger.warning(f"SLAM process {session_id} (PID {p_slam.pid}) did not terminate gracefully. Forcing.")
                        p_slam.terminate()
                        p_slam.join(timeout=3) # Wait for forced termination
                except Exception as e_proc_term_final:
                    logger.error(f"Error during SLAM process termination for session {session_id}: {e_proc_term_final}", exc_info=True)
                    if p_slam.is_alive(): # Ensure it's terminated if error occurred during graceful shutdown
                        p_slam.terminate()
                        p_slam.join(timeout=3)
            
            # Clean up queues
            if q_in_slam:
                try: q_in_slam.close(); q_in_slam.join_thread()
                except Exception as e_qin_close: logger.error(f"Error closing input queue for {session_id}: {e_qin_close}")
            if q_out_slam:
                try: q_out_slam.close(); q_out_slam.join_thread()
                except Exception as e_qout_close: logger.error(f"Error closing output queue for {session_id}: {e_qout_close}")
        else:
            logger.info(f"[{session_id}] No active SLAM session data found for cleanup, or already cleaned.")

        # Ensure WebSocket is closed
        if websocket.client_state != WebSocketDisconnect:
            try: await websocket.close(code=status.WS_1001_GOING_AWAY)
            except Exception: pass # Ignore errors on close
        logger.info(f"Cleanup for WebSocket session {session_id} complete.")


@app.get("/")
async def read_root():
    return {"message": "SLAM Backend (FastAPI with Multiprocessing for MaSt3R-SLAM) is running."}

@app.get("/active_sessions")
async def get_active_slam_processes_info():
    session_details_list = []
    count = 0
    with sessions_lock: # Ensure thread-safe access to shared dict
        count = len(active_slam_sessions)
        if not active_slam_sessions:
            return {"message": "No active SLAM sessions.", "active_sessions_count": 0, "sessions": []}

        for session_id, data in active_slam_sessions.items():
            pid_info = "N/A"; is_alive_info = False
            if data.get("process"):
                process_obj = data["process"]
                # Ensure process_obj is a Process instance before accessing pid
                pid_info = process_obj.pid if hasattr(process_obj, 'pid') and process_obj.pid is not None else "N/A"
                is_alive_info = process_obj.is_alive() if hasattr(process_obj, 'is_alive') else False
            
            frame_q_sz = "N/A"
            if data.get("frame_queue") and hasattr(data["frame_queue"], 'qsize'):
                try: frame_q_sz = data["frame_queue"].qsize()
                except NotImplementedError: frame_q_sz = "N/A (Not supported on this platform)" # macOS
                except Exception: frame_q_sz = "Error"


            result_q_sz = "N/A"
            if data.get("result_queue") and hasattr(data["result_queue"], 'qsize'):
                try: result_q_sz = data["result_queue"].qsize()
                except NotImplementedError: result_q_sz = "N/A (Not supported on this platform)"
                except Exception: result_q_sz = "Error"
                
            dim_info = data.get("dimensions", "N/A")

            session_details_list.append({
                "session_id": session_id,
                "pid": pid_info,
                "is_alive": is_alive_info,
                "dimensions_HxW": f"{dim_info[0]}x{dim_info[1]}" if isinstance(dim_info, tuple) and len(dim_info) == 2 else "N/A",
                "frame_q_approx_size": frame_q_sz,
                "result_q_approx_size": result_q_sz,
                "start_time_iso": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(data.get("start_time", 0))) if data.get("start_time") else "N/A"
            })
    return {"active_sessions_count": count, "sessions": session_details_list}

if __name__ == "__main__":
    import uvicorn
    import platform
    logger.info(f"Attempting to start Uvicorn server for SLAM Backend...")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Operating System: {platform.system()} {platform.release()}")
    logger.info(f"Multiprocessing start method: {mp.get_start_method(allow_none=True)}")
    logger.info(f"Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

    uvicorn.run(
        "new_fast:app", # Make sure 'new_fast' matches your filename
        host="0.0.0.0", 
        port=8000, 
        log_level="info", # Uvicorn's own log level
        reload=False # Important: Set reload=False for multiprocessing stability
    )