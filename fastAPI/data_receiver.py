"""
Data Receiver for Binary WebSocket Messages

Handles the reception and processing of binary image data and JSON metadata
from WebSocket connections using the new protocol.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, Union
from fastapi import WebSocket, BackgroundTasks

from .connection_manager import connection_manager
from .image_processor import image_processor
from .slam_initializer import SLAMInitializer
from .session_keyframes import SessionKeyframes
from .frame_tracker_wrapper import FrameTrackerWrapper, create_frame_tracker_wrapper, validate_tracking_requirements
from .global_optimizer import add_global_optimization_task
from .relocalization import add_relocalization_task
from .frame_saver import frame_saver
from mast3r_slam.frame import Mode, Frame

logger = logging.getLogger("DataReceiver")

class DataReceiver:
    """Handles reception and processing of WebSocket messages"""
    
    def __init__(self):
        self.messages_processed = 0
        self.binary_messages_processed = 0
        self.json_messages_processed = 0
        
    async def handle_websocket_message(self, websocket: WebSocket, session_id: str, message: Union[str, bytes]) -> bool:
        """
        Handle incoming WebSocket message (JSON or binary)
        
        Args:
            websocket: WebSocket connection
            session_id: Session identifier
            message: Raw message data (string for JSON, bytes for binary)
            
        Returns:
            True if message was handled successfully, False otherwise
        """
        try:
            self.messages_processed += 1
            
            if isinstance(message, str):
                return await self._handle_json_message(session_id, message)
            elif isinstance(message, bytes):
                return await self._handle_binary_message(session_id, message)
            else:
                logger.warning(f"Session {session_id}: Received unexpected message type: {type(message)}")
                await connection_manager.send_error_message(session_id, f"Unexpected message type: {type(message)}")
                return False
                
        except Exception as e:
            logger.error(f"Error handling message for session {session_id}: {e}", exc_info=True)
            await connection_manager.send_error_message(session_id, f"Message handling error: {str(e)}")
            return False
            
    async def _handle_json_message(self, session_id: str, message_str: str) -> bool:
        """Handle JSON message (metadata)"""
        try:
            self.json_messages_processed += 1
            data = json.loads(message_str)
            
            message_type = data.get("type")
            
            if message_type == "FRAME_METADATA":
                return await self._handle_frame_metadata(session_id, data)
            else:
                logger.warning(f"Session {session_id}: Unknown JSON message type: {message_type}")
                await connection_manager.send_debug_message(
                    session_id, "warning", f"Unknown message type: {message_type}"
                )
                return False
                
        except json.JSONDecodeError as e:
            logger.error(f"Session {session_id}: JSON decode error: {e}")
            await connection_manager.send_error_message(session_id, f"Invalid JSON: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Session {session_id}: Error handling JSON message: {e}", exc_info=True)
            await connection_manager.send_error_message(session_id, f"JSON processing error: {str(e)}")
            return False
            
    async def _handle_binary_message(self, session_id: str, binary_data: bytes) -> bool:
        """Handle binary message (image data)"""
        try:
            self.binary_messages_processed += 1
            
            # Get the stored metadata for this binary message
            metadata = await connection_manager.get_and_clear_metadata(session_id)
            
            if not metadata:
                logger.warning(f"Session {session_id}: Received binary data without preceding metadata")
                await connection_manager.send_error_message(
                    session_id, "Received binary image data without metadata"
                )
                return False
                
            frame_id = metadata.get('frameClientIndex', -1)
            data_size = len(binary_data)
            
            await connection_manager.send_debug_message(
                session_id, "info", 
                f"Received binary image data for frame {frame_id} ({data_size} bytes)"
            )
            
            # Validate metadata
            is_valid, error_msg = image_processor.validate_metadata(metadata)
            if not is_valid:
                logger.error(f"Session {session_id}: Invalid metadata: {error_msg}")
                await connection_manager.send_error_message(session_id, f"Invalid metadata: {error_msg}")
                return False
                
            # Get session data to access current camera pose
            session_data = await connection_manager.get_session(session_id)
            current_pose = session_data.current_camera_pose if session_data else None
            
            # Process the binary image through SLAM pipeline with current pose
            processing_start = time.time()
            result = image_processor.process_binary_frame(binary_data, metadata, current_pose)
            
            if not result or not result.get('success', False):
                logger.error(f"Session {session_id}: Failed to process frame {frame_id}")
                await connection_manager.send_error_message(
                    session_id, f"Failed to process frame {frame_id}"
                )
                return False
            
            # Save the original decoded frame for debugging (before SLAM processing)
            try:
                # Get the original decoded image from the image processor
                decoded_img = image_processor.decode_binary_image(binary_data, metadata.get('format', 'webp'))
                if decoded_img is not None:
                    # Store the decoded frame in session data for comparisons
                    session_data = await connection_manager.get_session(session_id)
                    if session_data:
                        session_data.last_decoded_frame = decoded_img
                    
                    # Save the original decoded frame (this should show actual differences)
                    frame_saver.save_frame(
                        session_id, 
                        decoded_img, 
                        frame_type="original", 
                        metadata={"frame_id": frame_id}
                    )
            except Exception as save_error:
                logger.warning(f"Failed to save original frame {frame_id} for session {session_id}: {save_error}")
            
            # SLAM Processing: Initialize or track the frame
            try:
                frame = result.get('frame_object')
                if frame is not None:
                    session_data = await connection_manager.get_session(session_id)
                    if session_data is None:
                        logger.error(f"Session {session_id}: Session data not found")
                        result['slam_initialized'] = False
                        result['slam_status'] = 'session_not_found'
                    elif session_data.slam_initializer is None:
                        logger.error(f"Session {session_id}: SLAM initializer not found - session setup may have failed")
                        result['slam_initialized'] = False
                        result['slam_status'] = 'slam_initializer_missing'
                    else:
                        # Process frame based on current SLAM mode
                        slam_result = await self._process_slam_frame(session_data, frame, session_id)
                        result.update(slam_result)
                else:
                    logger.warning(f"Session {session_id}: No frame object in result for frame {frame_id}")
                    result['slam_initialized'] = False
                    result['slam_status'] = 'no_frame_object'
                    
            except Exception as e:
                logger.error(f"Session {session_id}: SLAM processing failed for frame {frame_id}: {e}", exc_info=True)
                await connection_manager.send_error_message(
                    session_id, f"SLAM processing failed: {str(e)}"
                )
                result['slam_initialized'] = False
                result['slam_status'] = f'error: {str(e)}'
            
            processing_time = time.time() - processing_start
            
            # Update session statistics
            await connection_manager.update_frame_stats(session_id)
            
            # Send processing results back to frontend
            await self._send_processing_results(session_id, result, processing_time, data_size)
            
            return result.get('success', False)
            
        except Exception as e:
            logger.error(f"Session {session_id}: Error handling binary message: {e}", exc_info=True)
            await connection_manager.send_error_message(session_id, f"Binary processing error: {str(e)}")
            return False
            
    async def _handle_frame_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """Handle frame metadata message"""
        try:
            frame_id = metadata.get('frameClientIndex', -1)
            format_type = metadata.get('format', 'unknown')
            
            # Validate metadata
            is_valid, error_msg = image_processor.validate_metadata(metadata)
            if not is_valid:
                logger.error(f"Session {session_id}: Invalid frame metadata: {error_msg}")
                await connection_manager.send_error_message(session_id, f"Invalid metadata: {error_msg}")
                return False
                
            # Store metadata for the next binary message
            await connection_manager.store_frame_metadata(session_id, metadata)
            
            await connection_manager.send_debug_message(
                session_id, "info", 
                f"Received metadata for frame {frame_id} (format: {format_type})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Session {session_id}: Error handling frame metadata: {e}", exc_info=True)
            await connection_manager.send_error_message(session_id, f"Metadata processing error: {str(e)}")
            return False
            
    async def _send_processing_results(self, session_id: str, result: Dict[str, Any], 
                                     processing_time: float, data_size: int):
        """Send processing results back to frontend"""
        try:
            frame_id = result.get('frame_id', -1)
            success = result.get('success', False)
            
            if success:
                # Send success message with detailed info
                times = result.get('processing_times', {})
                message = (
                    f"Frame {frame_id} processed successfully! "
                    f"Total: {processing_time:.3f}s "
                    f"(decode+resize: {times.get('resize_img', 0):.3f}s, "
                    f"create_frame: {times.get('create_frame', 0):.3f}s) "
                    f"Data: {data_size} bytes"
                )
                
                await connection_manager.send_debug_message(session_id, "success", message)
                
                # Send detailed processing info (convert numpy arrays to lists for JSON serialization)
                def safe_convert_shape(shape):
                    """Convert numpy array or tensor to list for JSON serialization"""
                    if shape is None:
                        return None
                    if hasattr(shape, 'tolist'):  # numpy array or tensor
                        return shape.tolist()
                    elif hasattr(shape, 'cpu'):  # torch tensor
                        return shape.cpu().tolist()
                    else:
                        return list(shape) if hasattr(shape, '__iter__') else shape
                
                processing_info = {
                    "type": "frame_processed",
                    "frame_id": frame_id,
                    "timestamp": result.get('timestamp'),
                    "success": True,
                    "processing_times": times,
                    "data_size": data_size,
                    "original_shape": safe_convert_shape(result.get('original_shape')),
                    "resized_shape": safe_convert_shape(result.get('resized_shape')),
                    "decoded_shape": safe_convert_shape(result.get('decoded_shape')),
                    "slam_initialized": result.get('slam_initialized', False),
                    "slam_status": result.get('slam_status', 'unknown')
                }
                
                await connection_manager.send_message(session_id, processing_info)
                
            else:
                # Send error message
                error = result.get('error', 'Unknown error')
                await connection_manager.send_debug_message(
                    session_id, "error", f"Frame {frame_id} processing failed: {error}"
                )
                
                error_info = {
                    "type": "frame_error",
                    "frame_id": frame_id,
                    "timestamp": result.get('timestamp'),
                    "success": False,
                    "error": error
                }
                
                await connection_manager.send_message(session_id, error_info)
                
        except Exception as e:
            logger.error(f"Session {session_id}: Error sending processing results: {e}", exc_info=True)
            
    async def _process_slam_frame(self, session_data, frame: Frame, session_id: str) -> Dict[str, Any]:
        """
        Process frame through SLAM pipeline based on current mode
        
        Args:
            session_data: SessionData object containing SLAM state
            frame: Frame object to process
            session_id: Session identifier for logging
            
        Returns:
            Dictionary with SLAM processing results
        """
        try:
            # Update session tracking stats
            session_data.tracking_stats["frames_processed"] += 1
            session_data.last_frame = frame
            
            # Update current camera pose for next frame (like original main.py)
            session_data.current_camera_pose = frame.T_WC
            
            if session_data.slam_mode == Mode.INIT:
                return await self._handle_initialization_mode(session_data, frame, session_id)
            elif session_data.slam_mode == Mode.TRACKING:
                return await self._handle_tracking_mode(session_data, frame, session_id)
            elif session_data.slam_mode == Mode.RELOC:
                return await self._handle_relocalization_mode(session_data, frame, session_id)
            else:
                logger.error(f"Session {session_id}: Unknown SLAM mode: {session_data.slam_mode}")
                return {
                    'slam_initialized': False,
                    'slam_status': f'unknown_mode_{session_data.slam_mode}'
                }
                
        except Exception as e:
            logger.error(f"Session {session_id}: Error in _process_slam_frame: {e}", exc_info=True)
            return {
                'slam_initialized': False,
                'slam_status': f'processing_error: {str(e)}'
            }
    
    async def _handle_initialization_mode(self, session_data, frame: Frame, session_id: str) -> Dict[str, Any]:
        """Handle frame processing in INIT mode"""
        try:
            slam_initializer = session_data.slam_initializer
            
            if not slam_initializer.is_initialized:
                # Initialize SLAM with the first frame
                logger.info(f"Session {session_id}: Initializing SLAM with frame {frame.frame_id}")
                enhanced_frame = await slam_initializer.initialize_frame(
                    frame, connection_manager, session_id
                )
                
                # Set up tracking components after successful initialization
                await self._setup_tracking_components(session_data, enhanced_frame, session_id)
                
                # Transition to tracking mode
                session_data.slam_mode = Mode.TRACKING
                logger.info(f"Session {session_id}: SLAM initialized, transitioning to TRACKING mode")
                
                return {
                    'slam_initialized': True,
                    'slam_status': 'initialized_and_ready_for_tracking',
                    'slam_mode': 'TRACKING',
                    'num_keyframes': len(session_data.keyframes) if session_data.keyframes else 0
                }
            else:
                # Already initialized - this shouldn't happen in INIT mode
                logger.warning(f"Session {session_id}: SLAM already initialized but still in INIT mode")
                session_data.slam_mode = Mode.TRACKING
                return {
                    'slam_initialized': True,
                    'slam_status': 'already_initialized_switching_to_tracking',
                    'slam_mode': 'TRACKING'
                }
                
        except Exception as e:
            logger.error(f"Session {session_id}: Initialization failed: {e}", exc_info=True)
            return {
                'slam_initialized': False,
                'slam_status': f'initialization_failed: {str(e)}'
            }
    
    async def _handle_tracking_mode(self, session_data, frame: Frame, session_id: str) -> Dict[str, Any]:
        """Handle frame processing in TRACKING mode"""
        try:
            # Validate tracking requirements
            if not session_data.keyframes or not session_data.frame_tracker_instance:
                logger.error(f"Session {session_id}: Missing tracking components")
                return {
                    'slam_initialized': False,
                    'slam_status': 'missing_tracking_components'
                }
            
            # Validate tracking requirements
            is_valid, error_msg = validate_tracking_requirements(session_data.keyframes, frame)
            if not is_valid:
                logger.error(f"Session {session_id}: Tracking validation failed: {error_msg}")
                # Trigger relocalization
                session_data.slam_mode = Mode.RELOC
                session_data.tracking_stats["tracking_failures"] += 1
                return {
                    'slam_initialized': True,
                    'slam_status': f'tracking_validation_failed_switching_to_reloc: {error_msg}',
                    'slam_mode': 'RELOC'
                }
            
            # Perform tracking
            add_new_kf, try_reloc, tracking_info = session_data.frame_tracker.track_frame(
                session_id, frame, session_data.keyframes, session_data.frame_tracker_instance
            )
            
            # Save frame comparison for debugging (original decoded images)
            try:
                # Get the current frame's original decoded image
                session_data = await connection_manager.get_session(session_id)
                if session_data and hasattr(session_data, 'last_decoded_frame'):
                    current_decoded = session_data.last_decoded_frame
                    
                    # Get the keyframe's original image (if we stored it)
                    last_keyframe = session_data.keyframes.last_keyframe()
                    if last_keyframe is not None and hasattr(last_keyframe, 'original_img'):
                        keyframe_img = last_keyframe.original_img
                    else:
                        # Fallback to processed keyframe image
                        keyframe_img = last_keyframe.img if last_keyframe else None
                    
                    if keyframe_img is not None and current_decoded is not None:
                        frame_saver.save_frame_comparison(
                            session_id,
                            keyframe_img,
                            current_decoded,
                            label1="keyframe",
                            label2="current",
                            frame_id=frame.frame_id
                        )
            except Exception as save_error:
                logger.warning(f"Failed to save frame comparison for frame {frame.frame_id}: {save_error}")
            
            if try_reloc:
                # Tracking failed - switch to relocalization mode
                session_data.slam_mode = Mode.RELOC
                session_data.tracking_stats["tracking_failures"] += 1
                
                # Queue dummy relocalization task
                from fastapi import BackgroundTasks
                background_tasks = BackgroundTasks()
                add_relocalization_task(background_tasks, session_id, frame, "tracking_failure")
                
                logger.warning(f"Session {session_id}: Tracking failed, switching to RELOC mode")
                return {
                    'slam_initialized': True,
                    'slam_status': 'tracking_failed_switching_to_reloc',
                    'slam_mode': 'RELOC',
                    'tracking_info': tracking_info
                }
            
            if add_new_kf:
                # Add frame as new keyframe
                session_data.keyframes.append(frame)
                session_data.tracking_stats["keyframes_added"] += 1
                
                # Queue real global optimization as asyncio task
                keyframe_idx = len(session_data.keyframes) - 1
                import asyncio
                from .global_optimizer import trigger_global_optimization_async
                
                # Create asyncio task that runs in background
                asyncio.create_task(
                    trigger_global_optimization_async(session_data, keyframe_idx)
                )
                logger.info(f"[GLOBAL_OPT] Queued asyncio task for session {session_id}, keyframe {keyframe_idx}")
                
                logger.info(f"Session {session_id}: New keyframe added ({len(session_data.keyframes)} total)")
                return {
                    'slam_initialized': True,
                    'slam_status': 'tracking_success_new_keyframe_added',
                    'slam_mode': 'TRACKING',
                    'num_keyframes': len(session_data.keyframes),
                    'tracking_info': tracking_info
                }
            else:
                # Successful tracking without new keyframe
                logger.info(f"Session {session_id}: Frame tracked successfully (no new keyframe)")
                return {
                    'slam_initialized': True,
                    'slam_status': 'tracking_success_no_new_keyframe',
                    'slam_mode': 'TRACKING',
                    'num_keyframes': len(session_data.keyframes),
                    'tracking_info': tracking_info
                }
                
        except Exception as e:
            logger.error(f"Session {session_id}: Tracking mode error: {e}", exc_info=True)
            # Switch to relocalization on error
            session_data.slam_mode = Mode.RELOC
            session_data.tracking_stats["tracking_failures"] += 1
            return {
                'slam_initialized': True,
                'slam_status': f'tracking_error_switching_to_reloc: {str(e)}',
                'slam_mode': 'RELOC'
            }
    
    async def _handle_relocalization_mode(self, session_data, frame: Frame, session_id: str) -> Dict[str, Any]:
        """Handle frame processing in RELOC mode"""
        try:
            session_data.tracking_stats["relocalization_attempts"] += 1
            
            # Queue dummy relocalization task
            from fastapi import BackgroundTasks
            background_tasks = BackgroundTasks()
            add_relocalization_task(background_tasks, session_id, frame, "relocalization_mode")
            
            # For dummy implementation, randomly succeed or stay in reloc mode
            import random
            if random.random() > 0.7:  # 30% chance to succeed and return to tracking
                session_data.slam_mode = Mode.TRACKING
                logger.info(f"Session {session_id}: Dummy relocalization succeeded, returning to TRACKING")
                return {
                    'slam_initialized': True,
                    'slam_status': 'relocalization_success_returning_to_tracking',
                    'slam_mode': 'TRACKING'
                }
            else:
                logger.info(f"Session {session_id}: Dummy relocalization continuing...")
                return {
                    'slam_initialized': True,
                    'slam_status': 'relocalization_in_progress',
                    'slam_mode': 'RELOC'
                }
                
        except Exception as e:
            logger.error(f"Session {session_id}: Relocalization mode error: {e}", exc_info=True)
            return {
                'slam_initialized': True,
                'slam_status': f'relocalization_error: {str(e)}',
                'slam_mode': 'RELOC'
            }
    
    async def _setup_tracking_components(self, session_data, initialized_frame: Frame, session_id: str):
        """Set up tracking components after successful initialization"""
        try:
            # Get image dimensions from the initialized frame
            h, w = initialized_frame.img.shape[-2:]
            device = initialized_frame.img.device.type if hasattr(initialized_frame.img, 'device') else "cuda:0"
            
            # Create SessionKeyframes
            session_data.keyframes = SessionKeyframes(h, w, device=device)
            
            # Add the initialized frame as the first keyframe
            session_data.keyframes.append(initialized_frame)
            
            # Create FrameTrackerWrapper
            model = session_data.slam_initializer.model
            session_data.frame_tracker = create_frame_tracker_wrapper(model, device)
            
            # Create the actual FrameTracker instance
            session_data.frame_tracker_instance = session_data.frame_tracker.create_tracker_for_session(
                session_data.keyframes
            )
            
            logger.info(f"Session {session_id}: Tracking components set up successfully")
            
        except Exception as e:
            logger.error(f"Session {session_id}: Failed to setup tracking components: {e}", exc_info=True)
            raise e

    def get_stats(self) -> Dict[str, Any]:
        """Get data receiver statistics"""
        return {
            "messages_processed": self.messages_processed,
            "json_messages_processed": self.json_messages_processed,
            "binary_messages_processed": self.binary_messages_processed,
            "image_processor_stats": image_processor.get_stats()
        }

# Global data receiver instance
data_receiver = DataReceiver()
