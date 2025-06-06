"""
Data Receiver for Binary WebSocket Messages

Handles the reception and processing of binary image data and JSON metadata
from WebSocket connections using the new protocol.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, Union
from fastapi import WebSocket

from .connection_manager import connection_manager
from .image_processor import image_processor
from .slam_initializer import SLAMInitializer

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
                
            # Process the binary image through SLAM pipeline
            processing_start = time.time()
            result = image_processor.process_binary_frame(binary_data, metadata)
            
            if not result or not result.get('success', False):
                logger.error(f"Session {session_id}: Failed to process frame {frame_id}")
                await connection_manager.send_error_message(
                    session_id, f"Failed to process frame {frame_id}"
                )
                return False
            
            # SLAM Processing: Initialize or track the frame
            try:
                frame = result.get('frame_object')
                if frame is not None:
                    # Get session-specific SLAM initializer (should already exist from session setup)
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
                        slam_initializer = session_data.slam_initializer
                        
                        if not slam_initializer.is_initialized:
                            # Initialize SLAM with the first frame (model already loaded)
                            logger.info(f"Session {session_id}: Initializing SLAM with frame {frame_id}")
                            enhanced_frame = await slam_initializer.initialize_frame(
                                frame, connection_manager, session_id
                            )
                            result['slam_initialized'] = True
                            result['slam_status'] = 'initialized'
                        else:
                            # Future: Add tracking logic here
                            logger.info(f"Session {session_id}: SLAM already initialized, frame {frame_id} ready for tracking")
                            result['slam_initialized'] = True
                            result['slam_status'] = 'ready_for_tracking'
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
