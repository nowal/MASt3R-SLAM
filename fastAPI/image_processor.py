"""
Image Processing and SLAM Integration

Handles image decoding, conversion to numpy arrays, and integration with
the SLAM pipeline (resize_img and create_frame functions).
"""

import logging
import time
import cv2
import numpy as np
import torch
import lietorch
from typing import Optional, Tuple, Dict, Any

# Import SLAM functions
from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.frame import create_frame

logger = logging.getLogger("ImageProcessor")

class ImageProcessor:
    """Handles image processing and SLAM pipeline integration"""
    
    def __init__(self, device: str = "cuda:0", img_size: int = 512):
        self.device = device
        self.img_size = img_size
        self.frames_processed = 0
        
        logger.info(f"ImageProcessor initialized with device: {device}, img_size: {img_size}")
        
    def decode_binary_image(self, image_bytes: bytes, format_hint: str = "webp") -> Optional[np.ndarray]:
        """
        Decode binary image data (WebP/JPEG) to numpy array
        
        Args:
            image_bytes: Raw binary image data
            format_hint: Expected format ("webp" or "jpeg")
            
        Returns:
            numpy array in RGB format, float32, range [0,1] or None if failed
        """
        try:
            # Decode using OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                logger.error(f"Failed to decode {format_hint} image")
                return None
                
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1] and ensure float32, C-contiguous
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_contiguous = np.ascontiguousarray(img_normalized)
            
            logger.debug(f"Successfully decoded {format_hint} image: {img_contiguous.shape}")
            return img_contiguous
            
        except Exception as e:
            logger.error(f"❌ Error decoding {format_hint} image: {e}", exc_info=True)
            return None
            
    def process_frame_for_slam(self, img_array: np.ndarray, frame_id: int, timestamp: float) -> Optional[Dict[str, Any]]:
        """
        Process numpy image array through SLAM pipeline
        
        Args:
            img_array: RGB numpy array, float32, range [0,1]
            frame_id: Frame identifier
            timestamp: Frame timestamp
            
        Returns:
            Dictionary with processing results or None if failed
        """
        try:
            start_time = time.time()
            
            # Step 1: Create initial pose (identity for first frame, or could be from tracking)
            T_WC = lietorch.Sim3.Identity(1, device=self.device)
            
            # Step 2: Call create_frame (from frame.py) - this will handle resize_img internally
            logger.debug(f"Processing frame {frame_id} with shape {img_array.shape}")
            create_start = time.time()
            
            frame = create_frame(
                frame_id, 
                img_array,  # Pass the raw decoded image directly
                T_WC, 
                img_size=self.img_size, 
                device=self.device
            )
            create_time = time.time() - create_start
            
            if frame is None:
                logger.error(f"create_frame failed for frame {frame_id}")
                return None
                
            total_time = time.time() - start_time
            self.frames_processed += 1
            
            # Prepare result data
            result = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "original_shape": img_array.shape,
                "resized_shape": frame.img_true_shape.cpu().numpy(),
                "processing_times": {
                    "create_frame": create_time,
                    "total": total_time
                },
                "frame_object": frame,
                "success": True
            }
            
            logger.info(f"✓ Frame {frame_id} processed successfully in {total_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_id} for SLAM: {e}", exc_info=True)
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "error": str(e),
                "success": False
            }
            
    def process_binary_frame(self, image_bytes: bytes, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Complete pipeline: decode binary image and process for SLAM
        
        Args:
            image_bytes: Raw binary image data
            metadata: Frame metadata including format, frame_id, timestamp
            
        Returns:
            Processing results dictionary or None if failed
        """
        try:
            frame_id = metadata.get('frameClientIndex', -1)
            timestamp = metadata.get('timestamp', time.time() * 1000) / 1000.0  # Convert to seconds
            format_hint = metadata.get('format', 'webp')
            
            logger.debug(f"Processing binary frame {frame_id} (format: {format_hint})")
            
            # Step 1: Decode binary image
            img_array = self.decode_binary_image(image_bytes, format_hint)
            if img_array is None:
                return {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "error": f"Failed to decode {format_hint} image",
                    "success": False
                }
                
            # Step 2: Process through SLAM pipeline
            result = self.process_frame_for_slam(img_array, frame_id, timestamp)
            
            if result and result.get('success'):
                # Add metadata to result
                result['metadata'] = metadata
                result['decoded_shape'] = img_array.shape
                
            return result
            
        except Exception as e:
            logger.error(f"Error in process_binary_frame: {e}", exc_info=True)
            return {
                "frame_id": metadata.get('frameClientIndex', -1),
                "timestamp": metadata.get('timestamp', time.time() * 1000) / 1000.0,
                "error": str(e),
                "success": False
            }
            
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "frames_processed": self.frames_processed,
            "device": self.device,
            "img_size": self.img_size
        }
        
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate frame metadata
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = ['frameClientIndex', 'timestamp', 'format']
        
        for field in required_fields:
            if field not in metadata:
                return False, f"Missing required field: {field}"
                
        if not isinstance(metadata['frameClientIndex'], int):
            return False, "frameClientIndex must be an integer"
            
        if not isinstance(metadata['timestamp'], (int, float)):
            return False, "timestamp must be a number"
            
        if metadata['format'] not in ['webp', 'jpeg']:
            return False, f"Unsupported format: {metadata['format']}"
            
        return True, ""

# Global image processor instance
image_processor = ImageProcessor()
