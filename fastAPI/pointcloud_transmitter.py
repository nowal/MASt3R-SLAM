"""
Point Cloud Transmitter for Real-Time Visualization

This module handles the generation, compression, and transmission of point clouds
from SLAM keyframes to the frontend for Three.js visualization.
"""

import logging
import time
import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch

from .connection_manager import connection_manager
from .pointcloud_utils import generate_incremental_pointcloud, compress_draco
from mast3r_slam.config import config

logger = logging.getLogger("PointCloudTransmitter")


class PointCloudTransmitter:
    """
    Handles point cloud generation and transmission for real-time visualization
    """
    
    def __init__(self):
        self.transmission_count = 0
        self.total_transmission_time = 0.0
        self.total_points_sent = 0
        self.total_bytes_sent = 0
        
        # Get configuration
        self.transmission_interval = config.get("pointcloud", {}).get("transmission_interval", 5)
        self.confidence_threshold = config.get("pointcloud", {}).get("confidence_threshold", 0.5)
        self.max_points = config.get("pointcloud", {}).get("max_points_per_update", 100000)
        self.use_compression = config.get("pointcloud", {}).get("compression", "draco") == "draco"
        self.use_colors = config.get("pointcloud", {}).get("use_colors", True)
        
        logger.info(f"PointCloudTransmitter initialized - interval: {self.transmission_interval}, compression: {self.use_compression}")
    
    async def should_transmit_pointcloud(self, keyframe_count: int) -> bool:
        """
        Check if we should transmit point cloud for this keyframe count
        
        Args:
            keyframe_count: Current number of keyframes
            
        Returns:
            True if point cloud should be transmitted
        """
        return keyframe_count > 0 and keyframe_count % self.transmission_interval == 0
    
    async def generate_and_transmit_pointcloud(self, session_data, session_id: str, keyframe_idx: int) -> Dict[str, Any]:
        """
        Generate point cloud from all keyframes and transmit to frontend
        
        Args:
            session_data: Session data containing keyframes
            session_id: Session identifier
            keyframe_idx: Index of the keyframe that triggered transmission
            
        Returns:
            Dictionary with transmission results
        """
        start_time = time.time()
        self.transmission_count += 1
        
        logger.info(f"[POINTCLOUD] Starting transmission #{self.transmission_count} for session {session_id}, keyframe {keyframe_idx}")
        
        try:
            # Validate session data
            if not session_data or not session_data.keyframes:
                logger.error(f"Session {session_id}: No keyframes available for point cloud generation")
                return self._create_error_result("no_keyframes_available", start_time)
            
            keyframes = session_data.keyframes
            total_keyframes = len(keyframes)
            
            if total_keyframes == 0:
                logger.error(f"Session {session_id}: Empty keyframes list")
                return self._create_error_result("empty_keyframes_list", start_time)
            
            # Generate point cloud from all keyframes
            generation_start = time.time()
            all_points, all_colors, point_stats = await self._generate_combined_pointcloud(keyframes)
            generation_time = time.time() - generation_start
            
            if all_points is None or len(all_points) == 0:
                logger.warning(f"Session {session_id}: No valid points generated from {total_keyframes} keyframes")
                return self._create_error_result("no_valid_points_generated", start_time)
            
            # Apply point limit if necessary
            if len(all_points) > self.max_points:
                logger.info(f"Session {session_id}: Limiting points from {len(all_points)} to {self.max_points}")
                indices = np.random.choice(len(all_points), self.max_points, replace=False)
                all_points = all_points[indices]
                all_colors = all_colors[indices] if all_colors is not None else None
            
            # Prepare transmission data
            transmission_start = time.time()
            
            if self.use_compression:
                # Use Draco compression
                compressed_data, compression_stats = await self._compress_pointcloud(all_points, all_colors)
                transmission_data = compressed_data
                data_type = "draco_compressed"
            else:
                # Send as raw binary arrays
                transmission_data, compression_stats = await self._prepare_raw_pointcloud(all_points, all_colors)
                data_type = "raw_binary"
            
            # Send metadata first
            metadata = {
                "type": "POINT_CLOUD_UPDATE",
                "session_id": session_id,
                "keyframe_idx": keyframe_idx,
                "total_keyframes": total_keyframes,
                "transmission_count": self.transmission_count,
                "data_type": data_type,
                "compression": self.use_compression,
                "point_count": len(all_points),
                "has_colors": all_colors is not None,
                "data_size": len(transmission_data),
                "timestamp": time.time(),
                "generation_time": generation_time,
                "point_stats": point_stats,
                "compression_stats": compression_stats
            }
            
            # Send metadata as JSON
            await connection_manager.send_message(session_id, metadata)
            
            # Send binary point cloud data
            await connection_manager.send_binary_message(session_id, transmission_data)
            
            transmission_time = time.time() - transmission_start
            total_time = time.time() - start_time
            
            # Update statistics
            self.total_transmission_time += total_time
            self.total_points_sent += len(all_points)
            self.total_bytes_sent += len(transmission_data)
            
            # Create success result
            result = {
                "session_id": session_id,
                "keyframe_idx": keyframe_idx,
                "transmission_count": self.transmission_count,
                "status": "transmission_completed",
                "total_keyframes": total_keyframes,
                "point_count": len(all_points),
                "data_size": len(transmission_data),
                "generation_time": generation_time,
                "transmission_time": transmission_time,
                "total_time": total_time,
                "compression_ratio": compression_stats.get("compression_ratio", 1.0),
                "data_type": data_type
            }
            
            logger.info(
                f"[POINTCLOUD] Completed transmission #{self.transmission_count} for session {session_id}: "
                f"{len(all_points)} points, {len(transmission_data)} bytes, {total_time:.3f}s total"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Point cloud transmission failed for session {session_id}: {e}", exc_info=True)
            return self._create_error_result(f"transmission_failed: {str(e)}", start_time)
    
    async def _generate_combined_pointcloud(self, keyframes) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Generate combined point cloud from all keyframes
        
        Args:
            keyframes: SessionKeyframes object
            
        Returns:
            Tuple of (points, colors, stats)
        """
        try:
            all_points_list = []
            all_colors_list = []
            keyframe_stats = []
            
            total_keyframes = len(keyframes)
            
            for i in range(total_keyframes):
                try:
                    keyframe = keyframes[i]
                    
                    # Generate point cloud for this keyframe
                    points, colors = generate_incremental_pointcloud(keyframe, self.confidence_threshold)
                    
                    if points is not None and len(points) > 0:
                        all_points_list.append(points)
                        if colors is not None and self.use_colors:
                            all_colors_list.append(colors)
                        
                        keyframe_stats.append({
                            "keyframe_idx": i,
                            "frame_id": keyframe.frame_id,
                            "point_count": len(points),
                            "has_colors": colors is not None
                        })
                        
                        logger.debug(f"Keyframe {i} (frame_id={keyframe.frame_id}): {len(points)} points")
                    else:
                        logger.warning(f"Keyframe {i} (frame_id={keyframe.frame_id}): No valid points generated")
                        
                except Exception as e:
                    logger.warning(f"Failed to generate points for keyframe {i}: {e}")
                    continue
            
            # Combine all points
            if not all_points_list:
                logger.warning("No valid points generated from any keyframes")
                return None, None, {"error": "no_valid_points", "keyframe_count": total_keyframes}
            
            # Concatenate all points
            all_points = np.concatenate(all_points_list, axis=0).astype(np.float32)
            
            # Concatenate colors if available
            all_colors = None
            if all_colors_list and self.use_colors:
                all_colors = np.concatenate(all_colors_list, axis=0).astype(np.uint8)
            
            stats = {
                "total_keyframes": total_keyframes,
                "keyframes_with_points": len(all_points_list),
                "total_points": len(all_points),
                "has_colors": all_colors is not None,
                "keyframe_stats": keyframe_stats
            }
            
            logger.info(f"Generated combined point cloud: {len(all_points)} points from {len(all_points_list)}/{total_keyframes} keyframes")
            
            return all_points, all_colors, stats
            
        except Exception as e:
            logger.error(f"Failed to generate combined point cloud: {e}", exc_info=True)
            return None, None, {"error": str(e)}
    
    async def _compress_pointcloud(self, points: np.ndarray, colors: Optional[np.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress point cloud using Draco compression
        
        Args:
            points: Point positions as numpy array
            colors: Point colors as numpy array (optional)
            
        Returns:
            Tuple of (compressed_data, compression_stats)
        """
        try:
            compression_start = time.time()
            
            # Use existing Draco compression
            if colors is not None:
                compressed_points, compressed_colors = compress_draco(points, colors)
                
                # Combine compressed data with a simple format:
                # [4 bytes: points_size][points_data][4 bytes: colors_size][colors_data]
                points_size = len(compressed_points).to_bytes(4, byteorder='little')
                colors_size = len(compressed_colors).to_bytes(4, byteorder='little')
                
                compressed_data = points_size + compressed_points + colors_size + compressed_colors
            else:
                compressed_points, _ = compress_draco(points, None)
                
                # Just points data with size header
                points_size = len(compressed_points).to_bytes(4, byteorder='little')
                compressed_data = points_size + compressed_points
            
            compression_time = time.time() - compression_start
            
            # Calculate compression statistics
            original_size = points.nbytes + (colors.nbytes if colors is not None else 0)
            compressed_size = len(compressed_data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            stats = {
                "compression_time": compression_time,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "has_colors": colors is not None
            }
            
            logger.info(f"Draco compression: {original_size} → {compressed_size} bytes (ratio: {compression_ratio:.2f}x)")
            
            return compressed_data, stats
            
        except Exception as e:
            logger.error(f"Draco compression failed: {e}")
            # Fallback to raw binary
            return await self._prepare_raw_pointcloud(points, colors)
    
    async def _prepare_raw_pointcloud(self, points: np.ndarray, colors: Optional[np.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Prepare raw binary point cloud data
        
        Args:
            points: Point positions as numpy array
            colors: Point colors as numpy array (optional)
            
        Returns:
            Tuple of (binary_data, stats)
        """
        try:
            # Convert to contiguous arrays
            points_data = np.ascontiguousarray(points, dtype=np.float32).tobytes()
            
            if colors is not None:
                colors_data = np.ascontiguousarray(colors, dtype=np.uint8).tobytes()
                
                # Combine with size headers
                points_size = len(points_data).to_bytes(4, byteorder='little')
                colors_size = len(colors_data).to_bytes(4, byteorder='little')
                
                binary_data = points_size + points_data + colors_size + colors_data
            else:
                # Just points data with size header
                points_size = len(points_data).to_bytes(4, byteorder='little')
                binary_data = points_size + points_data
            
            stats = {
                "compression_time": 0.0,
                "original_size": len(binary_data),
                "compressed_size": len(binary_data),
                "compression_ratio": 1.0,
                "has_colors": colors is not None
            }
            
            return binary_data, stats
            
        except Exception as e:
            logger.error(f"Failed to prepare raw point cloud data: {e}")
            raise e
    
    def _create_error_result(self, error_msg: str, start_time: float) -> Dict[str, Any]:
        """Create an error result dictionary"""
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "transmission_count": self.transmission_count,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "status": "transmission_failed",
            "error": error_msg,
            "point_count": 0,
            "data_size": 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get point cloud transmission statistics"""
        avg_transmission_time = (
            self.total_transmission_time / self.transmission_count 
            if self.transmission_count > 0 else 0.0
        )
        
        avg_points_per_transmission = (
            self.total_points_sent / self.transmission_count 
            if self.transmission_count > 0 else 0
        )
        
        avg_bytes_per_transmission = (
            self.total_bytes_sent / self.transmission_count 
            if self.transmission_count > 0 else 0
        )
        
        return {
            "transmission_count": self.transmission_count,
            "total_transmission_time": self.total_transmission_time,
            "average_transmission_time": avg_transmission_time,
            "total_points_sent": self.total_points_sent,
            "total_bytes_sent": self.total_bytes_sent,
            "average_points_per_transmission": avg_points_per_transmission,
            "average_bytes_per_transmission": avg_bytes_per_transmission,
            "transmission_interval": self.transmission_interval,
            "use_compression": self.use_compression,
            "confidence_threshold": self.confidence_threshold,
            "max_points": self.max_points
        }
    
    def reset_stats(self) -> None:
        """Reset transmission statistics"""
        logger.info("Resetting point cloud transmission statistics")
        self.transmission_count = 0
        self.total_transmission_time = 0.0
        self.total_points_sent = 0
        self.total_bytes_sent = 0


# Global point cloud transmitter instance
pointcloud_transmitter = PointCloudTransmitter()
