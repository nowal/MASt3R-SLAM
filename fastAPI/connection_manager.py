"""
WebSocket Connection and Session Management

Handles WebSocket connections, session lifecycle, and connection state management.
"""

import asyncio
import uuid
import logging
import time
from typing import Dict, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from .session_setup import SessionSetup
from .session_keyframes import SessionKeyframes
from .frame_tracker_wrapper import FrameTrackerWrapper
from mast3r_slam.frame import Mode, Frame

logger = logging.getLogger("ConnectionManager")

class SessionData:
    """Data structure for managing individual session state"""
    
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.start_time = time.time()
        self.frames_received = 0
        self.last_frame_time = None
        self.is_connected = True
        self.metadata_buffer = {}  # Store metadata waiting for binary data
        self.slam_initializer = None  # Will be created when needed
        
        # SLAM tracking state
        self.slam_mode = Mode.INIT
        self.keyframes: Optional[SessionKeyframes] = None  # For tracking (stable poses)
        self.optimization_keyframes: Optional[SessionKeyframes] = None  # For bundle adjustment (optimized poses)
        self.frame_tracker: Optional[FrameTrackerWrapper] = None
        self.frame_tracker_instance = None  # The actual FrameTracker instance
        self.last_frame: Optional[Frame] = None
        self.current_camera_pose = None  # Current estimated camera pose (like states.get_frame().T_WC in original)
        self.tracking_stats = {
            "frames_processed": 0,
            "keyframes_added": 0,
            "tracking_failures": 0,
            "relocalization_attempts": 0
        }
        
    def update_frame_stats(self):
        """Update frame reception statistics"""
        self.frames_received += 1
        self.last_frame_time = time.time()
        
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information for debugging/monitoring"""
        slam_info = {
            "slam_mode": self.slam_mode.name if self.slam_mode else "UNKNOWN",
            "num_keyframes": len(self.keyframes) if self.keyframes else 0,
            "num_optimization_keyframes": len(self.optimization_keyframes) if self.optimization_keyframes else 0,
            "has_frame_tracker": self.frame_tracker is not None,
            "last_frame_id": self.last_frame.frame_id if self.last_frame else None,
            "tracking_stats": self.tracking_stats.copy()
        }
        
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "frames_received": self.frames_received,
            "last_frame_time": self.last_frame_time,
            "is_connected": self.is_connected,
            "uptime_seconds": time.time() - self.start_time,
            "slam_info": slam_info
        }
    
    def cleanup_slam_state(self):
        """Clean up SLAM tracking state"""
        if self.keyframes:
            self.keyframes.cleanup()
            self.keyframes = None
        
        if self.optimization_keyframes:
            self.optimization_keyframes.cleanup()
            self.optimization_keyframes = None
        
        self.frame_tracker = None
        self.frame_tracker_instance = None
        self.last_frame = None
        self.slam_mode = Mode.INIT
        
        # Reset tracking stats
        self.tracking_stats = {
            "frames_processed": 0,
            "keyframes_added": 0,
            "tracking_failures": 0,
            "relocalization_attempts": 0
        }

class ConnectionManager:
    """Manages WebSocket connections and session lifecycle"""
    
    def __init__(self):
        self.active_sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
        self.session_setup = SessionSetup()
        
    async def cleanup_stale_sessions(self):
        """Clean up sessions that are marked as disconnected but still in memory"""
        stale_sessions = []
        
        async with self._lock:
            for session_id, session_data in list(self.active_sessions.items()):
                if not session_data.is_connected:
                    stale_sessions.append(session_id)
                    
        if stale_sessions:
            logger.info(f"Cleaning up {len(stale_sessions)} stale sessions: {stale_sessions}")
            for session_id in stale_sessions:
                await self.disconnect_session(session_id)
        
    async def generate_session_id(self) -> str:
        """Generate a unique session ID"""
        session_id = str(uuid.uuid4())
        logger.info(f"Generated new session ID: {session_id}")
        return session_id
        
    async def connect_session(self, websocket: WebSocket, session_id: str) -> SessionData:
        """Accept WebSocket connection and create session with graceful reconnection handling"""
        await websocket.accept()
        
        async with self._lock:
            existing_session = self.active_sessions.get(session_id)
            
            if existing_session:
                # Check if existing session is actually dead/disconnected
                if not existing_session.is_connected:
                    logger.info(f"Found stale session {session_id}, cleaning up for reconnection")
                    # Clean up the stale session (but don't call disconnect_session to avoid deadlock)
                    if existing_session.slam_initializer is not None:
                        logger.info(f"Cleaning up SLAM initializer for stale session {session_id}")
                        existing_session.slam_initializer.cleanup()
                        existing_session.slam_initializer = None
                    
                    # Remove the stale session
                    del self.active_sessions[session_id]
                    logger.info(f"Stale session {session_id} cleaned up, proceeding with reconnection")
                else:
                    # Session is still active - this is a real collision
                    logger.warning(f"Session ID collision detected: {session_id} (session is still active)")
                    raise ValueError(f"Session {session_id} already exists and is active")
                
            # Create new session
            session_data = SessionData(session_id, websocket)
            self.active_sessions[session_id] = session_data
            
        logger.info(f"WebSocket connected for session: {session_id}")
        await self.send_debug_message(session_id, "info", f"WebSocket connected for session {session_id}")
        
        # Set up SLAM for this session immediately
        setup_success = await self.session_setup.setup_slam_for_session(
            session_data, self, session_id
        )
        
        if not setup_success:
            logger.warning(f"SLAM setup failed for session {session_id}, but session will continue")
            # Note: We don't disconnect the session if SLAM setup fails
            # The session can still be used, just without SLAM functionality
        
        return session_data
        
    async def disconnect_session(self, session_id: str):
        """Clean up session on disconnect"""
        async with self._lock:
            session_data = self.active_sessions.pop(session_id, None)
            
        if session_data:
            session_data.is_connected = False
            
            # Clean up SLAM initializer if it exists
            if session_data.slam_initializer is not None:
                logger.info(f"Cleaning up SLAM initializer for session {session_id}")
                session_data.slam_initializer.cleanup()
                session_data.slam_initializer = None
            
            # Clean up SLAM tracking state
            logger.info(f"Cleaning up SLAM tracking state for session {session_id}")
            session_data.cleanup_slam_state()
            
            session_info = session_data.get_session_info()
            logger.info(f"Session {session_id} disconnected. Stats: {session_info}")
            
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID"""
        async with self._lock:
            return self.active_sessions.get(session_id)
            
    async def send_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Send JSON message to a specific session"""
        session_data = await self.get_session(session_id)
        if not session_data or not session_data.is_connected:
            return False
            
        try:
            await session_data.websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send message to session {session_id}: {e}")
            await self.disconnect_session(session_id)
            return False
            
    async def send_debug_message(self, session_id: str, level: str, message: str) -> bool:
        """Send debug log message to frontend"""
        debug_msg = {
            "type": "debug_log",
            "level": level,
            "message": message,
            "timestamp": time.time() * 1000,  # JavaScript timestamp format
            "session_id": session_id
        }
        return await self.send_message(session_id, debug_msg)
        
    async def send_error_message(self, session_id: str, error: str) -> bool:
        """Send error message to frontend"""
        error_msg = {
            "type": "error",
            "message": error,
            "timestamp": time.time() * 1000,
            "session_id": session_id
        }
        return await self.send_message(session_id, error_msg)
        
    async def update_frame_stats(self, session_id: str):
        """Update frame reception statistics for a session"""
        session_data = await self.get_session(session_id)
        if session_data:
            session_data.update_frame_stats()
            
    async def get_all_sessions_info(self) -> Dict[str, Any]:
        """Get information about all active sessions"""
        async with self._lock:
            sessions_info = []
            for session_data in self.active_sessions.values():
                sessions_info.append(session_data.get_session_info())
                
        return {
            "active_sessions_count": len(sessions_info),
            "sessions": sessions_info
        }
        
    @asynccontextmanager
    async def session_context(self, websocket: WebSocket, session_id: str):
        """Context manager for handling session lifecycle"""
        session_data = None
        try:
            session_data = await self.connect_session(websocket, session_id)
            yield session_data
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session: {session_id}")
        except Exception as e:
            logger.error(f"Error in session {session_id}: {e}", exc_info=True)
            if session_data:
                await self.send_error_message(session_id, f"Session error: {str(e)}")
        finally:
            await self.disconnect_session(session_id)
            
    async def store_frame_metadata(self, session_id: str, metadata: Dict[str, Any]):
        """Store frame metadata waiting for binary data"""
        session_data = await self.get_session(session_id)
        if session_data:
            session_data.metadata_buffer = metadata
            logger.debug(f"Stored metadata for session {session_id}: frame {metadata.get('frameClientIndex')}")
            
    async def get_and_clear_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get and clear stored frame metadata"""
        session_data = await self.get_session(session_id)
        if session_data and session_data.metadata_buffer:
            metadata = session_data.metadata_buffer.copy()
            session_data.metadata_buffer = {}
            return metadata
        return None

# Global connection manager instance
connection_manager = ConnectionManager()
