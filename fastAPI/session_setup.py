"""
Session Setup Component

Handles SLAM model setup for new WebSocket sessions with real-time user feedback.
"""

import logging
from .slam_initializer import SLAMInitializer
from .global_optimizer import RealGlobalOptimizer
from .frame_saver import frame_saver

logger = logging.getLogger("SessionSetup")


class SessionSetup:
    """Handles SLAM model setup for new WebSocket sessions"""
    
    def __init__(self):
        logger.info("SessionSetup component initialized")
        
    async def setup_slam_for_session(self, session_data, connection_manager, session_id):
        """
        Set up SLAM initializer for a new session with real-time feedback
        
        Args:
            session_data: SessionData object to attach SLAM initializer to
            connection_manager: For sending messages to frontend
            session_id: Session identifier for logging/messaging
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            logger.info(f"Starting SLAM setup for session {session_id}")
            
            # Set up frame saving directory for this session
            frame_saver.setup_session(session_id)
            
            # Send loading message to frontend
            await connection_manager.send_debug_message(
                session_id, "info", "Loading MASt3R model for session..."
            )
            
            # Create SLAM initializer and load model (fast clone from GPU template)
            logger.info(f"Creating SLAMInitializer for session {session_id}")
            session_data.slam_initializer = SLAMInitializer()
            
            # Load model immediately (this will clone from GPU template)
            session_data.slam_initializer._load_model()
            
            # Initialize shared retrieval database
            await connection_manager.send_debug_message(
                session_id, "info", "Initializing shared retrieval database..."
            )
            
            try:
                from mast3r_slam.mast3r_utils import load_retriever
                model = session_data.slam_initializer.model
                device = model.device if hasattr(model, 'device') else "cuda"
                session_data.retrieval_database = load_retriever(model, device=device)
                logger.info(f"Shared retrieval database initialized for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to initialize retrieval database for session {session_id}: {e}")
                session_data.retrieval_database = None
                await connection_manager.send_debug_message(
                    session_id, "warning", "Retrieval database initialization failed - continuing without database"
                )
            
            # Initialize global optimizer with the loaded model (synchronous mode)
            await connection_manager.send_debug_message(
                session_id, "info", "Initializing synchronous global optimizer..."
            )
            
            try:
                model = session_data.slam_initializer.model
                device = model.device if hasattr(model, 'device') else "cuda"
                session_data.global_optimizer = RealGlobalOptimizer(model, device)
                logger.info(f"Synchronous global optimizer initialized for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to initialize global optimizer for session {session_id}: {e}")
                # Continue without global optimizer - tracking will still work
                session_data.global_optimizer = None
                await connection_manager.send_debug_message(
                    session_id, "warning", "Global optimizer initialization failed - continuing without optimization"
                )
            
            # Initialize real relocalization with the loaded model
            await connection_manager.send_debug_message(
                session_id, "info", "Initializing relocalization system..."
            )
            
            try:
                from .relocalization import RealRelocalizer
                model = session_data.slam_initializer.model
                device = model.device if hasattr(model, 'device') else "cuda"
                session_data.relocalizer = RealRelocalizer(model, device)
                logger.info(f"Real relocalizer initialized for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to initialize relocalizer for session {session_id}: {e}")
                # Continue without relocalizer - will fallback to global instance
                session_data.relocalizer = None
                await connection_manager.send_debug_message(
                    session_id, "warning", "Relocalizer initialization failed - will use fallback"
                )
            
            # Send ready message to frontend
            await connection_manager.send_debug_message(
                session_id, "success", "MASt3R model ready - you can start sending frames!"
            )
            
            logger.info(f"SLAM setup completed successfully for session {session_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to setup SLAM for session: {str(e)}"
            logger.error(f"Session {session_id}: {error_msg}", exc_info=True)
            
            # Send error message to frontend
            await connection_manager.send_error_message(session_id, error_msg)
            
            # Clean up any partial initialization
            if hasattr(session_data, 'slam_initializer') and session_data.slam_initializer is not None:
                try:
                    session_data.slam_initializer.cleanup()
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup for session {session_id}: {cleanup_error}")
                session_data.slam_initializer = None
            
            return False
    
    def get_stats(self):
        """Get session setup statistics"""
        return {
            "component": "SessionSetup",
            "status": "active"
        }
