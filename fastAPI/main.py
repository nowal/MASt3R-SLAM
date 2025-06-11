"""
FastAPI SLAM Backend - Main Application

Main FastAPI application with WebSocket endpoints for binary image transmission
and SLAM processing integration.
"""

import os
import logging
import torch
from contextlib import asynccontextmanager
from typing import Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .connection_manager import connection_manager
from .data_receiver import data_receiver
from .image_processor import image_processor
from .global_optimizer import get_global_optimization_stats, reset_global_optimization_stats
from .relocalization import get_relocalization_stats, reset_relocalization_stats

# Import SLAM config loading
from mast3r_slam.config import load_config, config
from mast3r_slam.mast3r_utils import load_mast3r

# Global GPU template model for fast cloning
gpu_template_model = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FastAPI_SLAM_Main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("FastAPI SLAM Backend starting up...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load SLAM configuration
    config_path = "config/base.yaml"
    logger.info(f"Loading SLAM configuration from: {config_path}")
    try:
        load_config(config_path)
        logger.info("✓ SLAM configuration loaded successfully")
        logger.info(f"Config dataset settings: {config.get('dataset', {})}")
    except Exception as e:
        logger.error(f"❌ Failed to load SLAM configuration: {e}")
        raise e
    
    # Initialize image processor with appropriate device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    global image_processor
    image_processor.device = device
    
    logger.info(f"Image processor initialized with device: {device}")
    
    # Load GPU template model for fast session cloning
    global gpu_template_model
    if torch.cuda.is_available():
        try:
            logger.info("Loading MASt3R template model on GPU for fast cloning...")
            
            def log_gpu_memory(context=""):
                allocated = torch.cuda.memory_allocated() / 1e9
                cached = torch.cuda.memory_reserved() / 1e9
                logger.info(f"{context} - GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
            
            log_gpu_memory("Before template loading")
            gpu_template_model = load_mast3r(device=device)
            log_gpu_memory("After template loading")
            
            logger.info("✓ GPU template model loaded successfully - ready for fast session creation")
            
        except Exception as e:
            logger.error(f"❌ Failed to load GPU template model: {e}")
            logger.info("Will fallback to per-session model loading")
            gpu_template_model = None
    else:
        logger.info("CUDA not available - will use per-session model loading")
        gpu_template_model = None
    
    yield
    
    logger.info("FastAPI SLAM Backend shutting down...")
    
    # Get all active sessions for cleanup
    sessions_info = await connection_manager.get_all_sessions_info()
    active_count = sessions_info.get('active_sessions_count', 0)
    
    if active_count > 0:
        logger.info(f"Cleaning up {active_count} active sessions...")
        # Note: Sessions will be cleaned up automatically when WebSocket connections close
    
    # Cleanup GPU template model
    if gpu_template_model is not None:
        logger.info("Cleaning up GPU template model...")
        del gpu_template_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✓ Template model cleaned up")
    
    logger.info("Shutdown complete.")

# Create FastAPI app
app = FastAPI(
    title="SLAM Backend API",
    description="FastAPI backend for real-time SLAM with binary image transmission",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
origins = [
    "http://localhost",
    "http://localhost:3000", 
    "http://localhost:8000",
    "http://localhost:5173",
    "https://localhost",
    "https://api.takeshapehome.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def read_root():
    """Root endpoint"""
    return {
        "message": "FastAPI SLAM Backend with Binary Image Transmission",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/connect")
async def handle_connect():
    """Generate a unique session ID for WebSocket connection"""
    try:
        session_id = await connection_manager.generate_session_id()
        logger.info(f"Generated session ID: {session_id}")
        
        return {
            "sessionId": session_id,
            "message": f"Connect WebSocket to /ws/{session_id} and send frame metadata + binary data",
            "protocol": "binary",
            "supported_formats": ["webp", "jpeg"]
        }
        
    except Exception as e:
        logger.error(f"Error generating session ID: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate session: {str(e)}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for binary image transmission"""
    
    async with connection_manager.session_context(websocket, session_id) as session_data:
        if not session_data:
            logger.error(f"Failed to create session context for {session_id}")
            return
            
        logger.info(f"WebSocket session {session_id} started")
        
        try:
            # Main message handling loop
            while True:
                # Receive message (can be text/JSON or binary)
                try:
                    # Use receive() to get the raw message and check its type
                    message_data = await websocket.receive()
                    
                    if "text" in message_data:
                        # Handle JSON/text message
                        message = message_data["text"]
                        success = await data_receiver.handle_websocket_message(
                            websocket, session_id, message
                        )
                        
                        if not success:
                            logger.warning(f"Session {session_id}: Failed to handle text message")
                            
                    elif "bytes" in message_data:
                        # Handle binary message
                        message = message_data["bytes"]
                        success = await data_receiver.handle_websocket_message(
                            websocket, session_id, message
                        )
                        
                        if not success:
                            logger.warning(f"Session {session_id}: Failed to handle binary message")
                            
                    else:
                        logger.warning(f"Session {session_id}: Received unknown message type: {message_data}")
                        
                except WebSocketDisconnect:
                    logger.info(f"Session {session_id}: WebSocket disconnected")
                    break
                except Exception as e:
                    logger.error(f"Session {session_id}: Unexpected error in message loop: {e}", exc_info=True)
                    await connection_manager.send_error_message(
                        session_id, f"Unexpected error: {str(e)}"
                    )
                    break
                    
        except Exception as e:
            logger.error(f"Session {session_id}: Error in WebSocket endpoint: {e}", exc_info=True)
            
        finally:
            logger.info(f"WebSocket session {session_id} ended")

@app.get("/active_sessions")
async def get_active_sessions():
    """Get information about active sessions"""
    try:
        sessions_info = await connection_manager.get_all_sessions_info()
        data_stats = data_receiver.get_stats()
        
        return {
            **sessions_info,
            "data_receiver_stats": data_stats,
            "system_info": {
                "cuda_available": torch.cuda.is_available(),
                "device": image_processor.device,
                "torch_version": torch.__version__
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting active sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get sessions info: {str(e)}")

@app.get("/tracking_stats")
async def get_tracking_stats():
    """Get comprehensive tracking statistics"""
    try:
        sessions_info = await connection_manager.get_all_sessions_info()
        data_stats = data_receiver.get_stats()
        global_opt_stats = get_global_optimization_stats()
        reloc_stats = get_relocalization_stats()
        
        return {
            "sessions": sessions_info,
            "data_receiver": data_stats,
            "global_optimization": global_opt_stats,
            "relocalization": reloc_stats,
            "system_info": {
                "cuda_available": torch.cuda.is_available(),
                "device": image_processor.device,
                "torch_version": torch.__version__
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting tracking stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get tracking stats: {str(e)}")

@app.post("/reset_stats")
async def reset_tracking_stats():
    """Reset all tracking statistics"""
    try:
        reset_global_optimization_stats()
        reset_relocalization_stats()
        
        logger.info("All tracking statistics reset")
        return {
            "status": "success",
            "message": "All tracking statistics have been reset"
        }
        
    except Exception as e:
        logger.error(f"Error resetting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset stats: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        sessions_info = await connection_manager.get_all_sessions_info()
        data_stats = data_receiver.get_stats()
        
        return {
            "status": "healthy",
            "active_sessions": sessions_info.get('active_sessions_count', 0),
            "messages_processed": data_stats.get('messages_processed', 0),
            "frames_processed": data_stats.get('image_processor_stats', {}).get('frames_processed', 0),
            "cuda_available": torch.cuda.is_available(),
            "device": image_processor.device
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    import platform
    
    logger.info("Starting FastAPI SLAM Backend...")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Operating System: {platform.system()} {platform.release()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    uvicorn.run(
        "fastAPI.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Important: Set reload=False for stability
    )
