import torch
import copy
from mast3r_slam.mast3r_utils import mast3r_inference_mono, load_mast3r
from mast3r_slam.frame import Mode


class SLAMInitializer:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.model = None
        self.is_initialized = False
        
    def _load_model(self):
        """Load MASt3R model using GPU template for fast cloning or fallback to direct loading"""
        if self.model is None:
            # Import the global template model
            from fastAPI.main import gpu_template_model
            
            if gpu_template_model is not None:
                # Fast path: Clone from GPU template
                print(f"[SLAMInitializer] Cloning model from GPU template...")
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                self.model = copy.deepcopy(gpu_template_model)
                end_time.record()
                
                torch.cuda.synchronize()
                clone_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                
                print(f"[SLAMInitializer] Model cloned from GPU template in {clone_time:.2f}s")
                
                # Log GPU memory after cloning
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"[SLAMInitializer] GPU Memory after cloning: {allocated:.1f}GB")
                
            else:
                # Fallback path: Direct loading
                print(f"[SLAMInitializer] GPU template not available, loading model directly on {self.device}")
                self.model = load_mast3r(device=self.device)
                print(f"[SLAMInitializer] Model loaded successfully via fallback")
        
    async def initialize_frame(self, frame, connection_manager, session_id):
        """Initialize SLAM with the first frame"""
        try:
            # Ensure model is loaded
            self._load_model()
            
            # Send initialization start message to frontend
            await connection_manager.send_message(session_id, {
                "type": "slam_log",
                "level": "info", 
                "message": f"Initializing SLAM with frame {frame.frame_id}"
            })
            
            print(f"[SLAMInitializer] Starting mono inference for frame {frame.frame_id}")
            
            # Add detailed logging for debugging
            print(f"[SLAMInitializer] Frame shape: {frame.img.shape}")
            print(f"[SLAMInitializer] Frame device: {frame.img.device}")
            print(f"[SLAMInitializer] Frame dtype: {frame.img.dtype}")
            print(f"[SLAMInitializer] Frame value range: [{frame.img.min():.3f}, {frame.img.max():.3f}]")
            
            # Perform mono inference
            X_init, C_init = mast3r_inference_mono(self.model, frame)
            
            # Add detailed logging for MASt3R results
            print(f"[SLAMInitializer] MASt3R X_init shape: {X_init.shape}")
            print(f"[SLAMInitializer] MASt3R C_init shape: {C_init.shape}")
            print(f"[SLAMInitializer] MASt3R C_init range: [{C_init.min():.3f}, {C_init.max():.3f}]")
            
            frame.update_pointmap(X_init, C_init)
            
            # Calculate metrics
            num_points = X_init.shape[0]  # Fixed: shape[0] = number of points, shape[1] = coordinates per point
            avg_confidence = C_init.mean().item()
            
            print(f"[SLAMInitializer] Generated {num_points} 3D points with avg confidence {avg_confidence:.4f}")
            
            # Send detailed results to frontend
            await connection_manager.send_message(session_id, {
                "type": "slam_log",
                "level": "success",
                "message": f"Generated {num_points} 3D points"
            })
            
            await connection_manager.send_message(session_id, {
                "type": "slam_log", 
                "level": "success",
                "message": f"Average confidence: {avg_confidence:.4f}"
            })
            
            await connection_manager.send_message(session_id, {
                "type": "slam_log",
                "level": "success", 
                "message": "Frame initialized successfully!"
            })
            
            self.is_initialized = True
            print(f"[SLAMInitializer] SLAM initialization completed successfully")
            
            return frame
            
        except Exception as e:
            error_msg = f"SLAM initialization failed: {str(e)}"
            print(f"[SLAMInitializer] ERROR: {error_msg}")
            
            await connection_manager.send_message(session_id, {
                "type": "slam_log",
                "level": "error",
                "message": error_msg
            })
            
            raise e
    
    def cleanup(self):
        """Clean up session model and recover GPU memory"""
        if self.model is not None:
            print(f"[SLAMInitializer] Cleaning up session model...")
            
            # Log GPU memory before cleanup
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated() / 1e9
                print(f"[SLAMInitializer] GPU Memory before cleanup: {allocated_before:.1f}GB")
            
            # Delete model and clear references
            del self.model
            self.model = None
            
            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated_after = torch.cuda.memory_allocated() / 1e9
                freed_memory = allocated_before - allocated_after
                print(f"[SLAMInitializer] GPU Memory after cleanup: {allocated_after:.1f}GB (freed {freed_memory:.1f}GB)")
            
            print(f"[SLAMInitializer] Session model cleaned up successfully")
        
        # Reset initialization state
        self.is_initialized = False
    
    def reset(self):
        """Reset the initializer state"""
        self.is_initialized = False
        print(f"[SLAMInitializer] Reset - ready for new initialization")
