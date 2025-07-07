import numpy as np
import torch
import DracoPy as dracopy
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray

def generate_incremental_pointcloud(keyframe, c_conf_threshold):
    if config["use_calib"]:
        X_canon = constrain_points_to_ray(
            keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
        )
        keyframe.X_canon = X_canon.squeeze(0)
    
    pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
    color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
    
    valid = (
        keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
        > c_conf_threshold
    )
    
    points = pW[valid]
    colors = color[valid]
    
    points = np.ascontiguousarray(points, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.uint8)

    return points, colors

def compress_draco(points: np.ndarray, colors: np.ndarray):
    # Create a DracoPy encoder
    encoder = dracopy.Encoder()

    # Compress the points
    compressed_points = encoder.encode(points)

    # Compress the colors
    compressed_colors = encoder.encode(colors)

    return compressed_points, compressed_colors

def generate_batched_pointcloud(keyframes, indices, c_conf_threshold):
    batched_data = {}
    for i in indices:
        keyframe = keyframes[i]
        points, colors = generate_incremental_pointcloud(keyframe, c_conf_threshold)
        compressed_points, compressed_colors = compress_draco(points, colors)
        batched_data[i] = {
            "points": compressed_points,
            "colors": compressed_colors
        }
    return batched_data
