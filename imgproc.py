# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageFile
import cv2

def loadImage(img_file):
    try:
        # Allow loading of truncated images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        # Load image using PIL
        with Image.open(img_file) as img:
            img = img.convert('RGB')  # Ensure image is in RGB format
            img = np.array(img)
            
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = img[:, :, :3]  # Drop alpha channel
            
    except OSError as e:
        raise RuntimeError(f"OS error loading image {img_file}: {e}")
    except ValueError as e:
        raise RuntimeError(f"Value error in processing image {img_file}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading image {img_file}: {e}")
    
    # Verify the image is loaded successfully
    if img is None:
        raise RuntimeError(f"Image loading failed for {img_file}")
    
    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    try:
        img = in_img.copy().astype(np.float32)

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Input image must be 3-dimensional with 3 channels")

        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    except TypeError as e:
        raise ValueError(f"Type error in normalization: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error in normalization: {e}")

    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    try:
        img = in_img.copy()

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Input image must be 3-dimensional with 3 channels")

        img *= variance
        img += mean
        img *= 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    except TypeError as e:
        raise ValueError(f"Type error in denormalization: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error in denormalization: {e}")

    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    try:
        if img is None or img.ndim != 3:
            raise ValueError("Input image must be a non-empty 3-dimensional array")
        
        height, width, channel = img.shape

        if height == 0 or width == 0:
            raise ValueError(f"Invalid image dimensions: height={height}, width={width}")
        
        # Magnify image size
        target_size = mag_ratio * max(height, width)

        # Set original image size
        if target_size > square_size:
            target_size = square_size
        
        ratio = target_size / max(height, width)

        target_h, target_w = int(height * ratio), int(width * ratio)
        proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

        # Make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resized[0:target_h, 0:target_w, :] = proc
        target_h, target_w = target_h32, target_w32

        size_heatmap = (int(target_w / 2), int(target_h / 2))

    except ValueError as e:
        raise ValueError(f"Value error in resizing image: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error in resizing image: {e}")

    return resized, ratio, size_heatmap

def cvt2HeatmapImg(img):
    try:
        if img.ndim != 2:
            raise ValueError("Input image must be a 2D grayscale image")
        
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    except ValueError as e:
        raise ValueError(f"Value error in converting to heatmap image: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error in converting to heatmap image: {e}")

    return img
