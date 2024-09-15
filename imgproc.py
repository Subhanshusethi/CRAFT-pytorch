# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import cv2

def loadImage(img_file):
    try:
        # Load image using PIL
        with Image.open(img_file) as img:
            img = img.convert('RGB')  # Ensure image is in RGB format
            img = np.array(img)
            
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = img[:, :, :3]  # Drop alpha channel
            
    except Exception as e:
        raise RuntimeError(f"Error loading image {img_file}: {e}")

    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    try:
        img = in_img.copy().astype(np.float32)

        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Error in normalization: {e}")

    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    try:
        img = in_img.copy()
        img *= variance
        img += mean
        img *= 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    except Exception as e:
        raise ValueError(f"Error in denormalization: {e}")

    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    try:
        height, width, channel = img.shape

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

        size_heatmap = (int(target_w/2), int(target_h/2))

    except Exception as e:
        raise ValueError(f"Error in resizing image: {e}")

    return resized, ratio, size_heatmap

def cvt2HeatmapImg(img):
    try:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    except Exception as e:
        raise ValueError(f"Error in converting to heatmap image: {e}")

    return img
