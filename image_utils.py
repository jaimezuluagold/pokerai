# src/utils/image_utils.py
"""
Image Processing Utilities for Poker AI

This module provides common image processing functions.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict

def resize_image(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    """
    Resize an image to a specified width or height, maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width (None to calculate from height)
        height: Target height (None to calculate from width)
        
    Returns:
        Resized image
    """
    if width is None and height is None:
        return image
        
    h, w = image.shape[:2]
    
    if width is None:
        # Calculate width to maintain aspect ratio
        ratio = height / float(h)
        width = int(w * ratio)
    elif height is None:
        # Calculate height to maintain aspect ratio
        ratio = width / float(w)
        height = int(h * ratio)
        
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized

def extract_roi(image: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
    """
    Extract a region of interest from an image.
    
    Args:
        image: Input image
        roi: Dictionary with keys left, top, width, height
        
    Returns:
        ROI image
    """
    x = roi["left"]
    y = roi["top"]
    w = roi["width"]
    h = roi["height"]
    
    # Ensure coordinates are within image bounds
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    
    return image[y:y+h, x:x+w].copy()

def enhance_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Enhance an image for better OCR results.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Increase size for better OCR
    gray = cv2.resize(gray, (0, 0), fx=2, fy=2)
    
    # Apply bilateral filter to preserve edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Invert if needed (text should be black on white for OCR)
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
        
    return thresh

def draw_bounding_boxes(image: np.ndarray, boxes: List[Dict], color: Tuple[int, int, int] = (0, 255, 0), 
                       thickness: int = 2, labels: Optional[List[str]] = None) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: Input image
        boxes: List of dictionaries with x, y, width, height keys
        color: Box color as BGR tuple
        thickness: Line thickness
        labels: Optional list of label strings
        
    Returns:
        Image with bounding boxes
    """
    result = image.copy()
    
    for i, box in enumerate(boxes):
        x = box["x"]
        y = box["y"]
        w = box["width"]
        h = box["height"]
        
        # Draw rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        # Add label if provided
        if labels and i < len(labels):
            cv2.putText(
                result, labels[i], (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
    return result