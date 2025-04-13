"""
Card Detector Module for Poker AI

This module handles the detection of cards on the poker table,
isolating card regions for further classification.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CardDetector:
    """
    Detects and isolates cards from the poker table image.
    """
    
    def __init__(self, 
                 min_card_width: int = 30,
                 min_card_height: int = 45,
                 max_card_width: int = 150,
                 max_card_height: int = 200,
                 edge_detection_params: Dict = None,
                 template_dir: str = None):
        """
        Initialize the card detector.
        
        Args:
            min_card_width: Minimum width of a card in pixels
            min_card_height: Minimum height of a card in pixels
            max_card_width: Maximum width of a card in pixels
            max_card_height: Maximum height of a card in pixels
            edge_detection_params: Parameters for edge detection
            template_dir: Directory containing card templates for template matching
        """
        self.min_card_width = min_card_width
        self.min_card_height = min_card_height
        self.max_card_width = max_card_width
        self.max_card_height = max_card_height
        
        # Default parameters for edge detection
        self.edge_detection_params = edge_detection_params or {
            "threshold1": 50,
            "threshold2": 150,
            "aperture_size": 3
        }
        
        # Template matching parameters
        self.template_dir = template_dir
        self.templates = {}
        
        # Load templates if directory is provided
        if template_dir and os.path.exists(template_dir):
            self._load_templates()
            
        # Cached results for optimization
        self.last_frame = None
        self.last_result = None
    
    def _load_templates(self) -> None:
        """Load card templates for template matching approach."""
        if not self.template_dir:
            return
            
        logger.info(f"Loading card templates from {self.template_dir}")
        
        # Iterate through template directory
        for root, _, files in os.walk(self.template_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    # Extract card info from filename (e.g., "2C.png" for 2 of clubs)
                    card_name = os.path.splitext(file)[0]
                    
                    # Load template
                    template_path = os.path.join(root, file)
                    template = cv2.imread(template_path)
                    
                    if template is not None:
                        # Convert to grayscale for matching
                        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                        self.templates[card_name] = template_gray
                        logger.debug(f"Loaded template for {card_name}")
                    else:
                        logger.warning(f"Failed to load template: {template_path}")
        
        logger.info(f"Loaded {len(self.templates)} card templates")
    
    def detect_cards(self, frame: np.ndarray, region: str = "unknown") -> List[Dict]:
        """
        Detect cards in the input frame.
        
        Args:
            frame: Input image frame
            region: Name of the region (for logging)
            
        Returns:
            List of dictionaries containing card regions and metadata
        """
        # Check if this is the same frame as before
        if self.last_frame is not None and np.array_equal(frame, self.last_frame):
            return self.last_result
            
        # Store frame for caching
        self.last_frame = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect card regions (using multiple strategies for robustness)
        cards = []
        
        # Method 1: Edge detection + contour finding
        cards_method1 = self._detect_cards_edges(blurred, frame, region)
        if cards_method1:
            cards.extend(cards_method1)
            
        # Method 2: Template matching (if templates are available)
        if self.templates:
            cards_method2 = self._detect_cards_templates(gray, frame, region)
            if cards_method2:
                # Merge with method 1 results, removing duplicates
                cards = self._merge_card_detections(cards, cards_method2)
        
        # Method 3: Color-based detection (specifically for poker table)
        if not cards:  # Fallback if other methods failed
            cards_method3 = self._detect_cards_color(frame, region)
            if cards_method3:
                cards = self._merge_card_detections(cards, cards_method3)
        
        # Filter out likely false positives
        cards = self._filter_card_candidates(cards, gray.shape)
        
        # Cache the result
        self.last_result = cards
        
        return cards
    
    def _detect_cards_edges(self, gray: np.ndarray, orig_frame: np.ndarray, region: str) -> List[Dict]:
        """
        Detect cards using edge detection and contour finding.
        
        Args:
            gray: Grayscale image
            orig_frame: Original color frame
            region: Region name for logging
            
        Returns:
            List of detected card regions
        """
        # Perform edge detection
        edges = cv2.Canny(
            gray, 
            self.edge_detection_params["threshold1"],
            self.edge_detection_params["threshold2"],
            apertureSize=self.edge_detection_params["aperture_size"]
        )
        
        # Dilate edges to connect broken contours
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cards = []
        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out non-card shapes based on size and aspect ratio
            if (self.min_card_width <= w <= self.max_card_width and 
                self.min_card_height <= h <= self.max_card_height and
                0.5 <= h/w <= 2.0):  # Aspect ratio constraint
                
                # Check if it has 4 corners (like a card)
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) >= 4:  # Allow some flexibility in corner detection
                    card_img = orig_frame[y:y+h, x:x+w].copy()
                    
                    # Store card information
                    card_info = {
                        "id": f"{region}_edge_{i}",
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "image": card_img,
                        "method": "edge",
                        "confidence": 0.7  # Default confidence
                    }
                    
                    cards.append(card_info)
        
        logger.debug(f"Detected {len(cards)} cards in {region} using edge detection")
        return cards
    
    def _detect_cards_templates(self, gray: np.ndarray, orig_frame: np.ndarray, region: str) -> List[Dict]:
        """
        Detect cards using template matching.
        
        Args:
            gray: Grayscale image
            orig_frame: Original color frame
            region: Region name for logging
            
        Returns:
            List of detected card regions
        """
        if not self.templates:
            return []
            
        cards = []
        match_threshold = 0.7  # Minimum matching threshold
        
        for card_name, template in self.templates.items():
            # Multi-scale template matching
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                # Resize template
                resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
                w, h = resized_template.shape[::-1]
                
                # Skip if template is too large
                if w > gray.shape[1] or h > gray.shape[0]:
                    continue
                
                # Template matching
                res = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)
                
                # Find locations above threshold
                locations = np.where(res >= match_threshold)
                
                # Process matches
                for pt in zip(*locations[::-1]):
                    x, y = pt
                    
                    # Check if this region overlaps with previously detected card
                    overlap = False
                    for existing_card in cards:
                        ex, ey = existing_card["x"], existing_card["y"]
                        ew, eh = existing_card["width"], existing_card["height"]
                        
                        # Check overlap
                        if (x < ex + ew and x + w > ex and 
                            y < ey + eh and y + h > ey):
                            # If overlap and higher confidence, replace
                            if res[y, x] > existing_card["confidence"]:
                                existing_card.update({
                                    "x": x,
                                    "y": y,
                                    "width": w,
                                    "height": h,
                                    "image": orig_frame[y:y+h, x:x+w].copy(),
                                    "card_name": card_name,
                                    "method": "template",
                                    "confidence": float(res[y, x])
                                })
                            overlap = True
                            break
                    
                    if not overlap:
                        card_info = {
                            "id": f"{region}_template_{len(cards)}",
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "image": orig_frame[y:y+h, x:x+w].copy(),
                            "card_name": card_name,
                            "method": "template",
                            "confidence": float(res[y, x])
                        }
                        cards.append(card_info)
        
        logger.debug(f"Detected {len(cards)} cards in {region} using template matching")
        return cards
    
    def _detect_cards_color(self, frame: np.ndarray, region: str) -> List[Dict]:
        """
        Detect cards based on color characteristics.
        
        Args:
            frame: Color image frame
            region: Region name for logging
            
        Returns:
            List of detected card regions
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color range for white cards
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cards = []
        for i, contour in enumerate(contours):
            # Get area and bounding rectangle
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size and aspect ratio
            if (self.min_card_width <= w <= self.max_card_width and 
                self.min_card_height <= h <= self.max_card_height and
                0.5 <= h/w <= 2.0 and  # Aspect ratio constraint
                area > 500):  # Minimum area to avoid noise
                
                # Create card object
                card_img = frame[y:y+h, x:x+w].copy()
                
                card_info = {
                    "id": f"{region}_color_{i}",
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "image": card_img,
                    "method": "color",
                    "confidence": 0.6  # Lower confidence for color method
                }
                
                cards.append(card_info)
        
        logger.debug(f"Detected {len(cards)} cards in {region} using color detection")
        return cards
    
    def _merge_card_detections(self, cards1: List[Dict], cards2: List[Dict]) -> List[Dict]:
        """
        Merge card detections from different methods, removing duplicates.
        
        Args:
            cards1: First list of card detections
            cards2: Second list of card detections
            
        Returns:
            Merged list of card detections
        """
        merged_cards = cards1.copy()
        
        for card2 in cards2:
            is_duplicate = False
            
            for i, card1 in enumerate(merged_cards):
                # Check if cards overlap significantly
                x1, y1, w1, h1 = card1["x"], card1["y"], card1["width"], card1["height"]
                x2, y2, w2, h2 = card2["x"], card2["y"], card2["width"], card2["height"]
                
                # Compute intersection
                x_intersect = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_intersect = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                intersection = x_intersect * y_intersect
                
                # Compute areas
                area1 = w1 * h1
                area2 = w2 * h2
                
                # If significant overlap
                if intersection > 0.5 * min(area1, area2):
                    # Keep the detection with higher confidence
                    if card2.get("confidence", 0) > card1.get("confidence", 0):
                        merged_cards[i] = card2
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_cards.append(card2)
        
        return merged_cards
    
    def _filter_card_candidates(self, cards: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
        """
        Filter out likely false positive card detections.
        
        Args:
            cards: List of detected cards
            frame_shape: Shape of the original frame (height, width)
            
        Returns:
            Filtered list of cards
        """
        frame_height, frame_width = frame_shape
        filtered_cards = []
        
        for card in cards:
            # Get card properties
            x, y, w, h = card["x"], card["y"], card["width"], card["height"]
            
            # Basic geometric checks
            # 1. Reasonable aspect ratio for a playing card
            aspect_ratio = h / w
            if not (0.5 <= aspect_ratio <= 2.0):
                continue
                
            # 2. Minimum size (avoid small noise)
            if w * h < 500:
                continue
                
            # 3. Card must be fully inside the frame
            if (x < 0 or y < 0 or 
                x + w > frame_width or 
                y + h > frame_height):
                continue
                
            # Additional checks for reasonable card placement
            # 4. Cards shouldn't be too close to frame edge
            border_margin = 2
            if (x <= border_margin or y <= border_margin or 
                x + w >= frame_width - border_margin or 
                y + h >= frame_height - border_margin):
                continue
            
            filtered_cards.append(card)
        
        return filtered_cards