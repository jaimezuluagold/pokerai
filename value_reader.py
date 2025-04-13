"""
Value Reader Module for Poker AI

This module handles the detection and reading of numerical values on the poker table,
such as pot size, player balances, and bet amounts.
"""

import cv2
import numpy as np
import re
import pytesseract
import logging
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValueReader:
    """
    Reads numerical values from the poker table interface.
    """
    
    def __init__(self, 
                 tesseract_path: Optional[str] = None,
                 preprocessing_level: str = "medium",
                 use_tesseract: bool = True,
                 confidence_threshold: float = 0.6):
        """
        Initialize the value reader.
        
        Args:
            tesseract_path: Path to Tesseract executable
            preprocessing_level: Level of preprocessing ("low", "medium", "high")
            use_tesseract: Whether to use Tesseract OCR
            confidence_threshold: Confidence threshold for OCR results
        """
        self.use_tesseract = use_tesseract
        self.confidence_threshold = confidence_threshold
        self.preprocessing_level = preprocessing_level
        
        # Set Tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # OCR configuration for numbers
        self.tesseract_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist="$0123456789,."'
        
        # Cache of recent results (region hash -> detected value)
        self.cache = {}
        
    def read_value(self, image: np.ndarray, region_name: str = "unknown") -> Dict:
        """
        Read a numerical value from an image region.
        
        Args:
            image: Input image containing a value
            region_name: Name of the region for logging
            
        Returns:
            Dictionary with detected value and metadata
        """
        # Check cache first
        img_hash = hash(image.tobytes())
        if img_hash in self.cache:
            return self.cache[img_hash]
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Resize if too small
        if gray.shape[0] < 20 or gray.shape[1] < 40:
            scaling_factor = max(20 / gray.shape[0], 40 / gray.shape[1])
            gray = cv2.resize(gray, (0, 0), fx=scaling_factor, fy=scaling_factor)
            
        # Apply preprocessing based on level
        processed = self._preprocess_image(gray)
        
        # Initialize result
        result = {
            "raw_text": None,
            "numeric_value": None,
            "has_dollar_sign": False,
            "confidence": 0.0,
            "success": False
        }
        
        if self.use_tesseract:
            try:
                # Perform OCR
                ocr_result = pytesseract.image_to_data(
                    processed, 
                    config=self.tesseract_config, 
                    output_type=pytesseract.Output.DICT
                )
                
                # Process OCR results
                highest_conf = 0
                best_text = ""
                
                for i in range(len(ocr_result["text"])):
                    text = ocr_result["text"][i]
                    conf = ocr_result["conf"][i]
                    
                    if conf > highest_conf and text.strip():
                        highest_conf = conf
                        best_text = text
                
                if highest_conf > self.confidence_threshold:
                    result["raw_text"] = best_text
                    result["confidence"] = highest_conf / 100.0  # Normalize to 0-1
                    
                    # Process the raw text
                    self._process_raw_text(result)
            except Exception as e:
                logger.error(f"OCR error: {e}")
                
        # If OCR failed or wasn't used, try digit contour detection
        if not result["success"]:
            digit_result = self._detect_digits_by_contours(processed)
            
            if digit_result["success"]:
                result = digit_result
                
        # Cache the result
        self.cache[img_hash] = result
        
        return result
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Preprocessed image
        """
        if self.preprocessing_level == "low":
            # Minimal preprocessing
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
            
        elif self.preprocessing_level == "high":
            # Aggressive preprocessing
            # Resize for better OCR
            image = cv2.resize(image, (0, 0), fx=2, fy=2)
            
            # Apply multiple filters
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Invert if needed (text should be black on white for OCR)
            if np.mean(opening) > 127:
                opening = cv2.bitwise_not(opening)
                
            return opening
            
        else:  # Medium (default)
            # Moderate preprocessing
            # Apply bilateral filter to preserve edges
            blurred = cv2.bilateralFilter(image, 9, 75, 75)
            
            # OTSU threshold
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if needed (text should be black on white for OCR)
            if np.mean(thresh) > 127:
                thresh = cv2.bitwise_not(thresh)
                
            return thresh
    
    def _process_raw_text(self, result: Dict) -> None:
        """
        Process OCR raw text to extract numeric value.
        
        Args:
            result: Result dictionary to be updated
        """
        if not result["raw_text"]:
            return
            
        raw_text = result["raw_text"].strip()
        
        # Check for dollar sign
        if '$' in raw_text:
            result["has_dollar_sign"] = True
            
        # Remove non-numeric characters except decimal point and comma
        numeric_str = re.sub(r'[^0-9.,]', '', raw_text)
        
        # Handle commas in numbers
        numeric_str = numeric_str.replace(',', '')
        
        # Convert to float if possible
        try:
            if numeric_str:
                numeric_value = float(numeric_str)
                result["numeric_value"] = numeric_value
                result["success"] = True
        except ValueError:
            logger.debug(f"Could not convert '{numeric_str}' to float")
    
    def _detect_digits_by_contours(self, image: np.ndarray) -> Dict:
        """
        Detect digits using contour analysis as a fallback method.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Result dictionary
        """
        result = {
            "raw_text": None,
            "numeric_value": None,
            "has_dollar_sign": False,
            "confidence": 0.0,
            "success": False
        }
        
        # Ensure binary image
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by x-coordinate (left-to-right reading)
        def get_contour_x(contour):
            x, _, _, _ = cv2.boundingRect(contour)
            return x
            
        sorted_contours = sorted(contours, key=get_contour_x)
        
        # Filter out small contours
        min_area = binary.shape[0] * binary.shape[1] * 0.001
        filtered_contours = [c for c in sorted_contours if cv2.contourArea(c) > min_area]
        
        # Check for dollar sign
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = binary[y:y+h, x:x+w]
            
            # Dollar sign typically has this aspect ratio and structure
            if 0.4 <= w/h <= 0.6:
                # Additional check for $ shape
                result["has_dollar_sign"] = True
                break
        
        # Skip to next attempt if no contours found
        if not filtered_contours:
            return result
            
        # This is a very basic digit recognition by contour shape
        # A full implementation would use a digit classifier
        raw_text = ""
        confidence_sum = 0
        
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            extent = cv2.contourArea(contour) / (w * h) if w * h > 0 else 0
            
            # Very basic shape-based recognition (low accuracy)
            # This should be replaced with a proper digit classifier
            if 0.8 <= aspect_ratio <= 1.2 and extent > 0.6:  # Like 0, 8
                raw_text += "0"
                confidence_sum += 0.5
            elif aspect_ratio > 1.8:  # Like 1
                raw_text += "1"
                confidence_sum += 0.7
            else:  # Default to a common digit
                raw_text += "5"
                confidence_sum += 0.3
        
        if raw_text:
            result["raw_text"] = raw_text
            result["confidence"] = confidence_sum / len(raw_text) if raw_text else 0
            
            # Try to convert to numeric
            try:
                result["numeric_value"] = float(raw_text)
                result["success"] = True
            except ValueError:
                logger.debug(f"Could not convert contour-detected '{raw_text}' to float")
        
        return result
    
    def read_pot_size(self, pot_image: np.ndarray) -> float:
        """
        Specifically read the pot size.
        
        Args:
            pot_image: Image of the pot area
            
        Returns:
            Pot size as a float, or 0 if not detected
        """
        result = self.read_value(pot_image, "pot")
        
        # Extra processing specific to pot display
        if result["success"] and result["numeric_value"] is not None:
            return result["numeric_value"]
            
        return 0.0
    
    def read_player_balance(self, balance_image: np.ndarray) -> float:
        """
        Read a player's balance.
        
        Args:
            balance_image: Image of the player balance area
            
        Returns:
            Player balance as a float, or 0 if not detected
        """
        result = self.read_value(balance_image, "player_balance")
        
        # Balance often has a $ sign and possibly comma separators
        if result["success"] and result["numeric_value"] is not None:
            return result["numeric_value"]
            
        return 0.0
    
    def read_bet_amount(self, bet_image: np.ndarray) -> float:
        """
        Read a bet amount.
        
        Args:
            bet_image: Image of the bet amount area
            
        Returns:
            Bet amount as a float, or 0 if not detected
        """
        result = self.read_value(bet_image, "bet_amount")
        
        if result["success"] and result["numeric_value"] is not None:
            return result["numeric_value"]
            
        return 0.0
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self.cache = {}