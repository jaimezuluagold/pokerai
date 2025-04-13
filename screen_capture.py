"""
Screen Capture Module for Poker AI

This module handles screen capture functionality, providing real-time images 
of the poker game interface for processing by other components.
"""

import numpy as np
import mss
import mss.tools
import time
import cv2
from typing import Dict, Tuple, Optional, List, Union


class ScreenCapture:
    """
    Handles screen capture operations for the poker game.
    Optimized for performance with configurable regions and change detection.
    """
    
    def __init__(self, 
                 monitor_number: int = 1, 
                 regions: Optional[Dict[str, Dict[str, int]]] = None,
                 capture_interval: float = 0.1,
                 detect_changes: bool = True,
                 change_threshold: float = 0.01):
        """
        Initialize the screen capture module.
        
        Args:
            monitor_number: The monitor to capture (1-based index)
            regions: Dictionary of named regions to capture {name: {top, left, width, height}}
            capture_interval: Minimum time between captures in seconds
            detect_changes: Whether to detect changes between frames
            change_threshold: Threshold for considering a change significant (0-1)
        """
        self.sct = mss.mss()
        self.monitor_number = monitor_number
        self.full_monitor = self.sct.monitors[monitor_number]
        
        # Default to full monitor if no regions specified
        self.regions = regions or {"full_screen": {
            "top": self.full_monitor["top"],
            "left": self.full_monitor["left"],
            "width": self.full_monitor["width"],
            "height": self.full_monitor["height"]
        }}
        
        self.capture_interval = capture_interval
        self.last_capture_time = 0
        self.last_frames = {}
        self.detect_changes = detect_changes
        self.change_threshold = change_threshold
        
        # Initialize last frames for each region
        for region_name in self.regions:
            self.last_frames[region_name] = None
    
    def capture(self, region_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Capture screen regions.
        
        Args:
            region_name: Name of specific region to capture, or None for all regions
        
        Returns:
            Dictionary of captured frames by region name
        """
        current_time = time.time()
        
        # Throttle capture rate
        if current_time - self.last_capture_time < self.capture_interval:
            time.sleep(self.capture_interval - (current_time - self.last_capture_time))
        
        self.last_capture_time = time.time()
        
        # Determine which regions to capture
        regions_to_capture = [region_name] if region_name else list(self.regions.keys())
        captured_frames = {}
        
        for name in regions_to_capture:
            if name not in self.regions:
                raise ValueError(f"Region '{name}' not defined")
            
            region = self.regions[name]
            sct_img = self.sct.grab(region)
            
            # Convert to numpy array (RGB format)
            frame = np.array(sct_img)
            
            # Convert from BGRA to BGR (OpenCV format)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Check if there's a significant change in the frame
            if self.detect_changes and self.last_frames[name] is not None:
                if not self._has_changed(self.last_frames[name], frame):
                    captured_frames[name] = self.last_frames[name]
                    continue
            
            # Store frame
            self.last_frames[name] = frame
            captured_frames[name] = frame
        
        return captured_frames
    
    def _has_changed(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        """
        Detect if there's a significant change between frames.
        
        Args:
            prev_frame: Previous captured frame
            curr_frame: Current captured frame
            
        Returns:
            True if significant change detected, False otherwise
        """
        # Convert to grayscale for faster comparison
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of changed pixels
        change_percent = np.count_nonzero(thresh) / thresh.size
        
        return change_percent > self.change_threshold
    
    def update_regions(self, regions: Dict[str, Dict[str, int]]) -> None:
        """
        Update capture regions dynamically.
        
        Args:
            regions: New set of regions
        """
        self.regions.update(regions)
        
        # Initialize last frames for new regions
        for region_name in regions:
            if region_name not in self.last_frames:
                self.last_frames[region_name] = None
    
    def auto_detect_game_window(self) -> Optional[Dict[str, int]]:
        """
        Attempt to automatically detect the poker game window.
        
        Returns:
            Region dict if successful, None otherwise
        """
        # Capture full screen
        full_screen = self.capture("full_screen")["full_screen"]
        
        # Convert to grayscale
        gray = cv2.cvtColor(full_screen, cv2.COLOR_BGR2GRAY)
        
        # Look for distinctive elements of the poker table (green felt)
        # This is a simplified approach - may need refinement based on the specific game
        mask = cv2.inRange(
            cv2.cvtColor(full_screen, cv2.COLOR_BGR2HSV),
            np.array([35, 50, 50]),  # Lower green boundary
            np.array([85, 255, 255])  # Upper green boundary
        )
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (likely the poker table)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Only accept if it's a reasonable size (at least 30% of screen)
            if (w * h) > (full_screen.shape[0] * full_screen.shape[1] * 0.3):
                game_region = {
                    "top": y + self.full_monitor["top"],
                    "left": x + self.full_monitor["left"],
                    "width": w,
                    "height": h
                }
                
                self.update_regions({"game_window": game_region})
                return game_region
        
        return None
    
    def get_sub_regions(self, base_region: str) -> Dict[str, Dict[str, int]]:
        """
        Divide a game window into logical sub-regions for card/button detection.
        
        Args:
            base_region: Name of the base region (usually "game_window")
            
        Returns:
            Dictionary of sub-regions
        """
        if base_region not in self.regions:
            raise ValueError(f"Base region '{base_region}' not defined")
        
        base = self.regions[base_region]
        width, height = base["width"], base["height"]
        
        # Define relative positions
        # These values will need tuning based on the specific poker interface
        sub_regions = {
            "player_cards": {
                "top": base["top"] + int(height * 0.7),
                "left": base["left"] + int(width * 0.4),
                "width": int(width * 0.2),
                "height": int(height * 0.2)
            },
            "community_cards": {
                "top": base["top"] + int(height * 0.4),
                "left": base["left"] + int(width * 0.3),
                "width": int(width * 0.4),
                "height": int(height * 0.2)
            },
            "pot": {
                "top": base["top"] + int(height * 0.3),
                "left": base["left"] + int(width * 0.45),
                "width": int(width * 0.1),
                "height": int(height * 0.1)
            },
            "action_buttons": {
                "top": base["top"] + int(height * 0.85),
                "left": base["left"] + int(width * 0.1),
                "width": int(width * 0.8),
                "height": int(height * 0.1)
            }
        }
        
        return sub_regions
    
    def close(self) -> None:
        """Release resources."""
        self.sct.close()