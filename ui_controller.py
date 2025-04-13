"""
UI Controller Module for Poker AI

This module handles interaction with the poker game interface,
automating clicks and other UI actions.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import pyautogui
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure PyAutoGUI
pyautogui.PAUSE = 0.5  # Add pause between actions
pyautogui.FAILSAFE = True  # Enable fail-safe (move mouse to corner to abort)


class UIController:
    """
    Controls UI interaction with the poker game interface.
    """
    
    def __init__(self, 
                 button_regions: Optional[Dict[str, Dict[str, int]]] = None,
                 input_regions: Optional[Dict[str, Dict[str, int]]] = None,
                 click_variation: int = 5,
                 action_delay: float = 0.5,
                 human_like: bool = True):
        """
        Initialize the UI controller.
        
        Args:
            button_regions: Dictionary of button regions
            input_regions: Dictionary of input regions
            click_variation: Random variation in click position (pixels)
            action_delay: Delay between actions (seconds)
            human_like: Whether to use human-like mouse movements
        """
        self.button_regions = button_regions or {}
        self.input_regions = input_regions or {}
        self.click_variation = click_variation
        self.action_delay = action_delay
        self.human_like = human_like
        
        # Track action history
        self.action_history = []
        self.max_history = 100
        
        # Last detected state
        self.last_state = None
        
    def set_regions(self, 
                   button_regions: Dict[str, Dict[str, int]],
                   input_regions: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        """
        Set UI regions for buttons and inputs.
        
        Args:
            button_regions: Dictionary of button regions
            input_regions: Dictionary of input regions
        """
        self.button_regions = button_regions
        if input_regions:
            self.input_regions = input_regions
            
        logger.info(f"UI regions updated: {len(button_regions)} buttons, {len(input_regions or {})} inputs")
    
    def perform_action(self, 
                      action: str, 
                      amount: Optional[float] = None,
                      verify: bool = True) -> bool:
        """
        Perform an action in the poker UI.
        
        Args:
            action: Action to perform ("fold", "check", "call", "raise")
            amount: Bet amount for raise
            verify: Whether to verify the action was successful
            
        Returns:
            True if action was performed successfully, False otherwise
        """
        # Check if action is valid
        if action not in ["fold", "check", "call", "raise"]:
            logger.error(f"Invalid action: {action}")
            return False
            
        # Check if button region is defined
        if action not in self.button_regions:
            logger.error(f"Button region not defined for action: {action}")
            return False
            
        # Record action
        self._record_action(action, amount)
        
        # Perform the action
        success = False
        
        if action == "raise" and amount is not None:
            # First set the amount, then click the raise button
            success = self._set_bet_amount(amount) and self._click_button(action)
        else:
            # Just click the button
            success = self._click_button(action)
            
        # Verify action if requested
        if verify and success:
            verification_success = self._verify_action(action, amount)
            if not verification_success:
                logger.warning(f"Action verification failed: {action}")
                return False
                
        return success
    
    def _click_button(self, button_name: str) -> bool:
        """
        Click a button in the UI.
        
        Args:
            button_name: Name of the button to click
            
        Returns:
            True if click was successful, False otherwise
        """
        if button_name not in self.button_regions:
            logger.error(f"Button region not defined: {button_name}")
            return False
            
        # Get button region
        region = self.button_regions[button_name]
        
        # Calculate center
        center_x = region["left"] + region["width"] // 2
        center_y = region["top"] + region["height"] // 2
        
        # Add random variation if human-like
        if self.human_like:
            center_x += random.randint(-self.click_variation, self.click_variation)
            center_y += random.randint(-self.click_variation, self.click_variation)
            
        try:
            # Move to button with human-like movement
            if self.human_like:
                pyautogui.moveTo(center_x, center_y, duration=0.2)
            else:
                pyautogui.moveTo(center_x, center_y)
                
            # Click the button
            pyautogui.click()
            
            # Add delay
            time.sleep(self.action_delay)
            
            logger.info(f"Clicked button: {button_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clicking button {button_name}: {e}")
            return False
    
    def _set_bet_amount(self, amount: float) -> bool:
        """
        Set a bet amount in the raise input field.
        
        Args:
            amount: Bet amount
            
        Returns:
            True if successful, False otherwise
        """
        if "raise_input" not in self.input_regions:
            logger.error("Raise input region not defined")
            return False
            
        # Get input region
        region = self.input_regions["raise_input"]
        
        # Calculate click point (center of input field)
        input_x = region["left"] + region["width"] // 2
        input_y = region["top"] + region["height"] // 2
        
        try:
            # Move to input field
            if self.human_like:
                pyautogui.moveTo(input_x, input_y, duration=0.2)
            else:
                pyautogui.moveTo(input_x, input_y)
                
            # Triple click to select all existing text
            pyautogui.click(clicks=3)
            
            # Clear field (delete key) as fallback
            pyautogui.press('delete')
            
            # Convert amount to string
            amount_str = str(int(amount))  # Convert to integer for cleaner input
            
            # Type the amount
            pyautogui.typewrite(amount_str)
            
            # Add delay
            time.sleep(self.action_delay)
            
            logger.info(f"Set bet amount: {amount}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting bet amount: {e}")
            return False
    
    def _verify_action(self, 
                      action: str, 
                      amount: Optional[float] = None) -> bool:
        """
        Verify that an action was carried out successfully.
        
        Args:
            action: Action that was performed
            amount: Bet amount if applicable
            
        Returns:
            True if verification succeeded, False otherwise
        """
        # Add a small delay for the UI to update
        time.sleep(1.0)
        
        # Basic verification - check if the button is still present/active
        # This is a simple approximation; a more sophisticated version would
        # analyze the game state to verify the action took effect
        
        if action in self.button_regions:
            # Take a screenshot of the button region
            region = self.button_regions[action]
            try:
                screenshot = pyautogui.screenshot(
                    region=(
                        region["left"], 
                        region["top"], 
                        region["width"], 
                        region["height"]
                    )
                )
                
                # Check if the button appears disabled (e.g., grayed out)
                # This is a simplified approach and may need adjustment for specific games
                gray_level = self._get_average_brightness(screenshot)
                
                # If the button is now darker/grayer, assume it's been clicked and disabled
                if gray_level < 100:  # Threshold may need adjustment
                    logger.info(f"Action verification succeeded: {action}")
                    return True
                    
                logger.warning(f"Button still appears active after action: {action}")
                return False
                
            except Exception as e:
                logger.error(f"Error during action verification: {e}")
                return False
                
        # If we can't verify, assume success
        return True
    
    def _get_average_brightness(self, image) -> float:
        """
        Calculate the average brightness of an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Average brightness (0-255)
        """
        # Convert to grayscale
        gray = image.convert('L')
        
        # Calculate average brightness
        pixels = list(gray.getdata())
        return sum(pixels) / len(pixels) if pixels else 0
    
    def find_buttons(self) -> Dict[str, Dict[str, int]]:
        """
        Attempt to automatically find poker buttons on screen.
        
        Returns:
            Dictionary of button regions
        """
        # This is a placeholder for more sophisticated button detection
        # In a real implementation, this would use image processing to locate buttons
        
        logger.info("Attempting to find poker buttons")
        
        button_names = ["fold", "check", "call", "raise"]
        found_buttons = {}
        
        # Placeholder implementation - not functional
        # In a real implementation, this would:
        # 1. Take a screenshot
        # 2. Use template matching or OCR to find buttons
        # 3. Return their locations
        
        screen_width, screen_height = pyautogui.size()
        
        # Example placeholder - assuming buttons are at bottom of screen
        button_height = int(screen_height * 0.05)
        button_width = int(screen_width * 0.15)
        button_top = int(screen_height * 0.9)
        
        # Arbitrary placement of buttons
        for i, button in enumerate(button_names):
            left = int(screen_width * 0.1) + (i * button_width * 1.2)
            
            found_buttons[button] = {
                "left": left,
                "top": button_top,
                "width": button_width,
                "height": button_height
            }
            
        logger.warning("Button detection not fully implemented. Using estimated positions.")
        return found_buttons
    
    def calibrate_ui(self) -> bool:
        """
        Run a calibration procedure to locate UI elements.
        
        Returns:
            True if calibration succeeded, False otherwise
        """
        logger.info("Starting UI calibration")
        
        try:
            # Look for buttons
            detected_buttons = self.find_buttons()
            
            if detected_buttons:
                self.button_regions = detected_buttons
                
                # Estimate input region based on raise button
                if "raise" in detected_buttons:
                    raise_region = detected_buttons["raise"]
                    
                    # Assume input is above the raise button
                    self.input_regions["raise_input"] = {
                        "left": raise_region["left"],
                        "top": raise_region["top"] - raise_region["height"] - 10,
                        "width": raise_region["width"],
                        "height": raise_region["height"]
                    }
                    
                logger.info("UI calibration completed")
                return True
            else:
                logger.error("Failed to find UI elements during calibration")
                return False
                
        except Exception as e:
            logger.error(f"Error during UI calibration: {e}")
            return False
    
    def _record_action(self, action: str, amount: Optional[float] = None) -> None:
        """
        Record an action in the history.
        
        Args:
            action: Action performed
            amount: Bet amount if applicable
        """
        timestamp = time.time()
        
        self.action_history.append({
            "action": action,
            "amount": amount,
            "timestamp": timestamp
        })
        
        # Trim history if needed
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
    
    def get_action_history(self) -> List[Dict]:
        """
        Get the action history.
        
        Returns:
            List of action records
        """
        return self.action_history
    
    def mouse_to_safe_position(self) -> None:
        """Move the mouse to a safe position away from the game."""
        try:
            # Move to top-left corner
            screen_width, screen_height = pyautogui.size()
            safe_x = int(screen_width * 0.1)
            safe_y = int(screen_height * 0.1)
            
            pyautogui.moveTo(safe_x, safe_y, duration=0.5)
            logger.debug("Moved mouse to safe position")
            
        except Exception as e:
            logger.error(f"Error moving mouse to safe position: {e}")
    
    def set_human_like(self, enabled: bool) -> None:
        """
        Enable or disable human-like mouse movements.
        
        Args:
            enabled: Whether to enable human-like movements
        """
        self.human_like = enabled
        logger.info(f"Human-like mouse movements {'enabled' if enabled else 'disabled'}")
    
    def set_action_delay(self, delay: float) -> None:
        """
        Set the delay between actions.
        
        Args:
            delay: Delay in seconds
        """
        self.action_delay = max(0.1, min(5.0, delay))
        logger.info(f"Action delay set to {self.action_delay} seconds")