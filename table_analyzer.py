"""
Table Analyzer Module for Poker AI

This module integrates all recognition components to analyze the complete poker table.
It provides a comprehensive assessment of the current game state, including cards,
player information, and possible actions.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import time

from src.recognition.card_detector import CardDetector
from src.recognition.card_classifier import CardClassifier
from src.recognition.value_reader import ValueReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableAnalyzer:
    """
    Analyzes the poker table to extract game state information.
    """
    
    def __init__(self,
                 card_detector: Optional[CardDetector] = None,
                 card_classifier: Optional[CardClassifier] = None,
                 value_reader: Optional[ValueReader] = None,
                 config: Dict = None):
        """
        Initialize the table analyzer.
        
        Args:
            card_detector: Card detector component
            card_classifier: Card classifier component
            value_reader: Value reader component
            config: Configuration dictionary
        """
        # Initialize components
        self.card_detector = card_detector or CardDetector()
        self.card_classifier = card_classifier or CardClassifier()
        self.value_reader = value_reader or ValueReader()
        
        # Configuration
        self.config = config or {}
        
        # Default region definitions (percentages of table width/height)
        self.default_regions = {
            "community_cards": {"top": 0.4, "left": 0.3, "width": 0.4, "height": 0.2},
            "player_cards": {"top": 0.7, "left": 0.4, "width": 0.2, "height": 0.2},
            "pot": {"top": 0.3, "left": 0.45, "width": 0.1, "height": 0.1},
            "player_balance": {"top": 0.8, "left": 0.45, "width": 0.1, "height": 0.05},
            "action_buttons": {"top": 0.85, "left": 0.1, "width": 0.8, "height": 0.1},
            # Define regions for each opponent position
            "opponent1": {"top": 0.1, "left": 0.1, "width": 0.2, "height": 0.2},
            "opponent2": {"top": 0.1, "left": 0.7, "width": 0.2, "height": 0.2},
            "opponent3": {"top": 0.5, "left": 0.1, "width": 0.2, "height": 0.2},
            "opponent4": {"top": 0.5, "left": 0.7, "width": 0.2, "height": 0.2}
        }
        
        # User-defined regions (absolute coordinates)
        self.regions = {}
        
        # Game state information
        self.current_state = self._initialize_game_state()
        
        # Cache for optimization
        self.frame_cache = None
        self.last_analysis_time = 0
        self.analysis_interval = 0.5  # seconds
    
    def _initialize_game_state(self) -> Dict:
        """
        Initialize the game state structure.
        
        Returns:
            Empty game state dictionary
        """
        return {
            "community_cards": [],
            "player_cards": [],
            "pot": 0.0,
            "player_balance": 0.0,
            "opponents": {},
            "active_player": None,
            "dealer_position": None,
            "current_bet": 0.0,
            "available_actions": [],
            "game_phase": None,  # pre-flop, flop, turn, river
            "timestamp": 0
        }
    
    def set_regions(self, regions: Dict[str, Dict[str, int]]) -> None:
        """
        Set the table regions for analysis.
        
        Args:
            regions: Dictionary of region definitions
        """
        self.regions = regions
    
    def _get_region_coords(self, frame: np.ndarray, region_name: str) -> Tuple[int, int, int, int]:
        """
        Get absolute coordinates for a region.
        
        Args:
            frame: Full table frame
            region_name: Name of the region
            
        Returns:
            Tuple of (x, y, width, height)
        """
        h, w = frame.shape[:2]
        
        # Check if region is defined with absolute coordinates
        if region_name in self.regions:
            region = self.regions[region_name]
            return (
                region["left"],
                region["top"],
                region["width"],
                region["height"]
            )
        
        # Fall back to default regions (percentage-based)
        if region_name in self.default_regions:
            region = self.default_regions[region_name]
            return (
                int(region["left"] * w),
                int(region["top"] * h),
                int(region["width"] * w),
                int(region["height"] * h)
            )
        
        # Region not found
        logger.warning(f"Region '{region_name}' not defined")
        return (0, 0, w, h)  # Return full frame as fallback
    
    def _extract_region(self, frame: np.ndarray, region_name: str) -> np.ndarray:
        """
        Extract a specific region from the frame.
        
        Args:
            frame: Full table frame
            region_name: Name of the region
            
        Returns:
            Image of the specified region
        """
        x, y, width, height = self._get_region_coords(frame, region_name)
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        width = max(1, min(width, w - x))
        height = max(1, min(height, h - y))
        
        return frame[y:y+height, x:x+width].copy()
    
    def analyze_table(self, frame: np.ndarray, force_analysis: bool = False) -> Dict:
        """
        Analyze the poker table to extract current game state.
        
        Args:
            frame: Full table frame
            force_analysis: Whether to force a new analysis
            
        Returns:
            Current game state
        """
        current_time = time.time()
        
        # Check if we can use cached results
        if (not force_analysis and 
            self.frame_cache is not None and 
            current_time - self.last_analysis_time < self.analysis_interval):
            return self.current_state
            
        # Store frame for cache
        self.frame_cache = frame.copy()
        self.last_analysis_time = current_time
        
        # Reset game state
        self.current_state = self._initialize_game_state()
        self.current_state["timestamp"] = current_time
        
        try:
            # Analyze community cards
            community_region = self._extract_region(frame, "community_cards")
            self._analyze_community_cards(community_region)
            
            # Analyze player cards
            player_cards_region = self._extract_region(frame, "player_cards")
            self._analyze_player_cards(player_cards_region)
            
            # Read pot size
            pot_region = self._extract_region(frame, "pot")
            self._analyze_pot(pot_region)
            
            # Read player balance
            balance_region = self._extract_region(frame, "player_balance")
            self._analyze_player_balance(balance_region)
            
            # Analyze opponents
            self._analyze_opponents(frame)
            
            # Identify available actions
            actions_region = self._extract_region(frame, "action_buttons")
            self._analyze_available_actions(actions_region)
            
            # Determine game phase
            self._determine_game_phase()
            
        except Exception as e:
            logger.error(f"Error during table analysis: {e}")
        
        return self.current_state
    
    def _analyze_community_cards(self, region: np.ndarray) -> None:
        """
        Analyze community cards region to identify cards.
        
        Args:
            region: Image of community cards region
        """
        # Detect cards in the region
        cards = self.card_detector.detect_cards(region, "community")
        
        community_cards = []
        for card in cards:
            card_img = card["image"]
            
            # Classify each detected card
            classification = self.card_classifier.classify_card(card_img)
            
            if classification["identified"]:
                card_info = {
                    "value": classification["value"],
                    "suit": classification["suit"],
                    "code": classification["card_code"],
                    "confidence": min(classification["confidence_value"], 
                                  classification["confidence_suit"])
                }
                community_cards.append(card_info)
        
        # Sort by x-position to maintain left-to-right order
        community_cards.sort(key=lambda c: cards[community_cards.index(c)]["x"])
        
        # Update game state
        self.current_state["community_cards"] = community_cards
        
        logger.info(f"Detected {len(community_cards)} community cards")
    
    def _analyze_player_cards(self, region: np.ndarray) -> None:
        """
        Analyze player cards region to identify cards.
        
        Args:
            region: Image of player cards region
        """
        # Detect cards in the region
        cards = self.card_detector.detect_cards(region, "player")
        
        player_cards = []
        for card in cards:
            card_img = card["image"]
            
            # Classify each detected card
            classification = self.card_classifier.classify_card(card_img)
            
            if classification["identified"]:
                card_info = {
                    "value": classification["value"],
                    "suit": classification["suit"],
                    "code": classification["card_code"],
                    "confidence": min(classification["confidence_value"], 
                                  classification["confidence_suit"])
                }
                player_cards.append(card_info)
        
        # Sort by x-position to maintain left-to-right order
        player_cards.sort(key=lambda c: cards[player_cards.index(c)]["x"])
        
        # Update game state
        self.current_state["player_cards"] = player_cards
        
        logger.info(f"Detected {len(player_cards)} player cards")
    
    def _analyze_pot(self, region: np.ndarray) -> None:
        """
        Analyze pot region to determine pot size.
        
        Args:
            region: Image of pot region
        """
        pot_size = self.value_reader.read_pot_size(region)
        
        # Update game state
        self.current_state["pot"] = pot_size
        
        logger.info(f"Detected pot size: {pot_size}")
    
    def _analyze_player_balance(self, region: np.ndarray) -> None:
        """
        Analyze player balance region.
        
        Args:
            region: Image of player balance region
        """
        balance = self.value_reader.read_player_balance(region)
        
        # Update game state
        self.current_state["player_balance"] = balance
        
        logger.info(f"Detected player balance: {balance}")
    
    def _analyze_opponents(self, frame: np.ndarray) -> None:
        """
        Analyze opponent positions to gather information.
        
        Args:
            frame: Full table frame
        """
        opponents = {}
        
        # Process each opponent position
        for i in range(1, 5):  # Up to 4 opponents
            region_name = f"opponent{i}"
            
            try:
                # Extract opponent region
                opponent_region = self._extract_region(frame, region_name)
                
                # Check if there's an opponent in this position
                # This could be done by checking for cards, chips, avatar, etc.
                hsv = cv2.cvtColor(opponent_region, cv2.COLOR_BGR2HSV)
                
                # Look for blue cards (opponent's hidden cards)
                lower_blue = np.array([100, 50, 50])
                upper_blue = np.array([130, 255, 255])
                blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                
                # If significant blue detected, assume opponent present
                if np.sum(blue_mask) > 1000:
                    # Extract balance information
                    balance_region = opponent_region[
                        int(opponent_region.shape[0] * 0.7):int(opponent_region.shape[0] * 0.9),
                        int(opponent_region.shape[1] * 0.3):int(opponent_region.shape[1] * 0.7)
                    ]
                    
                    balance = self.value_reader.read_player_balance(balance_region)
                    
                    # Check for dealer button
                    dealer_button = self._check_for_dealer_button(opponent_region)
                    
                    # Check for active player indicator
                    is_active = self._check_if_active_player(opponent_region)
                    
                    # Store opponent information
                    opponents[region_name] = {
                        "position": i,
                        "balance": balance,
                        "has_cards": True,
                        "is_dealer": dealer_button,
                        "is_active": is_active,
                        "current_bet": 0.0  # To be implemented
                    }
                    
                    # Update game state
                    if dealer_button:
                        self.current_state["dealer_position"] = i
                    
                    if is_active:
                        self.current_state["active_player"] = i
            except Exception as e:
                logger.error(f"Error analyzing opponent {i}: {e}")
        
        # Update game state
        self.current_state["opponents"] = opponents
        
        logger.info(f"Detected {len(opponents)} opponents")
    
    def _check_for_dealer_button(self, region: np.ndarray) -> bool:
        """
        Check if dealer button is present in a region.
        
        Args:
            region: Image region to check
            
        Returns:
            True if dealer button detected, False otherwise
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Define color range for dealer button (often a white or bright colored disc)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Check for circular shapes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Check if contour is roughly circular
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filter small contours
            if area < 100:
                continue
                
            # Calculate circularity (perfect circle = 1)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            if circularity > 0.7:  # Threshold for circularity
                # Look for "D" inside the button
                # This would require OCR or template matching
                return True
        
        return False
    
    def _check_if_active_player(self, region: np.ndarray) -> bool:
        """
        Check if a player position shows active player indicators.
        
        Args:
            region: Image region to check
            
        Returns:
            True if active player indicators detected, False otherwise
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Look for highlighting or glow effects (often yellowish)
        lower_highlight = np.array([20, 100, 200])
        upper_highlight = np.array([40, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_highlight, upper_highlight)
        
        # Check if there's significant highlighting
        highlight_ratio = np.sum(mask > 0) / (region.shape[0] * region.shape[1])
        
        return highlight_ratio > 0.05  # Threshold for considering it highlighted
    
    def _analyze_available_actions(self, region: np.ndarray) -> None:
        """
        Analyze action buttons region to identify available actions.
        
        Args:
            region: Image of action buttons region
        """
        available_actions = []
        
        # Define button templates (in a real implementation, use actual templates)
        buttons = {
            "fold": {"color": (50, 50, 50), "text": "Fold"},
            "check": {"color": (50, 50, 50), "text": "Check"},
            "call": {"color": (50, 50, 50), "text": "Call"},
            "raise": {"color": (50, 50, 50), "text": "Raise"},
            "all_in": {"color": (50, 50, 50), "text": "All In"}
        }
        
        # Simple implementation: split the region horizontally for buttons
        button_width = region.shape[1] // 3  # Assuming 3 buttons
        
        for i, action in enumerate(["fold", "check", "raise"]):
            button_region = region[:, i * button_width:(i + 1) * button_width]
            
            # Check if button is enabled (not grayed out)
            # For a real implementation, use template matching or OCR
            gray = cv2.cvtColor(button_region, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # If button is bright enough, consider it enabled
            if mean_brightness > 50:
                available_actions.append(action)
                
                # Special case: if "check" is not available, it might be "call"
                if action == "check" and action not in available_actions:
                    # Additional check for "call" text using OCR would go here
                    available_actions.append("call")
        
        # Update game state
        self.current_state["available_actions"] = available_actions
        
        logger.info(f"Detected available actions: {available_actions}")
    
    def _determine_game_phase(self) -> None:
        """
        Determine the current phase of the game based on community cards.
        """
        num_community_cards = len(self.current_state["community_cards"])
        
        if num_community_cards == 0:
            phase = "pre-flop"
        elif num_community_cards == 3:
            phase = "flop"
        elif num_community_cards == 4:
            phase = "turn"
        elif num_community_cards == 5:
            phase = "river"
        else:
            phase = "unknown"
        
        # Update game state
        self.current_state["game_phase"] = phase
        
        logger.info(f"Detected game phase: {phase}")
    
    def get_current_state(self) -> Dict:
        """
        Get the current game state.
        
        Returns:
            Current game state dictionary
        """
        return self.current_state
    
    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Perform calibration to auto-detect regions.
        
        Args:
            frame: Full table frame
            
        Returns:
            True if calibration succeeded, False otherwise
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Look for green felt (poker table)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            
            # Create mask
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.error("Could not find poker table (green felt)")
                return False
                
            # Find the largest contour (poker table)
            table_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(table_contour)
            
            # Define regions based on table bounds
            self.regions = {
                "table": {"top": y, "left": x, "width": w, "height": h},
                "community_cards": {
                    "top": y + int(h * 0.4),
                    "left": x + int(w * 0.3),
                    "width": int(w * 0.4),
                    "height": int(h * 0.2)
                },
                "player_cards": {
                    "top": y + int(h * 0.7),
                    "left": x + int(w * 0.4),
                    "width": int(w * 0.2),
                    "height": int(h * 0.2)
                },
                "pot": {
                    "top": y + int(h * 0.3),
                    "left": x + int(w * 0.45),
                    "width": int(w * 0.1),
                    "height": int(h * 0.1)
                },
                "player_balance": {
                    "top": y + int(h * 0.8),
                    "left": x + int(w * 0.45),
                    "width": int(w * 0.1),
                    "height": int(h * 0.05)
                },
                "action_buttons": {
                    "top": y + int(h * 0.85),
                    "left": x + int(w * 0.1),
                    "width": int(w * 0.8),
                    "height": int(h * 0.1)
                }
            }
            
            # Define opponent regions
            self.regions["opponent1"] = {
                "top": y + int(h * 0.1),
                "left": x + int(w * 0.1),
                "width": int(w * 0.2),
                "height": int(h * 0.2)
            }
            
            self.regions["opponent2"] = {
                "top": y + int(h * 0.1),
                "left": x + int(w * 0.7),
                "width": int(w * 0.2),
                "height": int(h * 0.2)
            }
            
            self.regions["opponent3"] = {
                "top": y + int(h * 0.5),
                "left": x + int(w * 0.1),
                "width": int(w * 0.2),
                "height": int(h * 0.2)
            }
            
            self.regions["opponent4"] = {
                "top": y + int(h * 0.5),
                "left": x + int(w * 0.7),
                "width": int(w * 0.2),
                "height": int(h * 0.2)
            }
            
            logger.info("Table calibration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during calibration: {e}")
            return False