"""
Main Module for Poker AI

This module integrates all components of the poker AI system
and implements the main execution loop.
"""

import logging
import time
import argparse
import os
import yaml
import sys
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any

from src.capture.screen_capture import ScreenCapture
from src.recognition.table_analyzer import TableAnalyzer
from src.game_state.state_tracker import StateTracker
from src.game_state.hand_evaluator import HandEvaluator
from src.strategy.decision_engine import DecisionEngine
from src.strategy.opponent_tracker import OpponentTracker
from src.llm_interface.lava_connector import LavaConnector
from src.action.ui_controller import UIController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("poker_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")


class PokerAI:
    """
    Main class that integrates all poker AI components.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Poker AI system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.screen_capture = None
        self.table_analyzer = None
        self.state_tracker = None
        self.hand_evaluator = None
        self.decision_engine = None
        self.opponent_tracker = None
        self.llm_connector = None
        self.ui_controller = None
        
        # Running state
        self.running = False
        self.paused = False
        self.autonomous_mode = False
        self.current_state = {}
        
        # Initialize components with config
        self._initialize_components()
        
        logger.info("Poker AI initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "capture": {
                "interval": 0.5,
                "detect_changes": True
            },
            "strategy": {
                "profile": "balanced",
                "risk_tolerance": 1.0,
                "use_position": True,
                "bluff_factor": 0.1
            },
            "llm": {
                "api_base": "http://localhost:11434/api",
                "model_name": "lava",
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "ui": {
                "human_like": True,
                "action_delay": 0.5
            },
            "execution": {
                "loop_delay": 1.0,
                "max_hands": 0,  # 0 means unlimited
                "autonomous_mode": False
            },
            "logging": {
                "level": "INFO",
                "log_to_file": True
            }
        }
        
        # Try to load config file
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    
                # Merge with default config
                if user_config:
                    self._deep_update(default_config, user_config)
                    
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.info("Using default configuration")
            
        return default_config
    
    def _deep_update(self, d: Dict, u: Dict) -> None:
        """
        Deep update dictionary d with values from dictionary u.
        
        Args:
            d: Dictionary to update
            u: Dictionary with new values
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
    
    def _initialize_components(self) -> None:
        """Initialize all system components using configuration."""
        try:
            # Screen Capture
            capture_config = self.config.get("capture", {})
            self.screen_capture = ScreenCapture(
                capture_interval=capture_config.get("interval", 0.5),
                detect_changes=capture_config.get("detect_changes", True)
            )
            
            # Hand Evaluator
            self.hand_evaluator = HandEvaluator()
            
            # Table Analyzer
            self.table_analyzer = TableAnalyzer()
            
            # State Tracker
            log_config = self.config.get("logging", {})
            self.state_tracker = StateTracker(
                log_to_file=log_config.get("log_to_file", True)
            )
            
            # Opponent Tracker
            self.opponent_tracker = OpponentTracker()
            
            # Strategy components
            strategy_config = self.config.get("strategy", {})
            self.decision_engine = DecisionEngine(
                hand_evaluator=self.hand_evaluator,
                strategy_profile=strategy_config.get("profile", "balanced"),
                risk_tolerance=strategy_config.get("risk_tolerance", 1.0),
                use_position=strategy_config.get("use_position", True),
                bluff_factor=strategy_config.get("bluff_factor", 0.1)
            )
            
            # LLM Interface
            llm_config = self.config.get("llm", {})
            self.llm_connector = LavaConnector(
                api_base=llm_config.get("api_base", "http://localhost:11434/api"),
                model_name=llm_config.get("model_name", "lava"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 1024)
            )
            
            # UI Controller
            ui_config = self.config.get("ui", {})
            self.ui_controller = UIController(
                human_like=ui_config.get("human_like", True),
                action_delay=ui_config.get("action_delay", 0.5)
            )
            
            # Set autonomous mode
            exec_config = self.config.get("execution", {})
            self.autonomous_mode = exec_config.get("autonomous_mode", False)
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
def start(self) -> None:
    """Start the Poker AI system."""
    logger.info("Starting Poker AI system")
    
    # Manual configuration based on measurements
    manual_game_region = {
        "top": 487,
        "left": 51,
        "width": 1136,
        "height": 770
    }
    
    # Update regions manually
    self.screen_capture.update_regions({"game_window": manual_game_region})
    
    # Calculate sub-regions based on the manual game region
    game_h = manual_game_region["height"]
    game_w = manual_game_region["width"]
    game_x = manual_game_region["left"]
    game_y = manual_game_region["top"]
    
    sub_regions = {
        "community_cards": {
            "top": game_y + int(game_h * 0.4),
            "left": game_x + int(game_w * 0.3),
            "width": int(game_w * 0.4),
            "height": int(game_h * 0.2)
        },
        "player_cards": {
            "top": game_y + int(game_h * 0.7),
            "left": game_x + int(game_w * 0.4),
            "width": int(game_w * 0.2),
            "height": int(game_h * 0.2)
        },
        "action_buttons": {
            "top": game_y + int(game_h * 0.85),
            "left": game_x + int(game_w * 0.1),
            "width": int(game_w * 0.8),
            "height": int(game_h * 0.1)
        },
        "pot": {
            "top": game_y + int(game_h * 0.3),
            "left": game_x + int(game_w * 0.45),
            "width": int(game_w * 0.1),
            "height": int(game_h * 0.1)
        },
        "player_balance": {
            "top": game_y + int(game_h * 0.8),
            "left": game_x + int(game_w * 0.45),
            "width": int(game_w * 0.1),
            "height": int(game_h * 0.05)
        }
    }
    self.screen_capture.update_regions(sub_regions)
    logger.info("Manual regions configured")
    
    # Add debug to verify regions
    import cv2
    import os
    
    # Create debug directory if it doesn't exist
    os.makedirs("debug/regions", exist_ok=True)
    
    # Capture and save each region
    frames = self.screen_capture.capture()
    if "full_screen" in frames:
        full_frame = frames["full_screen"]
        cv2.imwrite("debug/regions/full_screen.png", full_frame)
        
        # Draw rectangles on the full screen to visualize regions
        debug_img = full_frame.copy()
        
        # Draw game window
        cv2.rectangle(
            debug_img,
            (manual_game_region["left"], manual_game_region["top"]),
            (manual_game_region["left"] + manual_game_region["width"], 
             manual_game_region["top"] + manual_game_region["height"]),
            (0, 255, 0), 2
        )
        
        # Draw sub-regions
        for name, region in sub_regions.items():
            color = (0, 0, 255)  # Red for sub-regions
            cv2.rectangle(
                debug_img,
                (region["left"], region["top"]),
                (region["left"] + region["width"], region["top"] + region["height"]),
                color, 2
            )
            # Add label
            cv2.putText(
                debug_img, name, (region["left"], region["top"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
        
        cv2.imwrite("debug/regions/regions_visualization.png", debug_img)
        logger.info("Saved regions visualization to debug/regions/regions_visualization.png")
    
    # Capture each individual region
    frames = self.screen_capture.capture()
    if "game_window" in frames:
        game_frame = frames["game_window"]
        cv2.imwrite("debug/regions/game_window.png", game_frame)
        
        # Update table analyzer with manual regions
        self.table_analyzer.set_regions(sub_regions)
        
        # Extract and save individual regions
        for name, region in sub_regions.items():
            try:
                x = region["left"] - manual_game_region["left"]
                y = region["top"] - manual_game_region["top"]
                w = region["width"]
                h = region["height"]
                
                # Verify coordinates are within bounds
                if (x >= 0 and y >= 0 and 
                    x + w <= game_frame.shape[1] and 
                    y + h <= game_frame.shape[0]):
                    sub_region = game_frame[y:y+h, x:x+w]
                    cv2.imwrite(f"debug/regions/{name}.png", sub_region)
                    logger.info(f"Saved region {name} to debug/regions/{name}.png")
            except Exception as e:
                logger.error(f"Error saving region {name}: {e}")
    
    # Skip standard calibration
    # self._calibrate()
    
    # Set up UI controller with button regions
    button_regions = {}
    if "action_buttons" in sub_regions:
        button_region = sub_regions["action_buttons"]
        
        # Estimate individual button positions (divide region into 3-4 buttons)
        width = button_region["width"]
        button_width = width // 4
        
        # Define individual buttons
        for i, name in enumerate(["fold", "check", "call", "raise"]):
            button_regions[name] = {
                "left": button_region["left"] + (i * button_width),
                "top": button_region["top"],
                "width": button_width,
                "height": button_region["height"]
            }
            
    # Define input regions
    input_regions = {}
    if "raise" in button_regions:
        raise_button = button_regions["raise"]
        input_regions["raise_input"] = {
            "left": raise_button["left"],
            "top": raise_button["top"] - raise_button["height"] - 5,
            "width": raise_button["width"],
            "height": raise_button["height"]
        }
            
    # Set regions for UI controller
    self.ui_controller.set_regions(button_regions, input_regions)
    
    # Start main loop
    self.running = True
    self._main_loop()
    
    def stop(self) -> None:
        """Stop the Poker AI system."""
        logger.info("Stopping Poker AI system")
        self.running = False
    
    def pause(self) -> None:
        """Pause the Poker AI system."""
        logger.info("Pausing Poker AI system")
        self.paused = True
    
    def resume(self) -> None:
        """Resume the Poker AI system."""
        logger.info("Resuming Poker AI system")
        self.paused = False
    
    def _calibrate(self) -> bool:
        """
        Perform system calibration.
        
        Returns:
            True if calibration succeeded, False otherwise
        """
        logger.info("Starting system calibration")
        
        try:
            # Capture full screen
            frames = self.screen_capture.capture()
            if not frames:
                logger.error("Failed to capture screen")
                return False
                
            # Try to detect game window
            full_screen = frames.get("full_screen")
            if full_screen is None:
                logger.error("Failed to get full screen capture")
                return False
            
            # Debug: Save screenshot for analysis
            import cv2
            import os
            
            # Create debug directory if it doesn't exist
            os.makedirs("debug", exist_ok=True)
            
            # Save full screen capture
            cv2.imwrite("debug/debug_fullscreen.png", full_screen)
            logger.info("Saved debug screenshot to debug/debug_fullscreen.png")
            
            # Save HSV channels separately
            hsv = cv2.cvtColor(full_screen, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            cv2.imwrite("debug/hsv_hue.png", h)
            cv2.imwrite("debug/hsv_saturation.png", s)
            cv2.imwrite("debug/hsv_value.png", v)
            logger.info("Saved HSV channels separately")
            
            # Try different green masks
            # Normal green mask
            green_mask = cv2.inRange(
                hsv,
                np.array([35, 50, 50]),
                np.array([85, 255, 255])
            )
            cv2.imwrite("debug/green_mask_normal.png", green_mask)
            
            # More permissive green mask
            green_mask_permissive = cv2.inRange(
                hsv,
                np.array([30, 30, 30]),
                np.array([100, 255, 255])
            )
            cv2.imwrite("debug/green_mask_permissive.png", green_mask_permissive)
            
            # Very permissive mask (catches almost any green)
            green_mask_very_permissive = cv2.inRange(
                hsv,
                np.array([20, 20, 20]),
                np.array([120, 255, 255])
            )
            cv2.imwrite("debug/green_mask_very_permissive.png", green_mask_very_permissive)
            
            logger.info("Saved green masks with different sensitivity levels")
            
            # Auto-detect game window
            game_region = self.screen_capture.auto_detect_game_window()
            if not game_region:
                logger.warning("Could not auto-detect game window. Using full screen.")
                return False
                
            # Set sub-regions
            sub_regions = self.screen_capture.get_sub_regions("game_window")
            self.screen_capture.update_regions(sub_regions)
            
            # Calibrate table analyzer
            frames = self.screen_capture.capture()
            game_frame = frames.get("game_window")
            if game_frame is not None:
                self.table_analyzer.calibrate(game_frame)
                
                # Debug: Save detected game window
                cv2.imwrite("debug/detected_game_window.png", game_frame)
                logger.info("Saved detected game window to debug/detected_game_window.png")
            
            # Extract button regions from table analyzer
            button_regions = {}
            if "action_buttons" in sub_regions:
                button_region = sub_regions["action_buttons"]
                
                # Estimate individual button positions (divide region into 3-4 buttons)
                width = button_region["width"]
                button_width = width // 4
                
                # Define individual buttons
                for i, name in enumerate(["fold", "check", "call", "raise"]):
                    button_regions[name] = {
                        "left": button_region["left"] + (i * button_width),
                        "top": button_region["top"],
                        "width": button_width,
                        "height": button_region["height"]
                    }
                    
            # Define input regions
            input_regions = {}
            if "raise" in button_regions:
                raise_button = button_regions["raise"]
                input_regions["raise_input"] = {
                    "left": raise_button["left"],
                    "top": raise_button["top"] - raise_button["height"] - 5,
                    "width": raise_button["width"],
                    "height": raise_button["height"]
                }
                
            # Set regions for UI controller
            self.ui_controller.set_regions(button_regions, input_regions)
            
            logger.info("System calibration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during calibration: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _main_loop(self) -> None:
        """Main execution loop for the Poker AI system."""
        exec_config = self.config.get("execution", {})
        loop_delay = exec_config.get("loop_delay", 1.0)
        max_hands = exec_config.get("max_hands", 0)
        
        hands_played = 0
        
        try:
            while self.running:
                if self.paused:
                    time.sleep(1.0)
                    continue
                    
                # Capture screen
                frames = self.screen_capture.capture()
                if not frames or "game_window" not in frames:
                    logger.warning("Failed to capture game window")
                    time.sleep(loop_delay)
                    continue
                    
                game_frame = frames["game_window"]
                
                # Analyze table
                game_state = self.table_analyzer.analyze_table(game_frame)
                
                # Update state tracker
                changes = self.state_tracker.update_state(game_state)
                
                # Process significant changes
                if changes:
                    # New hand detected
                    if changes.get("new_hand", False):
                        hands_played += 1
                        logger.info(f"New hand detected (#{hands_played})")
                        
                        # Analyze previous hand if available
                        if len(self.state_tracker.state_history) > 1:
                            hand_history = self.state_tracker.get_hand_history()[:-1]  # Exclude current state
                            if hand_history:
                                self._analyze_completed_hand(hand_history)
                                
                        # Check max hands limit
                        if max_hands > 0 and hands_played >= max_hands:
                            logger.info(f"Reached max hands limit: {max_hands}")
                            self.stop()
                            break
                            
                    # Phase change
                    if "phase_change" in changes:
                        phase_info = changes["phase_change"]
                        logger.info(f"Phase change: {phase_info['from']} -> {phase_info['to']}")
                        
                    # New community cards
                    if "new_community_cards" in changes:
                        new_cards = changes["new_community_cards"]
                        card_codes = [card.get("code", "??") for card in new_cards]
                        logger.info(f"New community cards: {', '.join(card_codes)}")
                        
                # Check if it's player's turn
                if self.state_tracker.is_player_turn():
                    available_actions = self.state_tracker.get_available_actions()
                    
                    if available_actions:
                        # Make a decision
                        decision = self._make_decision(game_state, available_actions)
                        
                        # Execute decision if in autonomous mode
                        if self.autonomous_mode:
                            action = decision[0]
                            amount = decision[1]
                            
                            logger.info(f"Executing action: {action}" + 
                                       (f" {amount}" if amount is not None else ""))
                            
                            self.ui_controller.perform_action(action, amount)
                        else:
                            # In non-autonomous mode, just log the recommended action
                            action = decision[0]
                            amount = decision[1]
                            logger.info(f"Recommended action: {action}" + 
                                       (f" {amount}" if amount is not None else ""))
                
                # Sleep to avoid CPU overload
                time.sleep(loop_delay)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            # Cleanup
            if self.screen_capture:
                self.screen_capture.close()
                
            logger.info("Poker AI system stopped")
    
    def _make_decision(self, 
                      game_state: Dict, 
                      available_actions: List[str]) -> Tuple[str, Optional[float]]:
        """
        Make a poker decision based on current game state.
        
        Args:
            game_state: Current game state
            available_actions: Available actions
            
        Returns:
            Tuple of (action, bet_amount)
        """
        # Evaluate hand
        player_cards = game_state.get("player_cards", [])
        community_cards = game_state.get("community_cards", [])
        
        hand_eval = None
        win_probability = 0.0
        
        if player_cards:
            # Basic hand evaluation
            hand_eval = self.hand_evaluator.evaluate_hand(player_cards, community_cards)
            
            # Win probability calculation
            num_opponents = len(game_state.get("opponents", {}))
            probs = self.hand_evaluator.calculate_win_probability(
                player_cards, 
                community_cards, 
                num_opponents=max(1, num_opponents)
            )
            win_probability = probs["win_probability"]
            
            # Add to hand evaluation
            hand_eval["win_probability"] = win_probability
            
        # Get opponent information
        opponent_info = {}
        for opp_id in self.opponent_tracker.get_all_opponent_ids():
            analysis = self.opponent_tracker.analyze_opponent(opp_id)
            if analysis.get("reliable", False):
                opponent_info[opp_id] = analysis
        
        # Try to get LLM advice if available
        llm_advice = None
        if self.llm_connector.is_available():
            try:
                llm_advice = self.llm_connector.analyze_decision(
                    game_state, 
                    available_actions,
                    hand_eval,
                    opponent_info
                )
            except Exception as e:
                logger.error(f"Error getting LLM advice: {e}")
        
        # Use the decision engine for final decision
        decision = self.decision_engine.make_decision(
            game_state, 
            available_actions
        )
        
        # If we have LLM advice and it's reasonable, consider it
        if llm_advice and llm_advice.get("success", False):
            llm_action = llm_advice.get("recommendation")
            llm_amount = llm_advice.get("bet_amount")
            llm_confidence = llm_advice.get("confidence", "low")
            
            # Consider LLM advice if it's a valid action
            if llm_action in available_actions:
                if llm_confidence == "high":
                    # High confidence - use LLM suggestion
                    decision = (llm_action, llm_amount)
                    logger.info("Using high-confidence LLM recommendation")
                elif llm_confidence == "medium":
                    # Medium confidence - blend with decision engine
                    # For simplicity, still use LLM action but adjust bet amount
                    if llm_action == "raise" and decision[0] == "raise":
                        # Average the bet amounts
                        if llm_amount is not None and decision[1] is not None:
                            avg_amount = (llm_amount + decision[1]) / 2
                            decision = (llm_action, avg_amount)
                            logger.info("Blending LLM and engine recommendations")
                        else:
                            decision = (llm_action, llm_amount or decision[1])
                    else:
                        # Use LLM action but keep decision engine's amount
                        decision = (llm_action, decision[1] if llm_action == "raise" else None)
                        logger.info("Using medium-confidence LLM action")
        
        return decision
    
    def _analyze_completed_hand(self, hand_history: List[Dict]) -> None:
        """
        Analyze a completed hand for insights and learning.
        
        Args:
            hand_history: List of game states in the hand
        """
        if not hand_history:
            return
            
        logger.info("Analyzing completed hand")
        
        try:
            # Use LLM for hand analysis if available
            if self.llm_connector.is_available():
                analysis = self.llm_connector.analyze_hand_history(hand_history)
                
                if analysis.get("success", False):
                    insights = analysis.get("insights", [])
                    if insights:
                        logger.info("Hand analysis insights:")
                        for i, insight in enumerate(insights, 1):
                            logger.info(f"  {i}. {insight}")
            
            # Update opponent models
            # This is simplified - in a real implementation,
            # you would extract more detailed opponent actions
            for state in hand_history:
                if "opponents" in state:
                    for opp_id, opp_data in state["opponents"].items():
                        # Register opponent
                        position = opp_data.get("position", -1)
                        self.opponent_tracker.register_opponent(opp_id, position)
                        
                        # Track actions if available
                        if "last_action" in opp_data:
                            action = opp_data["last_action"]
                            amount = opp_data.get("last_bet")
                            phase = state.get("game_phase", "unknown")
                            pot = state.get("pot", 0)
                            
                            self.opponent_tracker.track_action(
                                opp_id, action, amount, phase, pot
                            )
            
            # Mark hand as complete
            winners = []  # In a real implementation, extract winners
            showdown = False  # In a real implementation, determine if went to showdown
            self.opponent_tracker.hand_complete(winners, showdown)
            
        except Exception as e:
            logger.error(f"Error analyzing hand: {e}")
    
    def set_autonomous_mode(self, enabled: bool) -> None:
        """
        Enable or disable autonomous mode.
        
        Args:
            enabled: Whether to enable autonomous mode
        """
        self.autonomous_mode = enabled
        logger.info(f"Autonomous mode {'enabled' if enabled else 'disabled'}")
    
    def get_current_state(self) -> Dict:
        """
        Get the current game state.
        
        Returns:
            Current game state
        """
        return self.state_tracker.get_current_state() if self.state_tracker else {}
    
    def calibrate_regions(self) -> bool:
        """
        Manually trigger system recalibration.
        
        Returns:
            True if calibration succeeded, False otherwise
        """
        return self._calibrate()
    
    def set_strategy_profile(self, profile: str) -> None:
        """
        Set the strategy profile.
        
        Args:
            profile: Strategy profile
        """
        if self.decision_engine:
            self.decision_engine.set_strategy_profile(profile)
            logger.info(f"Strategy profile set to {profile}")
    
    def set_risk_tolerance(self, risk_tolerance: float) -> None:
        """
        Set the risk tolerance.
        
        Args:
            risk_tolerance: Risk tolerance factor
        """
        if self.decision_engine:
            self.decision_engine.set_risk_tolerance(risk_tolerance)
            logger.info(f"Risk tolerance set to {risk_tolerance}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Poker AI')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--autonomous', action='store_true',
                       help='Enable autonomous mode')
    parser.add_argument('--hands', type=int, default=0,
                       help='Maximum number of hands to play (0 for unlimited)')
    parser.add_argument('--profile', type=str, default=None,
                       choices=['tight', 'loose', 'aggressive', 'balanced'],
                       help='Strategy profile')
    parser.add_argument('--risk', type=float, default=None,
                       help='Risk tolerance (0.5-2.0)')
    
    args = parser.parse_args()
    
    try:
        # Initialize Poker AI
        poker_ai = PokerAI(args.config)
        
        # Apply command line overrides
        if args.autonomous:
            poker_ai.set_autonomous_mode(True)
            
        if args.profile:
            poker_ai.set_strategy_profile(args.profile)
            
        if args.risk is not None:
            poker_ai.set_risk_tolerance(args.risk)
            
        if args.hands > 0:
            poker_ai.config["execution"]["max_hands"] = args.hands
        
        # Start the system
        poker_ai.start()
        
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        sys.exit(1)