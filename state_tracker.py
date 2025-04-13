"""
State Tracker Module for Poker AI

This module tracks the poker game state over time, maintaining
history and detecting changes in the game.
"""
import numpy as np
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StateTracker:
    """
    Tracks poker game state over time and maintains history.
    """
    
    def __init__(self, 
                 history_size: int = 10,
                 log_to_file: bool = True,
                 log_directory: str = "logs"):
        """
        Initialize the state tracker.
        
        Args:
            history_size: Number of states to keep in history
            log_to_file: Whether to log state changes to file
            log_directory: Directory for log files
        """
        self.history_size = history_size
        self.state_history = []
        self.current_state = None
        self.previous_state = None
        self.hand_id = 0
        self.session_start_time = time.time()
        
        # Logging
        self.log_to_file = log_to_file
        self.log_directory = log_directory
        
        if log_to_file:
            os.makedirs(log_directory, exist_ok=True)
            self.log_file = os.path.join(
                log_directory, 
                f"poker_session_{int(self.session_start_time)}.log"
            )
    
    def update_state(self, new_state: Dict) -> Dict[str, Any]:
        """
        Update the current state and track changes.
        
        Args:
            new_state: New game state
            
        Returns:
            Dictionary of changes detected
        """
        # Store previous state
        self.previous_state = self.current_state
        
        # Update current state
        self.current_state = new_state
        
        # Detect changes
        changes = self._detect_changes()
        
        # Add to history
        self._add_to_history()
        
        # Log state if significant changes detected
        if changes and any(changes.values()):
            self._log_state(changes)
        
        return changes
    
    def _detect_changes(self) -> Dict[str, Any]:
        """
        Detect changes between current and previous state.
        
        Returns:
            Dictionary of detected changes
        """
        if not self.previous_state:
            return {"initial_state": True}
            
        changes = {}
        
        # Check for new hand
        if self._is_new_hand():
            changes["new_hand"] = True
            self.hand_id += 1
        
        # Check for game phase change
        if self._get_value(self.current_state, "game_phase") != self._get_value(self.previous_state, "game_phase"):
            changes["phase_change"] = {
                "from": self._get_value(self.previous_state, "game_phase"),
                "to": self._get_value(self.current_state, "game_phase")
            }
        
        # Check for new community cards
        new_cards = self._detect_new_community_cards()
        if new_cards:
            changes["new_community_cards"] = new_cards
        
        # Check for pot change
        current_pot = self._get_value(self.current_state, "pot", 0)
        previous_pot = self._get_value(self.previous_state, "pot", 0)
        if abs(current_pot - previous_pot) > 0.1:  # Small threshold for floating point comparison
            changes["pot_change"] = {
                "from": previous_pot,
                "to": current_pot,
                "difference": current_pot - previous_pot
            }
        
        # Check for balance change
        current_balance = self._get_value(self.current_state, "player_balance", 0)
        previous_balance = self._get_value(self.previous_state, "player_balance", 0)
        if abs(current_balance - previous_balance) > 0.1:
            changes["balance_change"] = {
                "from": previous_balance,
                "to": current_balance,
                "difference": current_balance - previous_balance
            }
        
        # Check for active player change
        if self._get_value(self.current_state, "active_player") != self._get_value(self.previous_state, "active_player"):
            changes["active_player_change"] = {
                "from": self._get_value(self.previous_state, "active_player"),
                "to": self._get_value(self.current_state, "active_player")
            }
        
        # Check for available actions change
        current_actions = set(self._get_value(self.current_state, "available_actions", []))
        previous_actions = set(self._get_value(self.previous_state, "available_actions", []))
        if current_actions != previous_actions:
            changes["available_actions_change"] = {
                "from": list(previous_actions),
                "to": list(current_actions),
                "added": list(current_actions - previous_actions),
                "removed": list(previous_actions - current_actions)
            }
        
        return changes
    
    def _is_new_hand(self) -> bool:
        """
        Determine if a new hand has started.
        
        Returns:
            True if a new hand has started, False otherwise
        """
        if not self.previous_state:
            return True
            
        # Check if player cards went from none to some (new hand dealt)
        prev_player_cards = self._get_value(self.previous_state, "player_cards", [])
        curr_player_cards = self._get_value(self.current_state, "player_cards", [])
        
        if not prev_player_cards and curr_player_cards:
            return True
            
        # Check if community cards reset
        prev_community_cards = self._get_value(self.previous_state, "community_cards", [])
        curr_community_cards = self._get_value(self.current_state, "community_cards", [])
        
        if prev_community_cards and not curr_community_cards:
            return True
            
        # Check if game phase went from river/showdown to pre-flop
        prev_phase = self._get_value(self.previous_state, "game_phase")
        curr_phase = self._get_value(self.current_state, "game_phase")
        
        if prev_phase in ["river", "showdown"] and curr_phase == "pre-flop":
            return True
            
        return False
    
    def _detect_new_community_cards(self) -> List[Dict]:
        """
        Detect new community cards compared to previous state.
        
        Returns:
            List of new card dictionaries
        """
        if not self.previous_state:
            return self._get_value(self.current_state, "community_cards", [])
            
        prev_cards = self._get_value(self.previous_state, "community_cards", [])
        curr_cards = self._get_value(self.current_state, "community_cards", [])
        
        # Create sets of card codes for easy comparison
        prev_codes = {card.get("code") for card in prev_cards if card.get("code")}
        curr_codes = {card.get("code") for card in curr_cards if card.get("code")}
        
        # Find new card codes
        new_codes = curr_codes - prev_codes
        
        # Return the full card dictionaries for new cards
        return [card for card in curr_cards if card.get("code") in new_codes]
    
    def _add_to_history(self) -> None:
        """Add current state to history, maintaining history size limit."""
        if self.current_state:
            # Create a copy to avoid reference issues
            self.state_history.append(self.current_state.copy())
            
            # Trim history if needed
            while len(self.state_history) > self.history_size:
                self.state_history.pop(0)
    
def _log_state(self, changes: Dict) -> None:
    """
    Log the current state and changes.
    
    Args:
        changes: Dictionary of detected changes
    """
    if not self.log_to_file:
        return
        
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # Convert numpy types to Python standard types
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.integer):
                return int(obj)
            elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.floating):
                return float(obj)
            elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        state_copy = convert_to_serializable(self.current_state) if self.current_state else {}
        changes_copy = convert_to_serializable(changes) if changes else {}
        
        log_entry = {
            "timestamp": timestamp,
            "hand_id": self.hand_id,
            "state": state_copy,
            "changes": changes_copy
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    except Exception as e:
        logger.error(f"Error logging state: {e}")
    
    def _get_value(self, state: Dict, key: str, default: Any = None) -> Any:
        """
        Safely get a value from a state dictionary.
        
        Args:
            state: State dictionary
            key: Key to retrieve
            default: Default value if key not found
            
        Returns:
            Value or default
        """
        if not state:
            return default
            
        return state.get(key, default)
    
    def get_hand_history(self) -> List[Dict]:
        """
        Get the history of the current hand.
        
        Returns:
            List of states in the current hand
        """
        # Find the start of the current hand
        hand_start_index = 0
        for i in range(len(self.state_history) - 1, -1, -1):
            state = self.state_history[i]
            changes = self._detect_changes_between(
                self.state_history[i-1] if i > 0 else None, 
                state
            )
            
            if changes.get("new_hand", False):
                hand_start_index = i
                break
        
        return self.state_history[hand_start_index:]
    
    def _detect_changes_between(self, state1: Optional[Dict], state2: Optional[Dict]) -> Dict:
        """
        Detect changes between any two states.
        
        Args:
            state1: First state (older)
            state2: Second state (newer)
            
        Returns:
            Dictionary of changes
        """
        if not state1:
            return {"initial_state": True}
            
        changes = {}
        
        # Check for new hand indicators
        if (self._get_value(state1, "player_cards", []) == [] and 
            self._get_value(state2, "player_cards", []) != []):
            changes["new_hand"] = True
        
        # Check for game phase change
        if self._get_value(state1, "game_phase") != self._get_value(state2, "game_phase"):
            changes["phase_change"] = {
                "from": self._get_value(state1, "game_phase"),
                "to": self._get_value(state2, "game_phase")
            }
        
        # Basic comparison for other fields
        for key in ["pot", "player_balance", "active_player"]:
            if self._get_value(state1, key) != self._get_value(state2, key):
                changes[f"{key}_change"] = {
                    "from": self._get_value(state1, key),
                    "to": self._get_value(state2, key)
                }
        
        return changes
    
    def get_current_hand_phase(self) -> str:
        """
        Get the current phase of the hand.
        
        Returns:
            Current phase as a string
        """
        if not self.current_state:
            return "unknown"
            
        return self._get_value(self.current_state, "game_phase", "unknown")
    
    def get_current_state(self) -> Dict:
        """
        Get the current game state.
        
        Returns:
            Current state dictionary
        """
        return self.current_state or {}
    
    def is_player_turn(self) -> bool:
        """
        Check if it's the player's turn to act.
        
        Returns:
            True if it's the player's turn, False otherwise
        """
        if not self.current_state:
            return False
            
        # Check if any action buttons are available
        available_actions = self._get_value(self.current_state, "available_actions", [])
        return len(available_actions) > 0
    
    def get_available_actions(self) -> List[str]:
        """
        Get the list of currently available actions.
        
        Returns:
            List of action strings
        """
        if not self.current_state:
            return []
            
        return self._get_value(self.current_state, "available_actions", [])
    
    def get_hand_id(self) -> int:
        """
        Get the current hand ID.
        
        Returns:
            Current hand ID
        """
        return self.hand_id
    
    def get_session_duration(self) -> float:
        """
        Get the duration of the current session in seconds.
        
        Returns:
            Session duration in seconds
        """
        return time.time() - self.session_start_time
    
    def reset(self) -> None:
        """Reset the state tracker."""
        self.state_history = []
        self.current_state = None
        self.previous_state = None
        self.hand_id = 0
        self.session_start_time = time.time()
        
        logger.info("State tracker reset")