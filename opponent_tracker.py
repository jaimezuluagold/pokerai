"""
Opponent Tracker Module for Poker AI

This module tracks and analyzes opponent behavior to adapt
the AI's strategy based on observed patterns.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpponentTracker:
    """
    Tracks and analyzes opponent behavior over time.
    """
    
    def __init__(self, 
                 min_hands_for_analysis: int = 10,
                 aggression_smoothing: float = 0.7):
        """
        Initialize the opponent tracker.
        
        Args:
            min_hands_for_analysis: Minimum hands needed for reliable analysis
            aggression_smoothing: Smoothing factor for aggression metrics
        """
        self.min_hands_for_analysis = min_hands_for_analysis
        self.aggression_smoothing = aggression_smoothing
        
        # Dictionary to track opponents by position/name
        self.opponents = {}
        
        # Current hand actions
        self.current_hand_actions = defaultdict(list)
        
        # Hand counter
        self.hand_counter = 0
    
    def register_opponent(self, opponent_id: str, position: int) -> None:
        """
        Register a new opponent or update position.
        
        Args:
            opponent_id: Identifier for the opponent
            position: Table position
        """
        if opponent_id not in self.opponents:
            # Initialize new opponent tracking
            self.opponents[opponent_id] = {
                "position": position,
                "hands_played": 0,
                "vpip": 0.0,  # Voluntarily Put $ In Pot
                "pfr": 0.0,   # Pre-Flop Raise
                "af": 1.0,    # Aggression Factor
                "aggression_count": {"raise": 0, "call": 0, "check": 0, "fold": 0},
                "position_plays": defaultdict(lambda: {"hands": 0, "vpip": 0, "pfr": 0}),
                "phase_aggression": {
                    "pre-flop": 1.0,
                    "flop": 1.0,
                    "turn": 1.0,
                    "river": 1.0
                },
                "bluff_tendency": 0.5,  # Initial neutral estimate
                "showdown_count": 0,
                "showdown_win_count": 0,
                "hand_strengths": [],
                "recent_actions": [],
                "recent_bets": []
            }
            logger.info(f"New opponent registered: {opponent_id} at position {position}")
        else:
            # Update position if changed
            if self.opponents[opponent_id]["position"] != position:
                self.opponents[opponent_id]["position"] = position
                logger.info(f"Opponent {opponent_id} moved to position {position}")
    
    def track_action(self, 
                    opponent_id: str, 
                    action: str, 
                    amount: Optional[float] = None,
                    game_phase: str = "unknown",
                    pot_size: float = 0.0) -> None:
        """
        Track an opponent's action in the current hand.
        
        Args:
            opponent_id: Identifier for the opponent
            action: Action taken ("raise", "call", "check", "fold")
            amount: Bet amount if applicable
            game_phase: Current game phase
            pot_size: Current pot size
        """
        if opponent_id not in self.opponents:
            logger.warning(f"Tracking action for unregistered opponent {opponent_id}")
            self.register_opponent(opponent_id, -1)  # Register with unknown position
            
        # Track the action for the current hand
        self.current_hand_actions[opponent_id].append({
            "action": action,
            "amount": amount,
            "game_phase": game_phase,
            "pot_size": pot_size,
            "timestamp": time.time()
        })
        
        # Store in recent actions list
        self.opponents[opponent_id]["recent_actions"].append({
            "action": action,
            "amount": amount,
            "game_phase": game_phase,
            "pot_size": pot_size,
            "timestamp": time.time()
        })
        
        # Keep only the 20 most recent actions
        if len(self.opponents[opponent_id]["recent_actions"]) > 20:
            self.opponents[opponent_id]["recent_actions"].pop(0)
            
        # Track bet amounts separately
        if amount is not None and amount > 0:
            self.opponents[opponent_id]["recent_bets"].append({
                "amount": amount,
                "pot_size": pot_size,
                "game_phase": game_phase
            })
            
            # Keep only the 10 most recent bets
            if len(self.opponents[opponent_id]["recent_bets"]) > 10:
                self.opponents[opponent_id]["recent_bets"].pop(0)
    
    def hand_complete(self, winners: Optional[List[str]] = None, showdown: bool = False) -> None:
        """
        Process the completed hand and update opponent profiles.
        
        Args:
            winners: List of opponent IDs who won
            showdown: Whether the hand went to showdown
        """
        # Increment hand counter
        self.hand_counter += 1
        
        # Process actions for each opponent
        for opponent_id, actions in self.current_hand_actions.items():
            if opponent_id in self.opponents:
                self._process_opponent_hand(opponent_id, actions, winners, showdown)
                
        # Reset current hand actions
        self.current_hand_actions = defaultdict(list)
        
        logger.info(f"Hand {self.hand_counter} processed")
    
    def _process_opponent_hand(self, 
                              opponent_id: str, 
                              actions: List[Dict], 
                              winners: Optional[List[str]],
                              showdown: bool) -> None:
        """
        Process a single opponent's actions for the completed hand.
        
        Args:
            opponent_id: Identifier for the opponent
            actions: List of action dictionaries
            winners: List of opponent IDs who won
            showdown: Whether the hand went to showdown
        """
        opponent = self.opponents[opponent_id]
        
        # Increment hands played
        opponent["hands_played"] += 1
        position = opponent["position"]
        opponent["position_plays"][position]["hands"] += 1
        
        # Track showdown information
        if showdown:
            opponent["showdown_count"] += 1
            if winners and opponent_id in winners:
                opponent["showdown_win_count"] += 1
        
        # Process pre-flop actions
        preflop_actions = [a for a in actions if a["game_phase"] == "pre-flop"]
        if preflop_actions:
            self._update_preflop_stats(opponent, preflop_actions, position)
            
        # Update aggression metrics for each phase
        phase_actions = defaultdict(list)
        for action in actions:
            phase = action["game_phase"]
            phase_actions[phase].append(action)
            
        for phase, phase_action_list in phase_actions.items():
            self._update_aggression_metrics(opponent, phase_action_list, phase)
            
        # Update overall aggression counters
        for action in actions:
            action_type = action["action"]
            if action_type in ["raise", "call", "check", "fold"]:
                opponent["aggression_count"][action_type] += 1
                
        # Recalculate overall aggression factor
        self._recalculate_aggression_factor(opponent)
        
        # Update bluff tendency estimate
        if showdown and winners:
            self._update_bluff_tendency(opponent, actions, opponent_id in winners)
    
    def _update_preflop_stats(self, 
                             opponent: Dict, 
                             actions: List[Dict],
                             position: int) -> None:
        """
        Update pre-flop statistics for an opponent.
        
        Args:
            opponent: Opponent data dictionary
            actions: List of pre-flop actions
            position: Table position
        """
        # Check for voluntary money in pot (VPIP)
        vpip_actions = ["call", "raise"]
        has_vpip = any(a["action"] in vpip_actions for a in actions)
        
        # Check for pre-flop raise (PFR)
        has_pfr = any(a["action"] == "raise" for a in actions)
        
        # Update VPIP and PFR metrics
        n = opponent["hands_played"]
        
        # Overall stats with smoothing for stability
        if has_vpip:
            opponent["vpip"] = ((n - 1) * opponent["vpip"] + 1) / n
        else:
            opponent["vpip"] = ((n - 1) * opponent["vpip"]) / n
            
        if has_pfr:
            opponent["pfr"] = ((n - 1) * opponent["pfr"] + 1) / n
        else:
            opponent["pfr"] = ((n - 1) * opponent["pfr"]) / n
            
        # Position-specific stats
        position_data = opponent["position_plays"][position]
        position_data_n = position_data["hands"]
        
        if has_vpip:
            position_data["vpip"] += 1
        
        if has_pfr:
            position_data["pfr"] += 1
    
    def _update_aggression_metrics(self, 
                                  opponent: Dict, 
                                  actions: List[Dict],
                                  phase: str) -> None:
        """
        Update aggression metrics for a specific game phase.
        
        Args:
            opponent: Opponent data dictionary
            actions: List of actions in the phase
            phase: Game phase
        """
        if not actions:
            return
            
        # Count aggressive vs. passive actions
        aggressive_count = sum(1 for a in actions if a["action"] == "raise")
        passive_count = sum(1 for a in actions if a["action"] in ["call", "check"])
        
        # Skip if no relevant actions
        if aggressive_count == 0 and passive_count == 0:
            return
            
        # Calculate raw aggression factor for this phase
        # (Avoid division by zero)
        if passive_count == 0:
            raw_af = 3.0  # Cap at 3.0 for extremely aggressive
        else:
            raw_af = aggressive_count / passive_count
            raw_af = min(3.0, raw_af)  # Cap at 3.0
            
        # Apply smoothing to avoid overreacting to small samples
        current_af = opponent["phase_aggression"].get(phase, 1.0)
        smoothed_af = (self.aggression_smoothing * current_af + 
                      (1 - self.aggression_smoothing) * raw_af)
        
        # Update the phase aggression
        opponent["phase_aggression"][phase] = smoothed_af
    
    def _recalculate_aggression_factor(self, opponent: Dict) -> None:
        """
        Recalculate the overall aggression factor.
        
        Args:
            opponent: Opponent data dictionary
        """
        agg_counts = opponent["aggression_count"]
        aggressive = agg_counts["raise"]
        passive = agg_counts["call"] + agg_counts["check"]
        
        if passive == 0:
            opponent["af"] = 3.0  # Cap at 3.0 for extremely aggressive
        else:
            opponent["af"] = min(3.0, aggressive / passive)
    
    def _update_bluff_tendency(self, 
                              opponent: Dict, 
                              actions: List[Dict],
                              won_hand: bool) -> None:
        """
        Update bluff tendency based on showdown results.
        
        Args:
            opponent: Opponent data dictionary
            actions: List of actions in the hand
            won_hand: Whether opponent won the hand
        """
        # Simple heuristic: if player was aggressive but lost, might be a bluff
        aggressive_actions = [a for a in actions if a["action"] == "raise"]
        
        if not aggressive_actions:
            return  # No aggression, no bluff update
            
        # Look at late-street aggression
        late_aggression = any(
            a["action"] == "raise" and a["game_phase"] in ["turn", "river"] 
            for a in actions
        )
        
        # Calculate a bluff signal (-1 to 1)
        # Positive means likely bluffing, negative means likely value betting
        if late_aggression:
            if won_hand:
                # Aggressive and won - probably value betting
                bluff_signal = -0.2
            else:
                # Aggressive and lost - possible bluff
                bluff_signal = 0.3
        else:
            # Early aggression has less impact on bluff tendency
            if won_hand:
                bluff_signal = -0.1
            else:
                bluff_signal = 0.1
                
        # Apply smoothing to current estimate
        current = opponent["bluff_tendency"]
        # More weight to current estimate until we have more data
        smoothing_factor = min(0.8, opponent["showdown_count"] / 20)
        
        opponent["bluff_tendency"] = max(0.0, min(1.0, 
            (1 - smoothing_factor) * current + smoothing_factor * (current + bluff_signal)
        ))
    
    def analyze_opponent(self, opponent_id: str) -> Dict:
        """
        Get a comprehensive analysis of an opponent.
        
        Args:
            opponent_id: Identifier for the opponent
            
        Returns:
            Analysis dictionary
        """
        if opponent_id not in self.opponents:
            logger.warning(f"Attempting to analyze unknown opponent {opponent_id}")
            return {"reliable": False}
            
        opponent = self.opponents[opponent_id]
        
        # Check if we have enough data for reliable analysis
        reliable = opponent["hands_played"] >= self.min_hands_for_analysis
        
        # Basic classification
        player_type = self._classify_player_type(opponent)
        
        # Calculate positional tendencies
        positional_play = {}
        for pos, data in opponent["position_plays"].items():
            if data["hands"] > 0:
                positional_play[pos] = {
                    "vpip": data["vpip"] / data["hands"],
                    "pfr": data["pfr"] / data["hands"],
                    "hands": data["hands"]
                }
        
        # Betting patterns
        bet_patterns = self._analyze_bet_patterns(opponent)
        
        # Aggression by phase
        phase_agg = opponent["phase_aggression"]
        
        # Showdown tendencies
        if opponent["showdown_count"] > 0:
            showdown_win_rate = opponent["showdown_win_count"] / opponent["showdown_count"]
        else:
            showdown_win_rate = 0.0
            
        return {
            "reliable": reliable,
            "hands_analyzed": opponent["hands_played"],
            "player_type": player_type,
            "stats": {
                "vpip": opponent["vpip"],
                "pfr": opponent["pfr"],
                "af": opponent["af"],
                "bluff_tendency": opponent["bluff_tendency"],
                "showdown_win_rate": showdown_win_rate
            },
            "positional_play": positional_play,
            "phase_aggression": phase_agg,
            "betting_patterns": bet_patterns,
            "action_history": opponent["recent_actions"][-5:]  # Last 5 actions
        }
    
    def _classify_player_type(self, opponent: Dict) -> str:
        """
        Classify opponent into a player type.
        
        Args:
            opponent: Opponent data dictionary
            
        Returns:
            Player type classification
        """
        vpip = opponent["vpip"]
        pfr = opponent["pfr"]
        af = opponent["af"]
        
        # Basic classifications based on VPIP and PFR
        if vpip < 0.15:
            if pfr < 0.10:
                base_type = "rock"
            else:
                base_type = "nit"
        elif 0.15 <= vpip < 0.25:
            if pfr / vpip > 0.7:
                base_type = "tag"  # Tight-Aggressive
            else:
                base_type = "tap"  # Tight-Passive
        elif 0.25 <= vpip < 0.4:
            if pfr / vpip > 0.6:
                base_type = "lag"  # Loose-Aggressive
            else:
                base_type = "calling_station"
        else:  # vpip >= 0.4
            if af > 1.5:
                base_type = "maniac"
            else:
                base_type = "fish"
                
        # Refine based on aggression factor
        if base_type in ["tag", "lag"] and af < 1.0:
            base_type = base_type.replace("a", "p")  # Convert to passive version
            
        if base_type in ["tap", "calling_station"] and af > 2.0:
            if base_type == "tap":
                base_type = "tag"
            else:
                base_type = "lag"
                
        return base_type
    
    def _analyze_bet_patterns(self, opponent: Dict) -> Dict:
        """
        Analyze betting patterns.
        
        Args:
            opponent: Opponent data dictionary
            
        Returns:
            Betting pattern analysis
        """
        recent_bets = opponent["recent_bets"]
        if not recent_bets:
            return {"consistent": False, "pot_sensitivity": "unknown"}
            
        # Analyze bet sizing relative to pot
        pot_ratios = [bet["amount"] / bet["pot_size"] if bet["pot_size"] > 0 else 0 
                     for bet in recent_bets]
        
        if not pot_ratios:
            return {"consistent": False, "pot_sensitivity": "unknown"}
            
        # Check consistency of bet sizing
        avg_ratio = sum(pot_ratios) / len(pot_ratios)
        variance = sum((r - avg_ratio) ** 2 for r in pot_ratios) / len(pot_ratios)
        std_dev = math.sqrt(variance)
        
        # Determine consistency and pot sensitivity
        consistent = std_dev < 0.2  # Threshold for consistency
        
        if avg_ratio < 0.33:
            pot_sensitivity = "small_bets"
        elif avg_ratio < 0.66:
            pot_sensitivity = "medium_bets"
        else:
            pot_sensitivity = "large_bets"
            
        return {
            "consistent": consistent,
            "pot_sensitivity": pot_sensitivity,
            "avg_pot_ratio": avg_ratio,
            "std_dev": std_dev
        }
    
    def get_all_opponent_ids(self) -> List[str]:
        """
        Get a list of all tracked opponent IDs.
        
        Returns:
            List of opponent IDs
        """
        return list(self.opponents.keys())
    
    def get_opponent_data(self, opponent_id: str) -> Optional[Dict]:
        """
        Get raw data for a specific opponent.
        
        Args:
            opponent_id: Identifier for the opponent
            
        Returns:
            Opponent data dictionary or None if not found
        """
        return self.opponents.get(opponent_id)
    
    def reset_tracking(self) -> None:
        """Reset all opponent tracking data."""
        self.opponents = {}
        self.current_hand_actions = defaultdict(list)
        self.hand_counter = 0
        logger.info("Opponent tracking data reset")