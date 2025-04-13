"""
Decision Engine Module for Poker AI

This module handles strategic decision making for the poker AI,
combining hand evaluation, game state analysis, and opponent modeling.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import random

from src.game_state.hand_evaluator import HandEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Makes strategic decisions for the poker AI.
    """
    
    def __init__(self, 
                 hand_evaluator: Optional[HandEvaluator] = None,
                 strategy_profile: str = "balanced",
                 risk_tolerance: float = 1.0,
                 use_position: bool = True,
                 bluff_factor: float = 0.1):
        """
        Initialize the decision engine.
        
        Args:
            hand_evaluator: Hand evaluation component
            strategy_profile: Strategy profile ("tight", "loose", "aggressive", "balanced")
            risk_tolerance: Risk tolerance factor (< 1 more conservative, > 1 more aggressive)
            use_position: Whether to factor in table position
            bluff_factor: Frequency of bluffing (0-1)
        """
        self.hand_evaluator = hand_evaluator or HandEvaluator()
        self.strategy_profile = strategy_profile
        self.risk_tolerance = risk_tolerance
        self.use_position = use_position
        self.bluff_factor = bluff_factor
        
        # Strategy parameters based on profile
        self.strategy_params = self._initialize_strategy_params()
        
        # Decision history for this session
        self.decision_history = []
        
    def _initialize_strategy_params(self) -> Dict:
        """
        Initialize strategy parameters based on profile.
        
        Returns:
            Dictionary of strategy parameters
        """
        params = {
            "preflop_raise_threshold": 0.5,
            "preflop_call_threshold": 0.3,
            "postflop_raise_threshold": 0.6,
            "postflop_call_threshold": 0.4,
            "min_raise_hand_strength": 0.5,
            "min_call_hand_strength": 0.3,
            "max_bet_portion": 0.5,
            "bluff_frequency": self.bluff_factor,
            "position_weight": 0.2 if self.use_position else 0.0
        }
        
        # Adjust parameters based on strategy profile
        if self.strategy_profile == "tight":
            params["preflop_raise_threshold"] = 0.6
            params["preflop_call_threshold"] = 0.4
            params["postflop_raise_threshold"] = 0.7
            params["postflop_call_threshold"] = 0.5
            params["min_raise_hand_strength"] = 0.6
            params["min_call_hand_strength"] = 0.4
            params["bluff_frequency"] = max(0.05, self.bluff_factor - 0.05)
            
        elif self.strategy_profile == "loose":
            params["preflop_raise_threshold"] = 0.4
            params["preflop_call_threshold"] = 0.2
            params["postflop_raise_threshold"] = 0.5
            params["postflop_call_threshold"] = 0.3
            params["min_raise_hand_strength"] = 0.4
            params["min_call_hand_strength"] = 0.2
            params["bluff_frequency"] = min(0.3, self.bluff_factor + 0.1)
            
        elif self.strategy_profile == "aggressive":
            params["preflop_raise_threshold"] = 0.45
            params["preflop_call_threshold"] = 0.25
            params["postflop_raise_threshold"] = 0.55
            params["postflop_call_threshold"] = 0.35
            params["min_raise_hand_strength"] = 0.45
            params["min_call_hand_strength"] = 0.25
            params["max_bet_portion"] = 0.7
            params["bluff_frequency"] = min(0.2, self.bluff_factor + 0.05)
            
        return params
    
    def make_decision(self, 
                     game_state: Dict, 
                     available_actions: List[str]) -> Tuple[str, Optional[float]]:
        """
        Make a poker decision based on game state.
        
        Args:
            game_state: Current game state
            available_actions: List of available actions
            
        Returns:
            Tuple of (action, bet_amount)
        """
        if not available_actions:
            logger.warning("No available actions")
            return "check", None
            
        # Extract key information
        player_cards = game_state.get("player_cards", [])
        community_cards = game_state.get("community_cards", [])
        pot = game_state.get("pot", 0)
        player_balance = game_state.get("player_balance", 0)
        game_phase = game_state.get("game_phase", "unknown")
        current_bet = game_state.get("current_bet", 0)
        dealer_position = game_state.get("dealer_position")
        active_player = game_state.get("active_player")
        
        # Evaluate hand strength
        if player_cards:
            hand_eval = self.hand_evaluator.evaluate_hand(player_cards, community_cards)
            hand_strength = hand_eval["hand_strength"]
            hand_name = hand_eval["hand_name"]
            hand_description = hand_eval["hand_description"]
        else:
            hand_strength = 0
            hand_name = "unknown"
            hand_description = "No cards"
            
        logger.info(f"Hand strength: {hand_strength:.2f} ({hand_description})")
        
        # Calculate win probability
        num_opponents = len(game_state.get("opponents", {}))
        if player_cards:
            win_probs = self.hand_evaluator.calculate_win_probability(
                player_cards, 
                community_cards, 
                num_opponents=max(1, num_opponents)
            )
            win_probability = win_probs["win_probability"]
        else:
            win_probability = 0
            
        logger.info(f"Win probability: {win_probability:.2f}")
        
        # Adjust thresholds based on position
        position_adjustment = self._calculate_position_adjustment(
            dealer_position, active_player, num_opponents
        )
        
        # Decision logic based on game phase
        if game_phase == "pre-flop":
            decision = self._preflop_decision(
                hand_strength, 
                win_probability, 
                available_actions, 
                position_adjustment,
                pot,
                current_bet,
                player_balance
            )
        else:
            decision = self._postflop_decision(
                hand_strength, 
                win_probability, 
                available_actions, 
                game_phase,
                position_adjustment,
                pot,
                current_bet,
                player_balance
            )
            
        # Record decision
        self._record_decision(decision, game_state, hand_strength, win_probability)
        
        return decision
    
    def _preflop_decision(self, 
                         hand_strength: float, 
                         win_probability: float, 
                         available_actions: List[str],
                         position_adjustment: float,
                         pot: float,
                         current_bet: float,
                         player_balance: float) -> Tuple[str, Optional[float]]:
        """
        Make a pre-flop decision.
        
        Args:
            hand_strength: Current hand strength
            win_probability: Probability of winning
            available_actions: Available actions
            position_adjustment: Position-based adjustment
            pot: Current pot size
            current_bet: Current bet to call
            player_balance: Player's balance
            
        Returns:
            Tuple of (action, bet_amount)
        """
        # Adjust thresholds based on position
        raise_threshold = self.strategy_params["preflop_raise_threshold"] - position_adjustment
        call_threshold = self.strategy_params["preflop_call_threshold"] - position_adjustment
        
        # Apply risk tolerance
        raise_threshold /= self.risk_tolerance
        call_threshold /= self.risk_tolerance
        
        # Decision logic
        if hand_strength >= raise_threshold or win_probability >= raise_threshold:
            # Strong hand, try to raise
            if "raise" in available_actions:
                # Calculate raise amount (based on hand strength and pot size)
                raise_amount = self._calculate_bet_amount(
                    hand_strength, 
                    pot, 
                    player_balance, 
                    is_preflop=True
                )
                return "raise", raise_amount
            elif "call" in available_actions:
                return "call", current_bet
            elif "check" in available_actions:
                return "check", None
                
        elif hand_strength >= call_threshold or win_probability >= call_threshold:
            # Decent hand, call or check
            if "call" in available_actions:
                # Check pot odds
                pot_odds = self.hand_evaluator.calculate_pot_odds(pot, current_bet)
                if self.hand_evaluator.should_call(win_probability, pot_odds, self.risk_tolerance):
                    return "call", current_bet
                
            if "check" in available_actions:
                return "check", None
                
        # Consider bluffing
        if random.random() < self.strategy_params["bluff_frequency"]:
            if "raise" in available_actions:
                bluff_amount = self._calculate_bluff_amount(pot, player_balance)
                return "raise", bluff_amount
                
        # Default to folding or checking
        if "check" in available_actions:
            return "check", None
        else:
            return "fold", None
    
    def _postflop_decision(self, 
                          hand_strength: float, 
                          win_probability: float, 
                          available_actions: List[str],
                          game_phase: str,
                          position_adjustment: float,
                          pot: float,
                          current_bet: float,
                          player_balance: float) -> Tuple[str, Optional[float]]:
        """
        Make a post-flop decision.
        
        Args:
            hand_strength: Current hand strength
            win_probability: Probability of winning
            available_actions: Available actions
            game_phase: Current game phase
            position_adjustment: Position-based adjustment
            pot: Current pot size
            current_bet: Current bet to call
            player_balance: Player's balance
            
        Returns:
            Tuple of (action, bet_amount)
        """
        # Adjust thresholds based on position and game phase
        raise_threshold = self.strategy_params["postflop_raise_threshold"] - position_adjustment
        call_threshold = self.strategy_params["postflop_call_threshold"] - position_adjustment
        
        # Further adjustments based on game phase
        # As the hand progresses, be more selective
        if game_phase == "turn":
            raise_threshold += 0.05
            call_threshold += 0.05
        elif game_phase == "river":
            raise_threshold += 0.1
            call_threshold += 0.1
            
        # Apply risk tolerance
        raise_threshold /= self.risk_tolerance
        call_threshold /= self.risk_tolerance
        
        # Decision logic
        if hand_strength >= raise_threshold or win_probability >= raise_threshold:
            # Strong hand, try to raise
            if "raise" in available_actions:
                # Calculate raise amount
                raise_amount = self._calculate_bet_amount(
                    hand_strength, 
                    pot, 
                    player_balance, 
                    is_preflop=False
                )
                return "raise", raise_amount
            elif "call" in available_actions:
                return "call", current_bet
            elif "check" in available_actions:
                return "check", None
                
        elif hand_strength >= call_threshold or win_probability >= call_threshold:
            # Decent hand, call or check
            if "call" in available_actions:
                # Check pot odds
                pot_odds = self.hand_evaluator.calculate_pot_odds(pot, current_bet)
                if self.hand_evaluator.should_call(win_probability, pot_odds, self.risk_tolerance):
                    return "call", current_bet
                
            if "check" in available_actions:
                return "check", None
                
        # Bluff logic - more likely to bluff on later streets
        bluff_chance = self.strategy_params["bluff_frequency"]
        if game_phase == "turn":
            bluff_chance *= 1.2
        elif game_phase == "river":
            bluff_chance *= 1.5
            
        if random.random() < bluff_chance:
            if "raise" in available_actions:
                bluff_amount = self._calculate_bluff_amount(pot, player_balance)
                return "raise", bluff_amount
                
        # Default to folding or checking
        if "check" in available_actions:
            return "check", None
        else:
            return "fold", None
    
    def _calculate_bet_amount(self, 
                             hand_strength: float, 
                             pot: float, 
                             player_balance: float, 
                             is_preflop: bool) -> float:
        """
        Calculate an appropriate bet amount.
        
        Args:
            hand_strength: Hand strength
            pot: Current pot size
            player_balance: Player's balance
            is_preflop: Whether this is pre-flop
            
        Returns:
            Bet amount
        """
        max_portion = self.strategy_params["max_bet_portion"]
        
        # Scale bet amount based on hand strength and pot size
        if is_preflop:
            # Pre-flop bets are typically smaller
            bet_portion = min(0.33, hand_strength * max_portion)
        else:
            # Post-flop bets scale more with hand strength
            bet_portion = min(0.75, hand_strength * max_portion)
            
            # Add variance to bet sizing
            bet_portion *= random.uniform(0.8, 1.2)
            
        # Calculate raw bet amount
        bet_amount = pot * bet_portion
        
        # Make sure bet is reasonable
        min_bet = max(pot * 0.1, 1)  # At least 10% of pot or 1 unit
        max_bet = min(player_balance, pot * 2)  # No more than player balance or 2x pot
        
        # Adjust bet to reasonable range
        bet_amount = max(min_bet, min(max_bet, bet_amount))
        
        # Round to nearest whole number for cleaner bets
        return round(bet_amount)
    
    def _calculate_bluff_amount(self, pot: float, player_balance: float) -> float:
        """
        Calculate an appropriate bluff amount.
        
        Args:
            pot: Current pot size
            player_balance: Player's balance
            
        Returns:
            Bluff amount
        """
        # Bluffs should be substantial enough to be credible
        # but not too large to risk too much
        bluff_portion = random.uniform(0.5, 0.8)
        
        # Calculate raw bluff amount
        bluff_amount = pot * bluff_portion
        
        # Make sure bluff is reasonable
        min_bluff = max(pot * 0.3, 1)  # At least 30% of pot or 1 unit
        max_bluff = min(player_balance, pot * 1.5)  # No more than player balance or 1.5x pot
        
        # Adjust bluff to reasonable range
        bluff_amount = max(min_bluff, min(max_bluff, bluff_amount))
        
        # Round to nearest whole number for cleaner bets
        return round(bluff_amount)
    
    def _calculate_position_adjustment(self, 
                                      dealer_position: Optional[int], 
                                      active_player: Optional[int],
                                      num_opponents: int) -> float:
        """
        Calculate position-based adjustment.
        
        Args:
            dealer_position: Position of the dealer
            active_player: Position of the active player
            num_opponents: Number of opponents
            
        Returns:
            Position adjustment factor
        """
        if not self.use_position or dealer_position is None or active_player is None:
            return 0.0
            
        # Determine relative position
        # Late position is better (more information)
        total_players = num_opponents + 1  # Including player
        
        # Calculate positions clockwise from dealer
        positions = [(dealer_position + i) % total_players for i in range(total_players)]
        
        # Find player's position in the sequence
        try:
            player_index = positions.index(active_player)
        except ValueError:
            return 0.0
            
        # Convert to a 0-1 scale (0 = earliest position, 1 = latest position)
        position_value = player_index / (total_players - 1) if total_players > 1 else 0.5
        
        # Calculate adjustment (-0.1 to +0.1)
        adjustment = (position_value - 0.5) * self.strategy_params["position_weight"]
        
        return adjustment
    
    def _record_decision(self, 
                        decision: Tuple[str, Optional[float]], 
                        game_state: Dict, 
                        hand_strength: float,
                        win_probability: float) -> None:
        """
        Record a decision for later analysis.
        
        Args:
            decision: The decision made
            game_state: Current game state
            hand_strength: Current hand strength
            win_probability: Calculated win probability
        """
        timestamp = time.time()
        
        decision_record = {
            "timestamp": timestamp,
            "action": decision[0],
            "amount": decision[1],
            "hand_strength": hand_strength,
            "win_probability": win_probability,
            "game_phase": game_state.get("game_phase"),
            "pot": game_state.get("pot"),
            "player_balance": game_state.get("player_balance"),
            "player_cards": [card.get("code") for card in game_state.get("player_cards", [])],
            "community_cards": [card.get("code") for card in game_state.get("community_cards", [])]
        }
        
        self.decision_history.append(decision_record)
        
        # Log the decision
        action_str = decision[0]
        if decision[1] is not None:
            action_str += f" {decision[1]}"
            
        logger.info(f"Decision: {action_str}")
    
    def set_strategy_profile(self, profile: str) -> None:
        """
        Change the strategy profile.
        
        Args:
            profile: New strategy profile
        """
        if profile in ["tight", "loose", "aggressive", "balanced"]:
            self.strategy_profile = profile
            self.strategy_params = self._initialize_strategy_params()
            logger.info(f"Strategy profile changed to {profile}")
    
    def set_risk_tolerance(self, risk_tolerance: float) -> None:
        """
        Change the risk tolerance.
        
        Args:
            risk_tolerance: New risk tolerance factor
        """
        self.risk_tolerance = max(0.1, min(3.0, risk_tolerance))
        logger.info(f"Risk tolerance changed to {self.risk_tolerance}")
    
    def set_bluff_factor(self, bluff_factor: float) -> None:
        """
        Change the bluff factor.
        
        Args:
            bluff_factor: New bluff factor
        """
        self.bluff_factor = max(0.0, min(0.5, bluff_factor))
        self.strategy_params["bluff_frequency"] = self.bluff_factor
        logger.info(f"Bluff factor changed to {self.bluff_factor}")
        
    def get_decision_history(self) -> List[Dict]:
        """
        Get the decision history.
        
        Returns:
            List of decision records
        """
        return self.decision_history