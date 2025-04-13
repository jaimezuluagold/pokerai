"""
Hand Evaluator Module for Poker AI

This module evaluates poker hands, calculates hand strength,
and provides probability estimates for win/loss outcomes.
"""

import logging
import itertools
from typing import Dict, List, Tuple, Set, Optional, Any
import random
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Card values ordered from lowest to highest
CARD_VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
CARD_SUITS = ['clubs', 'diamonds', 'hearts', 'spades']
HAND_RANKS = [
    'high_card',
    'pair',
    'two_pair',
    'three_of_a_kind',
    'straight',
    'flush',
    'full_house',
    'four_of_a_kind',
    'straight_flush',
    'royal_flush'
]


class HandEvaluator:
    """
    Evaluates poker hands and calculates winning probabilities.
    """
    
    def __init__(self, use_fast_eval: bool = True):
        """
        Initialize the hand evaluator.
        
        Args:
            use_fast_eval: Whether to use fast evaluation (less accurate but quicker)
        """
        self.use_fast_eval = use_fast_eval
    
    def evaluate_hand(self, 
                      player_cards: List[Dict], 
                      community_cards: List[Dict]) -> Dict:
        """
        Evaluate the current hand strength.
        
        Args:
            player_cards: List of player card dictionaries
            community_cards: List of community card dictionaries
            
        Returns:
            Dictionary with hand evaluation information
        """
        # Convert card dictionaries to standardized format
        cards = self._standardize_cards(player_cards + community_cards)
        
        # Initialize result
        result = {
            "hand_name": "unrecognized",
            "hand_rank": -1,
            "hand_cards": [],
            "kickers": [],
            "hand_strength": 0.0,  # 0 to 1 scale
            "hand_description": ""
        }
        
        # Check for valid cards
        if len(cards) < 2:
            return result
            
        # Evaluate hand
        best_hand, hand_name, hand_cards, kickers = self._get_best_hand(cards)
        
        # Set result information
        result["hand_name"] = hand_name
        result["hand_rank"] = HAND_RANKS.index(hand_name) if hand_name in HAND_RANKS else -1
        result["hand_cards"] = hand_cards
        result["kickers"] = kickers
        
        # Calculate normalized hand strength (0 to 1)
        if hand_name in HAND_RANKS:
            # Base strength from hand rank
            base_strength = HAND_RANKS.index(hand_name) / (len(HAND_RANKS) - 1)
            
            # Adjust for quality within rank
            quality_adjustment = self._calculate_quality_adjustment(hand_name, hand_cards, kickers)
            
            result["hand_strength"] = base_strength + quality_adjustment
            
        # Generate hand description
        result["hand_description"] = self._describe_hand(hand_name, hand_cards, kickers)
        
        return result
    
    def _standardize_cards(self, cards: List[Dict]) -> List[Tuple[str, str]]:
        """
        Convert card dictionaries to standardized (value, suit) tuples.
        
        Args:
            cards: List of card dictionaries
            
        Returns:
            List of (value, suit) tuples
        """
        standardized = []
        
        for card in cards:
            # Skip invalid or unrecognized cards
            if not card.get("value") or not card.get("suit"):
                continue
                
            value = card["value"]
            suit = card["suit"]
            
            # Fix for question mark values
            if value == "?":
                logger.warning(f"Found card with '?' value, skipping: {card}")
                continue
                
            # Validate value and suit
            if value in CARD_VALUES and suit in CARD_SUITS:
                standardized.append((value, suit))
            else:
                logger.warning(f"Invalid card value or suit: {value} of {suit}")
                
        return standardized
    
    def _get_best_hand(self, 
                       cards: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str, List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Find the best 5-card hand from the available cards.
        
        Args:
            cards: List of (value, suit) tuples
            
        Returns:
            Tuple of (best hand, hand name, hand cards, kickers)
        """
        if len(cards) < 5:
            return cards, "high_card", cards, []
            
        # Generate all possible 5-card combinations
        all_combos = list(itertools.combinations(cards, 5))
        
        # For efficiency with 7 cards, only evaluate a subset if fast mode
        # This is a time-space tradeoff
        if self.use_fast_eval and len(all_combos) > 100:
            all_combos = random.sample(all_combos, 100)
            
        best_hand = None
        best_rank = -1
        best_hand_name = "high_card"
        best_hand_cards = []
        best_kickers = []
        
        # Evaluate each combination
        for hand in all_combos:
            hand_name, hand_cards, kickers = self._evaluate_five_card_hand(list(hand))
            rank = HAND_RANKS.index(hand_name) if hand_name in HAND_RANKS else -1
            
            # Compare with current best
            if rank > best_rank:
                best_hand = hand
                best_rank = rank
                best_hand_name = hand_name
                best_hand_cards = hand_cards
                best_kickers = kickers
            elif rank == best_rank:
                # If same rank, compare cards for tie-breaker
                if self._compare_same_rank_hands(hand_cards, best_hand_cards, kickers, best_kickers) > 0:
                    best_hand = hand
                    best_hand_cards = hand_cards
                    best_kickers = kickers
        
        return list(best_hand), best_hand_name, best_hand_cards, best_kickers
    
    def _evaluate_five_card_hand(self, 
                                hand: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Evaluate a specific 5-card hand.
        
        Args:
            hand: List of 5 (value, suit) tuples
            
        Returns:
            Tuple of (hand name, hand cards, kickers)
        """
        # Check for flush
        suits = [card[1] for card in hand]
        is_flush = len(set(suits)) == 1
        
        # Check for straight
        values = [card[0] for card in hand]
        value_indices = [CARD_VALUES.index(v) for v in values]
        value_indices.sort()
        
        # Check regular straight
        is_straight = (
            len(set(value_indices)) == 5 and
            max(value_indices) - min(value_indices) == 4
        )
        
        # Check A-5 straight (special case)
        is_a_to_5_straight = (
            set(value_indices) == {0, 1, 2, 3, 12}  # 2,3,4,5,A
        )
        
        if is_a_to_5_straight:
            is_straight = True
            # Adjust Ace to be low in this case
            for i, card in enumerate(hand):
                if card[0] == 'A':
                    hand[i] = ('1', card[1])  # Temporarily make Ace low
                    
        # Sort hand by value for easier processing
        hand.sort(key=lambda card: CARD_VALUES.index(card[0]))
        
        # Check for straight flush and royal flush
        if is_straight and is_flush:
            if not is_a_to_5_straight and hand[-1][0] == 'A':
                return "royal_flush", hand, []
            return "straight_flush", hand, []
            
        # Count values for pairs, trips, etc.
        value_counts = Counter(values)
        
        # Check for four of a kind
        if 4 in value_counts.values():
            quads_value = [v for v, count in value_counts.items() if count == 4][0]
            quads_cards = [card for card in hand if card[0] == quads_value]
            kickers = [card for card in hand if card[0] != quads_value]
            return "four_of_a_kind", quads_cards, kickers
            
        # Check for full house
        if 3 in value_counts.values() and 2 in value_counts.values():
            trips_value = [v for v, count in value_counts.items() if count == 3][0]
            pair_value = [v for v, count in value_counts.items() if count == 2][0]
            trips_cards = [card for card in hand if card[0] == trips_value]
            pair_cards = [card for card in hand if card[0] == pair_value]
            return "full_house", trips_cards + pair_cards, []
            
        # Check for flush
        if is_flush:
            return "flush", hand, []
            
        # Check for straight
        if is_straight:
            return "straight", hand, []
            
        # Check for three of a kind
        if 3 in value_counts.values():
            trips_value = [v for v, count in value_counts.items() if count == 3][0]
            trips_cards = [card for card in hand if card[0] == trips_value]
            kickers = [card for card in hand if card[0] != trips_value]
            return "three_of_a_kind", trips_cards, kickers
            
        # Check for two pair
        if list(value_counts.values()).count(2) == 2:
            pair_values = [v for v, count in value_counts.items() if count == 2]
            high_pair_value = max(pair_values, key=lambda v: CARD_VALUES.index(v))
            low_pair_value = min(pair_values, key=lambda v: CARD_VALUES.index(v))
            high_pair_cards = [card for card in hand if card[0] == high_pair_value]
            low_pair_cards = [card for card in hand if card[0] == low_pair_value]
            kickers = [card for card in hand if card[0] not in pair_values]
            return "two_pair", high_pair_cards + low_pair_cards, kickers
            
        # Check for one pair
        if 2 in value_counts.values():
            pair_value = [v for v, count in value_counts.items() if count == 2][0]
            pair_cards = [card for card in hand if card[0] == pair_value]
            kickers = [card for card in hand if card[0] != pair_value]
            return "pair", pair_cards, kickers
            
        # Must be high card
        high_card = [max(hand, key=lambda card: CARD_VALUES.index(card[0]))]
        kickers = [card for card in hand if card != high_card[0]]
        return "high_card", high_card, kickers
    
    def _compare_same_rank_hands(self, 
                                hand1_cards: List[Tuple[str, str]], 
                                hand2_cards: List[Tuple[str, str]],
                                hand1_kickers: List[Tuple[str, str]],
                                hand2_kickers: List[Tuple[str, str]]) -> int:
        """
        Compare two hands of the same rank.
        
        Args:
            hand1_cards: First hand cards
            hand2_cards: Second hand cards
            hand1_kickers: First hand kickers
            hand2_kickers: Second hand kickers
            
        Returns:
            1 if first hand is better, -1 if second hand is better, 0 if tie
        """
        # Compare main hand cards first by value
        hand1_values = [CARD_VALUES.index(card[0]) for card in hand1_cards]
        hand2_values = [CARD_VALUES.index(card[0]) for card in hand2_cards]
        
        hand1_values.sort(reverse=True)
        hand2_values.sort(reverse=True)
        
        for v1, v2 in zip(hand1_values, hand2_values):
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
                
        # If main cards tied, compare kickers
        kicker1_values = [CARD_VALUES.index(card[0]) for card in hand1_kickers]
        kicker2_values = [CARD_VALUES.index(card[0]) for card in hand2_kickers]
        
        kicker1_values.sort(reverse=True)
        kicker2_values.sort(reverse=True)
        
        # Compare as many kickers as available
        for v1, v2 in zip(kicker1_values, kicker2_values):
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
                
        # If everything is tied
        return 0
    
    def _calculate_quality_adjustment(self, 
                                     hand_name: str, 
                                     hand_cards: List[Tuple[str, str]], 
                                     kickers: List[Tuple[str, str]]) -> float:
        """
        Calculate a quality adjustment factor for hands of the same rank.
        
        Args:
            hand_name: Name of the hand
            hand_cards: Cards that form the hand
            kickers: Kicker cards
            
        Returns:
            Quality adjustment value (0 to 0.1)
        """
        # Maximum adjustment factor (added to base rank)
        max_adjustment = 0.1
        
        # Get values as indices for easier comparison
        hand_values = [CARD_VALUES.index(card[0]) for card in hand_cards]
        kicker_values = [CARD_VALUES.index(card[0]) for card in kickers]
        
        # Normalize card values (0 to 1)
        max_value_idx = len(CARD_VALUES) - 1
        norm_hand_values = [v / max_value_idx for v in hand_values]
        norm_kicker_values = [v / max_value_idx for v in kicker_values]
        
        # Different adjustment calculations based on hand type
        if hand_name == "royal_flush":
            return max_adjustment  # Always max quality
            
        elif hand_name == "straight_flush":
            # Higher straights are better
            return max_adjustment * (max(hand_values) / max_value_idx)
            
        elif hand_name == "four_of_a_kind":
            # Higher quads are better
            quads_value = max(set(hand_values), key=hand_values.count)
            return max_adjustment * (quads_value / max_value_idx)
            
        elif hand_name == "full_house":
            # Higher trips component is more important
            counts = Counter(hand_values)
            trips_value = max(counts.keys(), key=lambda v: counts[v])
            return max_adjustment * (trips_value / max_value_idx)
            
        elif hand_name == "flush":
            # Higher cards in flush are better
            return max_adjustment * (max(hand_values) / max_value_idx)
            
        elif hand_name == "straight":
            # Higher straights are better
            return max_adjustment * (max(hand_values) / max_value_idx)
            
        elif hand_name == "three_of_a_kind":
            # Higher trips are better
            trips_value = max(set(hand_values), key=hand_values.count)
            return max_adjustment * (trips_value / max_value_idx)
            
        elif hand_name == "two_pair":
            # Higher top pair is more important
            # Get the two most common values
            counts = Counter(hand_values)
            pair_values = [v for v, c in counts.items() if c == 2]
            top_pair = max(pair_values)
            return max_adjustment * (top_pair / max_value_idx)
            
        elif hand_name == "pair":
            # Higher pair is better
            pair_value = max(set(hand_values), key=hand_values.count)
            return max_adjustment * (pair_value / max_value_idx)
            
        elif hand_name == "high_card":
            # Higher card is better
            return max_adjustment * (max(hand_values) / max_value_idx)
            
        return 0.0
    
    def _describe_hand(self, 
                      hand_name: str, 
                      hand_cards: List[Tuple[str, str]], 
                      kickers: List[Tuple[str, str]]) -> str:
        """
        Generate a human-readable description of the hand.
        
        Args:
            hand_name: Name of the hand
            hand_cards: Cards that form the hand
            kickers: Kicker cards
            
        Returns:
            Human-readable hand description
        """
        if not hand_cards:
            return "No valid hand"
            
        # Get values for description
        hand_values = [card[0] for card in hand_cards]
        
        # Convert face cards for display
        def card_name(value):
            if value == 'A':
                return 'Ace'
            elif value == 'K':
                return 'King'
            elif value == 'Q':
                return 'Queen'
            elif value == 'J':
                return 'Jack'
            else:
                return value
        
        if hand_name == "royal_flush":
            suit = hand_cards[0][1]
            return f"Royal Flush of {suit}"
            
        elif hand_name == "straight_flush":
            high_card = max(hand_cards, key=lambda c: CARD_VALUES.index(c[0]))
            suit = high_card[1]
            high_value = card_name(high_card[0])
            return f"{high_value}-high Straight Flush of {suit}"
            
        elif hand_name == "four_of_a_kind":
            quad_value = max(set(hand_values), key=hand_values.count)
            return f"Four of a Kind, {card_name(quad_value)}s"
            
        elif hand_name == "full_house":
            value_counts = Counter(hand_values)
            trip_value = max(value_counts.keys(), key=lambda v: value_counts[v])
            pair_value = min(value_counts.keys(), key=lambda v: value_counts[v])
            return f"Full House, {card_name(trip_value)}s full of {card_name(pair_value)}s"
            
        elif hand_name == "flush":
            suit = hand_cards[0][1]
            high_card = max(hand_cards, key=lambda c: CARD_VALUES.index(c[0]))
            high_value = card_name(high_card[0])
            return f"{high_value}-high Flush of {suit}"
            
        elif hand_name == "straight":
            high_card = max(hand_cards, key=lambda c: CARD_VALUES.index(c[0]))
            high_value = card_name(high_card[0])
            return f"{high_value}-high Straight"
            
        elif hand_name == "three_of_a_kind":
            trip_value = max(set(hand_values), key=hand_values.count)
            return f"Three of a Kind, {card_name(trip_value)}s"
            
        elif hand_name == "two_pair":
            value_counts = Counter(hand_values)
            pair_values = [v for v, count in value_counts.items() if count == 2]
            high_pair = max(pair_values, key=lambda v: CARD_VALUES.index(v))
            low_pair = min(pair_values, key=lambda v: CARD_VALUES.index(v))
            return f"Two Pair, {card_name(high_pair)}s and {card_name(low_pair)}s"
            
        elif hand_name == "pair":
            pair_value = max(set(hand_values), key=hand_values.count)
            return f"Pair of {card_name(pair_value)}s"
            
        elif hand_name == "high_card":
            high_card = max(hand_cards, key=lambda c: CARD_VALUES.index(c[0]))
            high_value = card_name(high_card[0])
            return f"{high_value} High"
            
        return "Unknown hand"
    
    def calculate_win_probability(self, 
                                 player_cards: List[Dict], 
                                 community_cards: List[Dict], 
                                 num_opponents: int = 1,
                                 num_simulations: int = 1000) -> Dict:
        """
        Calculate win probability through Monte Carlo simulation.
        
        Args:
            player_cards: List of player card dictionaries
            community_cards: List of community card dictionaries
            num_opponents: Number of opponents to simulate
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with probability information
        """
        # Convert to standardized format
        std_player_cards = self._standardize_cards(player_cards)
        std_community_cards = self._standardize_cards(community_cards)
        
        # For efficiency, limit simulations based on parameters
        if num_opponents > 2 and num_simulations > 500:
            num_simulations = 500
        elif num_opponents > 4 and num_simulations > 200:
            num_simulations = 200
        
        # Prepare result
        result = {
            "win_probability": 0.0,
            "tie_probability": 0.0,
            "loss_probability": 0.0,
            "confidence": 0.0
        }
        
        # If no player cards, return zero probabilities
        if len(std_player_cards) == 0:
            return result
        
        # Validate and filter community cards
        valid_community_cards = []
        for card in std_community_cards:
            if card[0] != "?" and card[0] in CARD_VALUES and card[1] in CARD_SUITS:
                valid_community_cards.append(card)
            else:
                logger.warning(f"Skipping invalid card in win probability calculation: {card}")
        
        # All known cards
        all_known = std_player_cards + valid_community_cards
        
        # Create a full deck
        full_deck = [(v, s) for v in CARD_VALUES for s in CARD_SUITS]
        
        # Remove known cards from deck
        available_cards = [card for card in full_deck if card not in all_known]
        
        if not available_cards:
            return result
            
        # Track simulation outcomes
        wins = 0
        ties = 0
        losses = 0
        
        # Run simulations
        for _ in range(num_simulations):
            # Shuffle remaining cards
            shuffled = available_cards.copy()
            random.shuffle(shuffled)
            
            # Determine how many community cards to deal
            cards_to_deal = 5 - len(valid_community_cards)
            if cards_to_deal < 0:
                cards_to_deal = 0
                
            # Deal community cards
            sim_community = valid_community_cards + shuffled[:cards_to_deal]
            shuffled = shuffled[cards_to_deal:]
            
            # Evaluate player's hand
            player_hand = self.evaluate_hand(
                [{"value": c[0], "suit": c[1]} for c in std_player_cards], 
                [{"value": c[0], "suit": c[1]} for c in sim_community]
            )
            
            # Create opponent hands
            opponent_results = []
            for i in range(num_opponents):
                # Deal 2 cards to opponent
                if len(shuffled) < 2:
                    break
                    
                opponent_cards = shuffled[:2]
                shuffled = shuffled[2:]
                
                # Evaluate opponent's hand
                opponent_hand = self.evaluate_hand(
                    [{"value": c[0], "suit": c[1]} for c in opponent_cards],
                    [{"value": c[0], "suit": c[1]} for c in sim_community]
                )
                
                opponent_results.append(opponent_hand)
            
            # Compare hand strengths
            player_strength = player_hand["hand_strength"]
            
            # Find strongest opponent
            max_opponent_strength = 0
            for opp_hand in opponent_results:
                max_opponent_strength = max(max_opponent_strength, opp_hand["hand_strength"])
            
            # Determine outcome
            if player_strength > max_opponent_strength:
                wins += 1
            elif player_strength == max_opponent_strength:
                ties += 1
            else:
                losses += 1
        
        # Calculate probabilities
        total = wins + ties + losses
        if total > 0:
            result["win_probability"] = wins / total
            result["tie_probability"] = ties / total
            result["loss_probability"] = losses / total
            
            # Confidence based on number of simulations
            result["confidence"] = min(1.0, num_simulations / 1000)
        
        # If we have missing/uncertain cards, adjust probability 
        if len(valid_community_cards) < len(std_community_cards):
            logger.info("Some community cards are uncertain, using a more conservative estimate")
            # Reduce confidence 
            result["confidence"] *= 0.8  # Reduce confidence
            
            # Adjust probabilities to be more realistic
            if result["win_probability"] > 0.7:
                result["win_probability"] = 0.7  # Cap at 70% with uncertain cards
        
        # Get current hand evaluation for consistency check
        current_hand = self.evaluate_hand(player_cards, community_cards)
        hand_strength = current_hand.get("hand_strength", 0)
        
        # If win probability seems unusually high compared to hand strength
        if result["win_probability"] > hand_strength * 2 and hand_strength < 0.4:
            logger.debug(f"Adjusting unusually high win probability from {result['win_probability']:.2f} to {hand_strength * 1.5:.2f}")
            result["win_probability"] = min(0.85, hand_strength * 1.5)
            
        # If win probability seems unusually low compared to hand strength
        if result["win_probability"] < hand_strength * 0.5 and hand_strength > 0.6:
            logger.debug(f"Adjusting unusually low win probability from {result['win_probability']:.2f} to {hand_strength * 0.75:.2f}")
            result["win_probability"] = max(0.1, hand_strength * 0.75)
        
        # Ensure probabilities sum to 1
        total_prob = result["win_probability"] + result["tie_probability"] + result["loss_probability"]
        if total_prob != 1.0:
            # Normalize
            result["win_probability"] /= total_prob
            result["tie_probability"] /= total_prob
            result["loss_probability"] /= total_prob
        
        # Log the computed win probability
        logger.info(f"Calculated win probability: {result['win_probability']:.2f} (confidence: {result['confidence']:.2f})")
        
        return result
    
    def hand_vs_range(self, 
                     player_cards: List[Dict], 
                     community_cards: List[Dict],
                     opponent_range: List[Tuple[str, str]],
                     num_simulations: int = 500) -> Dict:
        """
        Calculate win probability against a specific opponent range.
        
        Args:
            player_cards: List of player card dictionaries
            community_cards: List of community card dictionaries
            opponent_range: List of opponent starting hand tuples (e.g., [('A', 'A'), ('A', 'K')])
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with probability information
        """
        # Convert to standardized format
        std_player_cards = self._standardize_cards(player_cards)
        std_community_cards = self._standardize_cards(community_cards)
        
        # Prepare result
        result = {
            "win_probability": 0.0,
            "tie_probability": 0.0,
            "loss_probability": 0.0,
            "confidence": 0.0
        }
        
        # If no player cards or no opponent range, return zero probabilities
        if len(std_player_cards) == 0 or len(opponent_range) == 0:
            return result
            
        # All known cards
        all_known = std_player_cards + std_community_cards
        
        # Create a full deck
        full_deck = [(v, s) for v in CARD_VALUES for s in CARD_SUITS]
        
        # Remove known cards from deck
        available_cards = [card for card in full_deck if card not in all_known]
        
        # Track simulation outcomes
        wins = 0
        ties = 0
        losses = 0
        
        # Run simulations
        for _ in range(num_simulations):
            # Shuffle remaining cards
            shuffled = available_cards.copy()
            random.shuffle(shuffled)
            
            # Pick a random hand from opponent range
            range_hand = random.choice(opponent_range)
            
            # Create specific cards for the range hand
            opp_cards = []
            for value, suit_type in range_hand:
                # Find a matching card in the deck
                for card in shuffled:
                    if card[0] == value and (suit_type == 's' and card[1] == shuffled[0][1] or
                                           suit_type == 'o' and card[1] != shuffled[0][1] or
                                           suit_type == card[1]):
                        opp_cards.append(card)
                        shuffled.remove(card)
                        break
            
            # If couldn't create a valid opponent hand, skip this simulation
            if len(opp_cards) < 2:
                continue
                
            # Determine how many community cards to deal
            cards_to_deal = 5 - len(std_community_cards)
            if cards_to_deal < 0:
                cards_to_deal = 0
                
            # Deal community cards
            sim_community = std_community_cards + shuffled[:cards_to_deal]
            
            # Evaluate hands
            player_hand = self.evaluate_hand(
                [{"value": c[0], "suit": c[1]} for c in std_player_cards], 
                [{"value": c[0], "suit": c[1]} for c in sim_community]
            )
            
            opponent_hand = self.evaluate_hand(
                [{"value": c[0], "suit": c[1]} for c in opp_cards],
                [{"value": c[0], "suit": c[1]} for c in sim_community]
            )
            
            # Compare hand strengths
            player_strength = player_hand["hand_strength"]
            opponent_strength = opponent_hand["hand_strength"]
            
            # Determine outcome
            if player_strength > opponent_strength:
                wins += 1
            elif player_strength == opponent_strength:
                ties += 1
            else:
                losses += 1
        
        # Calculate probabilities
        total = wins + ties + losses
        if total > 0:
            result["win_probability"] = wins / total
            result["tie_probability"] = ties / total
            result["loss_probability"] = losses / total
            
            # Confidence based on number of simulations
            result["confidence"] = min(1.0, num_simulations / 1000)
        
        return result
    
    def calculate_pot_odds(self, 
                         current_pot: float, 
                         call_amount: float) -> float:
        """
        Calculate pot odds.
        
        Args:
            current_pot: Current pot size
            call_amount: Amount to call
            
        Returns:
            Pot odds as a ratio
        """
        if call_amount <= 0:
            return float('inf')  # Infinite odds if no call required
        
        # Pot odds as a ratio (pot : call)
        return current_pot / call_amount
    
    def should_call(self, 
                   win_probability: float, 
                   pot_odds: float, 
                   risk_factor: float = 1.0) -> bool:
        """
        Determine if a call is profitable based on pot odds.
        
        Args:
            win_probability: Probability of winning
            pot_odds: Pot odds (pot : call ratio)
            risk_factor: Adjustment for risk tolerance (< 1 more conservative, > 1 more aggressive)
            
        Returns:
            True if call is profitable, False otherwise
        """
        # Required win probability for call to be profitable
        required_win_prob = 1 / (pot_odds + 1)
        
        # Apply risk factor adjustment
        adjusted_win_prob = win_probability * risk_factor
        
        return adjusted_win_prob >= required_win_prob