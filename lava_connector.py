"""
LLM Interface Module for Poker AI

This module interfaces with the Lava 1.6 34B LLM model via Ollama
to provide advanced reasoning for complex poker decisions.
"""

import logging
import time
import json
import requests
from typing import Dict, List, Tuple, Optional, Any
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LavaConnector:
    """
    Interface to the Lava LLM for poker strategy reasoning.
    """
    
    def __init__(self, 
                 api_base: str = "http://localhost:11434/api",
                 model_name: str = "lava",
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 request_timeout: int = 60,
                 system_message: Optional[str] = None):
        """
        Initialize the LLM interface.
        
        Args:
            api_base: Base URL for Ollama API
            model_name: Name of the model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            request_timeout: Timeout for API requests in seconds
            system_message: Optional system message
        """
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        
        # Default system message if none provided
        self.system_message = system_message or self._default_system_message()
        
        # Track response times
        self.request_times = []
        self.max_request_history = 10
        
        # Check if the model is available
        self.model_available = self._check_model_availability()
        if not self.model_available:
            logger.warning(f"Model {model_name} not available. Some features will be limited.")
    
    def _default_system_message(self) -> str:
        """
        Create a default system message for poker strategy.
        
        Returns:
            Default system message
        """
        return """You are a world-class poker strategy advisor. Your role is to analyze 
the current poker situation and provide strategic advice based on:

1. The player's cards and community cards
2. The current game state (pot size, player stacks, etc.)
3. The betting history and available actions
4. Your knowledge of optimal poker strategy

Respond with clear, concise analysis and specific action recommendations.
Always explain your reasoning and include quantitative assessments where possible.
Focus on maximizing expected value (EV) in each decision.
"""
    
    def _check_model_availability(self) -> bool:
        """
        Check if the specified model is available.
        
        Returns:
            True if available, False otherwise
        """
        try:
            url = f"{self.api_base}/tags"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                models = response.json()
                model_names = [model["name"] for model in models.get("models", [])]
                return self.model_name in model_names
            return False
            
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    def analyze_decision(self, 
                        game_state: Dict, 
                        available_actions: List[str],
                        hand_eval: Dict = None,
                        opponent_info: Dict = None) -> Dict:
        """
        Get strategic advice for the current poker decision.
        
        Args:
            game_state: Current game state
            available_actions: Available actions
            hand_eval: Hand evaluation results
            opponent_info: Opponent analysis information
            
        Returns:
            Response dictionary with advice and reasoning
        """
        if not self.model_available:
            return {
                "success": False,
                "message": f"Model {self.model_name} not available",
                "recommendation": "Use built-in strategy",
                "reasoning": "LLM not available"
            }
            
        # Build the prompt
        prompt = self._create_decision_prompt(game_state, available_actions, hand_eval, opponent_info)
        
        try:
            # Measure request time
            start_time = time.time()
            
            # Make the API request
            response = self._generate(prompt)
            
            # Record request time
            elapsed = time.time() - start_time
            self._record_request_time(elapsed)
            
            if response:
                # Parse the response
                parsed = self._parse_decision_response(response)
                parsed["success"] = True
                parsed["response_time"] = elapsed
                return parsed
            else:
                return {
                    "success": False,
                    "message": "Failed to get response from LLM",
                    "recommendation": "Use built-in strategy",
                    "reasoning": "LLM error"
                }
                
        except Exception as e:
            logger.error(f"Error getting LLM decision: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "recommendation": "Use built-in strategy",
                "reasoning": "LLM error"
            }
    
    def _create_decision_prompt(self, 
                               game_state: Dict, 
                               available_actions: List[str],
                               hand_eval: Optional[Dict] = None,
                               opponent_info: Optional[Dict] = None) -> str:
        """
        Create a prompt for poker decision analysis.
        
        Args:
            game_state: Current game state
            available_actions: Available actions
            hand_eval: Hand evaluation results
            opponent_info: Opponent analysis information
            
        Returns:
            Formatted prompt string
        """
        # Extract key information
        player_cards = game_state.get("player_cards", [])
        community_cards = game_state.get("community_cards", [])
        pot = game_state.get("pot", 0)
        player_balance = game_state.get("player_balance", 0)
        game_phase = game_state.get("game_phase", "unknown")
        current_bet = game_state.get("current_bet", 0)
        
        # Format cards for readability
        player_card_str = ", ".join([card.get("code", "??") for card in player_cards]) or "Unknown"
        community_card_str = ", ".join([card.get("code", "??") for card in community_cards]) or "None"
        
        # Hand evaluation info
        hand_info = ""
        if hand_eval:
            hand_info = f"""
Hand evaluation:
- Hand: {hand_eval.get('hand_description', 'Unknown')}
- Hand strength: {hand_eval.get('hand_strength', 0):.2f} (0-1 scale)
- Win probability: {hand_eval.get('win_probability', 0):.2f}
"""
        
        # Opponent info
        opponent_str = ""
        if opponent_info:
            opponent_data = []
            for opp_id, opp in opponent_info.items():
                if isinstance(opp, dict):
                    opp_type = opp.get("player_type", "unknown")
                    vpip = opp.get("vpip", 0) if isinstance(opp, dict) else 0
                    pfr = opp.get("pfr", 0) if isinstance(opp, dict) else 0
                    opponent_data.append(f"{opp_id}: {opp_type} (VPIP: {vpip:.2f}, PFR: {pfr:.2f})")
            
            if opponent_data:
                opponent_str = "Opponent information:\n- " + "\n- ".join(opponent_data)
        
        # Build the full prompt
        prompt = f"""Analyze this poker situation and recommend the best action:

Game state:
- Phase: {game_phase}
- Player cards: {player_card_str}
- Community cards: {community_card_str}
- Pot size: ${pot}
- Player balance: ${player_balance}
- Current bet to call: ${current_bet}
- Available actions: {', '.join(available_actions)}

{hand_info}
{opponent_str}

Please provide:
1. A clear action recommendation (one of: {', '.join(available_actions)})
2. If raising, suggest a bet amount
3. Brief explanation of your reasoning
4. Confidence level in your recommendation (low/medium/high)

Format your response as:
ACTION: [recommended action]
AMOUNT: [bet amount if raising, otherwise N/A]
REASONING: [your strategic reasoning]
CONFIDENCE: [low/medium/high]
"""
        
        return prompt
    
    def _parse_decision_response(self, response: str) -> Dict:
        """
        Parse the LLM response into structured data.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Structured response dictionary
        """
        # Initialize default values
        result = {
            "recommendation": "check",  # Default to safest action
            "bet_amount": None,
            "reasoning": "Failed to parse LLM response",
            "confidence": "low"
        }
        
        try:
            # Use regex to extract structured information
            action_match = re.search(r'ACTION:\s*(\w+)', response, re.IGNORECASE)
            amount_match = re.search(r'AMOUNT:\s*(\$?[\d.]+|N\/A)', response, re.IGNORECASE)
            reasoning_match = re.search(r'REASONING:\s*(.*?)(?=\nCONFIDENCE:|$)', response, re.IGNORECASE | re.DOTALL)
            confidence_match = re.search(r'CONFIDENCE:\s*(\w+)', response, re.IGNORECASE)
            
            # Extract matches
            if action_match:
                action = action_match.group(1).lower()
                # Validate action (fold, check, call, raise)
                if action in ["fold", "check", "call", "raise"]:
                    result["recommendation"] = action
                    
            if amount_match:
                amount_str = amount_match.group(1)
                if amount_str.lower() != "n/a" and amount_str != "$0":
                    # Remove $ and convert to float
                    amount_str = amount_str.replace("$", "")
                    try:
                        amount = float(amount_str)
                        if amount > 0:
                            result["bet_amount"] = amount
                    except ValueError:
                        pass
                        
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                if reasoning:
                    result["reasoning"] = reasoning
                    
            if confidence_match:
                confidence = confidence_match.group(1).lower()
                if confidence in ["low", "medium", "high"]:
                    result["confidence"] = confidence
                    
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            result["reasoning"] = f"Error parsing response: {str(e)}"
            
        # If a raise is recommended but no amount specified, use default
        if result["recommendation"] == "raise" and result["bet_amount"] is None:
            result["bet_amount"] = 0  # The strategy engine will use a default amount
            
        return result
    
    def _generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        try:
            url = f"{self.api_base}/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": self.system_message,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                url, 
                json=payload,
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"API error: {response.status_code}, {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return ""
    
    def _record_request_time(self, elapsed: float) -> None:
        """
        Record LLM request time for monitoring.
        
        Args:
            elapsed: Time elapsed in seconds
        """
        self.request_times.append(elapsed)
        
        # Keep only recent history
        if len(self.request_times) > self.max_request_history:
            self.request_times.pop(0)
            
        # Log average response time
        avg_time = sum(self.request_times) / len(self.request_times)
        logger.debug(f"LLM request time: {elapsed:.2f}s (avg: {avg_time:.2f}s)")
    
    def analyze_hand_history(self, hand_history: List[Dict]) -> Dict:
        """
        Analyze hand history for insights and learning.
        
        Args:
            hand_history: List of game state dictionaries
            
        Returns:
            Analysis and insights
        """
        if not self.model_available or not hand_history:
            return {
                "success": False,
                "message": "Model not available or empty hand history",
                "insights": []
            }
            
        # Build a summary of the hand
        summary = self._create_hand_summary(hand_history)
        
        prompt = f"""Analyze this poker hand and provide strategic insights:

Hand summary:
{summary}

Provide 3-5 specific insights about the gameplay, including:
1. Key decision points and whether optimal choices were made
2. Missed opportunities or mistakes
3. Patterns that could be exploited in future hands
4. Recommendations for improvement

Format each insight as a separate numbered point.
"""
        
        try:
            # Generate insights
            response = self._generate(prompt)
            
            if response:
                # Parse insights
                insights = self._parse_insights(response)
                return {
                    "success": True,
                    "message": "Hand analysis complete",
                    "insights": insights
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to get response from LLM",
                    "insights": []
                }
                
        except Exception as e:
            logger.error(f"Error analyzing hand history: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "insights": []
            }
    
    def _create_hand_summary(self, hand_history: List[Dict]) -> str:
        """
        Create a summary of a hand for analysis.
        
        Args:
            hand_history: List of game state dictionaries
            
        Returns:
            Hand summary string
        """
        # Extract key events from the hand history
        summary_lines = []
        
        # Initial state
        if hand_history:
            initial_state = hand_history[0]
            phase = initial_state.get("game_phase", "unknown")
            player_cards = initial_state.get("player_cards", [])
            player_card_str = ", ".join([card.get("code", "??") for card in player_cards]) or "Unknown"
            
            summary_lines.append(f"Starting phase: {phase}")
            summary_lines.append(f"Player cards: {player_card_str}")
            
        # Track community cards and pot through the hand
        community_cards_by_phase = {}
        actions_by_phase = {}
        
        for state in hand_history:
            phase = state.get("game_phase", "unknown")
            community_cards = state.get("community_cards", [])
            
            # Track community cards by phase
            if community_cards and phase not in community_cards_by_phase:
                community_card_str = ", ".join([card.get("code", "??") for card in community_cards])
                community_cards_by_phase[phase] = community_card_str
                
            # Track actions by phase
            if "last_action" in state:
                action = state["last_action"]
                if phase not in actions_by_phase:
                    actions_by_phase[phase] = []
                actions_by_phase[phase].append(action)
        
        # Add community cards by phase
        for phase, cards in community_cards_by_phase.items():
            summary_lines.append(f"{phase.capitalize()} community cards: {cards}")
            
        # Add actions by phase
        for phase, actions in actions_by_phase.items():
            action_str = "; ".join([f"{a.get('actor', 'Player')}: {a.get('action', 'unknown')}" +
                                  (f" ${a.get('amount')}" if a.get('amount') is not None else "")
                                  for a in actions])
            summary_lines.append(f"{phase.capitalize()} actions: {action_str}")
            
        # Final state
        if hand_history:
            final_state = hand_history[-1]
            final_pot = final_state.get("pot", 0)
            final_balance = final_state.get("player_balance", 0)
            
            summary_lines.append(f"Final pot: ${final_pot}")
            summary_lines.append(f"Final player balance: ${final_balance}")
            
            # Add outcome if available
            if "outcome" in final_state:
                outcome = final_state["outcome"]
                summary_lines.append(f"Outcome: {outcome}")
        
        return "\n".join(summary_lines)
    
    def _parse_insights(self, response: str) -> List[str]:
        """
        Parse insights from the LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Split by numbered points and clean up
        lines = response.strip().split("\n")
        current_insight = ""
        
        for line in lines:
            # If this line starts a new insight
            if re.match(r'^\d+\.', line):
                # Save the previous insight if it exists
                if current_insight:
                    insights.append(current_insight.strip())
                    
                # Start a new insight
                current_insight = line
            else:
                # Continue current insight
                current_insight += " " + line.strip()
                
        # Don't forget the last insight
        if current_insight:
            insights.append(current_insight.strip())
            
        return insights
    
    def get_hand_range_advice(self, 
                             position: str, 
                             num_players: int,
                             tournament_stage: str = "middle") -> Dict:
        """
        Get advice on starting hand ranges for a given position.
        
        Args:
            position: Table position (early, middle, late, button, blinds)
            num_players: Number of players at the table
            tournament_stage: Tournament stage (early, middle, late)
            
        Returns:
            Hand range advice
        """
        if not self.model_available:
            return {
                "success": False,
                "message": "Model not available",
                "range": "",
                "explanation": ""
            }
            
        prompt = f"""Provide a starting hand range recommendation for Texas Hold'em:

Position: {position}
Number of players: {num_players}
Tournament stage: {tournament_stage}

Please specify:
1. A specific hand range recommendation in standard notation
2. A brief explanation of the strategy for this position

Format your response as:
RANGE: [hand range notation]
EXPLANATION: [strategic explanation]
"""
        
        try:
            response = self._generate(prompt)
            
            if response:
                # Parse the response
                range_match = re.search(r'RANGE:\s*(.*?)(?=\nEXPLANATION:|$)', response, re.IGNORECASE | re.DOTALL)
                explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?=$)', response, re.IGNORECASE | re.DOTALL)
                
                range_text = range_match.group(1).strip() if range_match else ""
                explanation = explanation_match.group(1).strip() if explanation_match else ""
                
                return {
                    "success": True,
                    "range": range_text,
                    "explanation": explanation
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to get response from LLM",
                    "range": "",
                    "explanation": ""
                }
                
        except Exception as e:
            logger.error(f"Error getting hand range advice: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "range": "",
                "explanation": ""
            }
    
    def get_average_response_time(self) -> float:
        """
        Get the average LLM response time.
        
        Returns:
            Average response time in seconds
        """
        if not self.request_times:
            return 0.0
        return sum(self.request_times) / len(self.request_times)
    
    def update_settings(self, 
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       system_message: Optional[str] = None) -> None:
        """
        Update LLM settings.
        
        Args:
            temperature: New temperature setting
            max_tokens: New max tokens setting
            system_message: New system message
        """
        if temperature is not None:
            self.temperature = max(0.1, min(1.0, temperature))
            
        if max_tokens is not None:
            self.max_tokens = max(32, min(8192, max_tokens))
            
        if system_message is not None:
            self.system_message = system_message
            
        logger.info(f"Updated LLM settings: temp={self.temperature}, max_tokens={self.max_tokens}")
    
    def is_available(self) -> bool:
        """
        Check if the LLM is currently available.
        
        Returns:
            True if available, False otherwise
        """
        # Refresh availability status
        self.model_available = self._check_model_availability()
        return self.model_available