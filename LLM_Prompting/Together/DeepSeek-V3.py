"""
DeepSeek-V3 Integration with Pure Pydantic Structured Response

This module uses Together AI's structured response capability with Pydantic models
to ensure reliable JSON output from the DeepSeek-V3 language model.

Key features:
- Guaranteed valid JSON structure through Toon_schema response format
- Type validation and data integrity through Pydantic models
- Better error handling and debugging capabilities
- Pure Pydantic data structures throughout

The structured response uses Pydantic models to define:
- PayoffMatrix: Structured payoff values for 2x2 games
- PlayerActions: Available actions for each player
- ActionSituation: Complete game scenario with all IAD framework elements
- GameScenariosResponse: Container for multiple action situations

All functions return and work with pure Pydantic model instances or their
validated dictionary representations.
"""

from together import Together
from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Union

load_dotenv() # load .env 
api_key = os.getenv("TOGETHER_API_KEY") # access your key
client = Together(api_key=api_key)

# Get the directory of the current file to make paths relative
current_dir = os.path.dirname(os.path.abspath(__file__))
txts_dir = os.path.join(current_dir, "..", "Txts")

with open(os.path.join(txts_dir, "odd.txt"), "r", encoding="utf-8") as f:
    odd = f.read()

with open(os.path.join(txts_dir, "game_stuff.txt"), "r", encoding="utf-8") as f:
    game = f.read()


# Define Pydantic models for structured response
class PayoffMatrix(BaseModel):
    """Payoff matrix for a two-player game"""
    action1_action1: List[float] = Field(description="Payoffs when both players choose action1 [player1_payoff, player2_payoff]")
    action1_action2: List[float] = Field(description="Payoffs when player1 chooses action1, player2 chooses action2")
    action2_action1: List[float] = Field(description="Payoffs when player1 chooses action2, player2 chooses action1")
    action2_action2: List[float] = Field(description="Payoffs when both players choose action2")

class PlayerActions(BaseModel):
    """Actions available to each player"""
    player1: List[str] = Field(description="List of actions available to player 1")
    player2: List[str] = Field(description="List of actions available to player 2")

class ActionSituation(BaseModel):
    """A single action situation with game-theoretic analysis"""
    title: str = Field(description="A concise name summarizing the action situation")
    description: str = Field(description="Detailed description of the action situation")
    location: str = Field(description="The physical or institutional setting where the action situation occurs")
    players: List[str] = Field(description="The agents involved (e.g., farmers, regulators)")
    roles: List[str] = Field(description="The roles players occupy (e.g., irrigator, enforcer)")
    actions: PlayerActions = Field(description="The set of choices available to each player")
    control_rules: str = Field(description="How actions lead to outcomes")
    information: str = Field(description="What information players have access to")
    outcomes: str = Field(description="The results of actions")
    payoffs: str = Field(description="The consequences for each player")
    strategic_tension: str = Field(description="Description of strategic dilemmas and game type")
    temporal_structure: str = Field(description="Whether interaction is one-shot, repeated, or continuous")
    relevant_rules: str = Field(description="Explicit boundary, position, choice, or control rules")
    payoff_matrix: PayoffMatrix = Field(description="The game payoff matrix with realistic values")
    strategic_core: str = Field(description="Game type description (e.g., prisoner's dilemma, coordination game)")
    rationale: str = Field(description="Why payoffs make sense and comply with ODD+D description")

class GameScenariosResponse(BaseModel):
    """Complete response containing all action situations"""
    action_situations: List[ActionSituation] = Field(
        description="List of distinct action situations with strategic tensions",
        min_items=1
    )


prompt = f"""
Given the following ODD+D description of a water use model:

{odd}

Extract all **distinct action situations** described in the model using the IAD framework. Each action situation should reflect a **unique strategic tension**.
To help inspire diverse and concrete strategic tensions, consider parallels to the following sustainability-related games, each with its own type of dilemma:

- **Cooperation, Coordination, and Conflict Game** – In this set of games standard 2-player games on cooperation and coordination are framed in a natural resource management context. With this set of games we can see how small changes in the payoff matrix affect the nature of the social dilemma and expected outcomes.
- **Game of Trust** – With this game we can measure trust and truthworthiness. The trust game is played by two players, where the first player have to decide how much to trust player 2 by giving this player an amount of his/her resources. The amount given to player 2 will be increased and player 2 will demonstrate it’s trustworthiness by deciding how much to give back to player 1 and how much to keep for him/herself.
- **Public Goods Game** – This game is the classic public good game which captures a dilemma between what is good for the individual and for the group. An example is contributing labor to a community project. If everyone contributes to the project, everybody benefits, but if one person freerides and does not contribute labor, that person will still get the benefits.
- **Common Pool Resource Game** – In this game participants share a common resource. We can create the situation of over-harvesting, popularly known as the “tragedy of the common”.
- **Channel Irrigation Game** In this game players need to take water use decision in the face of water scarcity. They can chose to grow water efficient or water consumptive crops while water is only sufficient for every player to grow the water efficient crop.
- **Watershed Game** – In this game participants experience asymmetries in a watershed where upstream players experience different problems and incentives compared to those downstream. The players make decisions on land use and whether they want to provide or accept payments to compensate the consequences of upstream players.
- **Negotiations Game** – In this game players will be in a large group but make each round a decision in a 2-player game where one player has control and propose a solution to a negotiation, and the other player can accept. Each round the player will play with somebody else, and we will see whether this distribution negotiation lead to a group level agreement.
- **Dam Maintenance Game** – In this game players need to jointly contribute to a water harvesting infrastructure from which all group members receive benefits. The game is based on a public good game.
- **Surface Water Game** – This game features typical water management challenges faced by local communities. It is based on a common pool resource games. Players can take decisions on contributing to a water harvesting structure which makes water available for the group and water appropriation decisions framed as the choice of crops with different water efficiencies.
- **Irrigation Game** – In this game participants have to make decisions to invest in the irrigation system maintenance and to extract water for irrigating their individual plots.
- **Fishery Game** – In this game participants have to decide where to fish and how much to fish.


### Output Instructions
For each action situation, specify the following elements from IAD in detail:

1. **Title** – A concise name summarizing the action situation  
2. **Location** – The physical or institutional setting where the action situation occurs (e.g., farm level, river basin, regulatory office)  
3. **Players** – The agents involved (e.g., farmers, regulators), as defined by boundary rules  
4. **Roles** – The roles players occupy (e.g., irrigator, enforcer, allocator, predictor)  
5. **Actions** – The set of choices available to each player or role, per choice rules  
6. **Control Rules** – How actions lead to outcomes, including any deterministic or stochastic effects  
7. **Information** – What information players have access to (e.g., past inflow data, peer behavior). Is it complete, partial, or noisy?  
8. **Outcomes** – The results of actions (e.g., crop yield, budget change, fish population)  
9. **Payoffs** – The consequences for each player, including economic, institutional, or ecological impacts  
10. **Strategic Tension** – Does the situation involve any dilemmas between players? Clearly specify if the interaction is **strategic** or **non-strategic**. If the action situation is strategic, specify the type of game that would be used to model it. 
    Prioritize the sustainability games mentioned. Include a short description of how/why there's this tension in this action situation.
11. **Temporal Structure** – Is the interaction one-shot, repeated annually, or continuous over time?  
12. **Relevant Rules** – Describe any explicit boundary, position, choice, or control rules involved in the action situation

If the situation is strategic, turn the situation into a two-player normal form game.
Include the game description, the players, the actions, and a payoff matrix with realistic values.

Ensure all game elements make logical sense:
    - The available actions for each player must be rational, context-appropriate, and economically or behaviorally realistic.
    - The payoffs should reflect the likely consequences of each combined action (e.g., coordination failure, overuse, free-riding, trust, punishment, etc.).
    - Do not include unrealistic or illogical actions unless justified.
Avoid symmetric payoff duplication unless the game is truly symmetric.

After listing all action situations and their initial payoff matrices:

- **Analyze the strategic core** of each situation (e.g., is it a coordination game, prisoner’s dilemma, asymmetric conflict, etc.).
- Explicitly **compare** all strategic action situations. If two or more have similar:
  - Player roles or decision types
  - Payoff structures or incentive logic
  - Upstream farmer and downstream farmer differences
    In a decentralized regime, an upstream farmer has access to water first, while a downstream farmer accesses fish first
    In a centralized regime, the national authority allocates the same amount of water to each farmer
  - Social dilemmas (e.g., both are CPR or public goods)

Then you **must revise or replace** one of them to ensure strategic diversity.

Aim for **at least 4 clearly distinct types** of strategic dilemmas (e.g., tragedy of the commons).
Your goal is to produce a set of action situations where each one reflects a **different game-theoretic logic**, not just different labels.

For each game:
- Ensure that payoff values are grounded in the action situation: what does each agent gain or lose?.
- There should be tension between the payoff values.
- Briefly explain why each outcome in the payoff matrix makes sense.


Some important points to consider:
    - Players are rational
    - In decentralized water use games, consider spatial asymmetries:
        - If one farmer is upstream, their actions can impact downstream users, but not vice versa.
        - Payoffs must reflect these hydrological relationships even in simultaneous-move games.
    - Do not assume actions are made in isolation. Use the control rules and spatial logic from the model to define interaction effects.
    - Ensure payoffs reflect environmental feedback (e.g., water scarcity, fish population decline, budget constraints, yield effects).
    - Avoid identical or symmetric payoff matrices unless the situation explicitly supports it.
Specify why or why not each game complies with the odd+d description. If it does not, revise the game to make it compliant with the odd+d protocol. 
The revised game should be the only ones shown in the actual output.

Please use clear, structured formatting. Number each action situation. Do not repeat strategic tensions across situations.

Most important: For each strategic action situation, the payoff matrix should use the structured format defined in the schema.

Each game should represent a distinct strategic tension with realistic payoffs (numbers between 0 and 100) grounded in the water use context.
"""

def generate_game_scenarios_and_payoffs(save_to_file=True, output_file="game_matrices.txt"):
    """Generate Action Situations and save full JSON structure to file
    
    Args:
        save_to_file (bool): Whether to save results to file
        output_file (str): Output file path
        
    Returns:
        dict: Generated game scenarios data
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in institutional analysis, game theory, agent-based modeling, and IAD/ODD+D frameworks. You must respond with valid JSON that matches the provided schema exactly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            response_format={
                "type": "json_schema",
                "schema": GameScenariosResponse.model_json_schema(),
            }
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse and validate JSON response using Pydantic
        try:
            scenarios_data = json.loads(content)
            # Validate against Pydantic model
            validated_response = GameScenariosResponse(**scenarios_data)
            game_result = validated_response.model_dump()
            
            # Save full JSON structure to file if requested
            if save_to_file:
                save_full_scenarios_to_file(game_result, output_file)
            
            return game_result
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            raise ValueError(f"Response validation failed: {e}")
            
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")

def extract_upstream_downstream_matrix(scenarios_data=None, file_path="game_matrices.txt"):
    """Extract upstream-downstream irrigation scenario from file or provided data
    
    Args:
        scenarios_data (dict, optional): Game scenarios data. If None, reads from file.
        file_path (str): Path to file containing scenarios data
        
    Returns:
        ActionSituation: Upstream-downstream scenario as Pydantic model
    """
    try:
        # If no data provided, read from file
        if scenarios_data is None:
            scenarios_data = read_scenarios_from_file(file_path)
        
        # Validate and convert to Pydantic model
        validated_scenarios = GameScenariosResponse(**scenarios_data)
        
        # Look for upstream-downstream scenarios in the action situations
        for situation in validated_scenarios.action_situations:
            title = situation.title.lower()
            description = situation.description.lower()
            players = [str(player).lower() for player in situation.players]
            
            # Check if this is an upstream-downstream scenario
            is_upstream_downstream = (
                ("upstream" in title and "downstream" in title) or
                ("upstream" in description and "downstream" in description) or
                any("upstream" in player for player in players) or
                any("downstream" in player for player in players) or
                ("irrigation" in title and ("spatial" in description or "asymmetric" in description))
            )
            
            if is_upstream_downstream:
                return situation
        
        raise ValueError("No upstream-downstream irrigation scenario found")
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract upstream-downstream matrix: {e}")

def read_scenarios_from_file(file_path="game_matrices.txt"):
    """Read game scenarios from file
    
    Args:
        file_path (str): Path to file containing scenarios (relative to Txts directory)
        
    Returns:
        dict: Game scenarios data
    """
    try:
        # Ensure file path points to Txts directory
        if not os.path.isabs(file_path):
            file_path = os.path.join(txts_dir, file_path)
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract JSON from file content
        # Look for ALL_SCENARIOS = {...}
        import re
        pattern = r'ALL_SCENARIOS = ({.*})'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        else:
            raise ValueError("No ALL_SCENARIOS found in file")
            
    except Exception as e:
        raise RuntimeError(f"Failed to read scenarios from file {file_path}: {e}")



def convert_json_matrix_to_tuple_keys(matrix_json):
    """Convert JSON matrix with string keys to tuple keys for backward compatibility"""
    result = {}
    for key, value in matrix_json.items():
        # Convert string key like "(L, H)" to tuple key like ("L", "H")
        if isinstance(key, str) and key.startswith("(") and key.endswith(")"):
            # Parse string tuple representation
            clean_key = key.strip("()")
            actions = [action.strip().strip("'\"") for action in clean_key.split(",")]
            if len(actions) == 2:
                result[(actions[0], actions[1])] = tuple(value) if isinstance(value, list) else value
        else:
            # Direct assignment if key is already in correct format
            result[key] = tuple(value) if isinstance(value, list) else value
    return result

def save_full_scenarios_to_file(game_result, output_file="game_matrices.txt"):
    """Save full game scenarios JSON structure to file
    
    Args:
        game_result (dict): Complete game scenarios data
        output_file (str): Output file path (relative to Txts directory)
        
    Returns:
        bool: True if successful, False otherwise
    """
    from datetime import datetime
    
    try:
        # Ensure output file is saved in the Txts directory
        if not os.path.isabs(output_file):
            output_file = os.path.join(txts_dir, output_file)
        
        # Format content with full JSON structure
        formatted_content = f"""# STRATEGIC IRRIGATION GAME SCENARIOS
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Source: DeepSeek-V3 Structured Response (Pure Pydantic)
# Status: VALIDATED_PYDANTIC_MATRICES

# Full Game Scenarios (JSON format):
ALL_SCENARIOS = {json.dumps(game_result, indent=2)}

# Matrix Type: Comprehensive Action Situations with Pydantic Structure
"""
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(formatted_content)
        
        print(f"✓ Saved full game scenarios to {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error saving scenarios to file: {e}")
        return False


def get_pydantic_payoff_matrix(action_situation):
    """Extract payoff matrix from Pydantic ActionSituation as structured dictionary
    
    Args:
        action_situation (ActionSituation): Pydantic model instance
        
    Returns:
        dict: Structured payoff matrix data
    """
    if not isinstance(action_situation, ActionSituation):
        raise ValueError("Expected ActionSituation Pydantic model")
    
    return {
        "action1_action1": action_situation.payoff_matrix.action1_action1,
        "action1_action2": action_situation.payoff_matrix.action1_action2,
        "action2_action1": action_situation.payoff_matrix.action2_action1,
        "action2_action2": action_situation.payoff_matrix.action2_action2,
        "player1_actions": action_situation.actions.player1,
        "player2_actions": action_situation.actions.player2
    }


def get_all_action_situations(scenarios_data=None, file_path="game_matrices.txt"):
    """Get all action situations as Pydantic models
    
    Args:
        scenarios_data (dict, optional): Game scenarios data. If None, reads from file.
        file_path (str): Path to file containing scenarios data
        
    Returns:
        List[ActionSituation]: List of validated Pydantic action situation models
    """
    try:
        # If no data provided, read from file
        if scenarios_data is None:
            scenarios_data = read_scenarios_from_file(file_path)
        
        # Validate and convert to Pydantic model
        validated_scenarios = GameScenariosResponse(**scenarios_data)
        return validated_scenarios.action_situations
        
    except Exception as e:
        raise RuntimeError(f"Failed to get action situations: {e}")



if __name__ == "__main__":
    try:
        # Generate comprehensive scenarios and save full JSON structure
        print("Generating game scenarios with structured response...")
        scenarios_data = generate_game_scenarios_and_payoffs(save_to_file=True, output_file="game_matrices.txt")
        
        print(f"✓ Generated {len(scenarios_data.get('action_situations', []))} action situations")
        print("✓ Full JSON structure saved to game_matrices.txt")
        
        # Extract upstream-downstream specific scenario from saved file
        print("Extracting upstream-downstream scenario from file...")
        upstream_downstream_scenario = extract_upstream_downstream_matrix(file_path="game_matrices.txt")
        
        print("✓ Successfully extracted upstream-downstream scenario")
        print(f"Upstream-Downstream Scenario: {upstream_downstream_scenario.title}")
        print(f"Strategic Core: {upstream_downstream_scenario.strategic_core}")
        
        # Show payoff matrix in structured Pydantic format
        payoff_matrix = upstream_downstream_scenario.payoff_matrix
        print(f"Payoff Matrix (Pydantic format):")
        print(f"  Both choose action1: {payoff_matrix.action1_action1}")
        print(f"  Player1 action1, Player2 action2: {payoff_matrix.action1_action2}")
        print(f"  Player1 action2, Player2 action1: {payoff_matrix.action2_action1}")
        print(f"  Both choose action2: {payoff_matrix.action2_action2}")
        
        # Show available actions
        actions = upstream_downstream_scenario.actions
        print(f"Player Actions:")
        print(f"  Player 1: {actions.player1}")
        print(f"  Player 2: {actions.player2}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise