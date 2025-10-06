SYSTEM_PROMPT_FARMER = (
    "You are a purely self-interested player who always seeks to maximize your own gain and ensure that the outcome is as favorable as possible for yourself. \
          Answer only in JSON format."
)

TOGETHER_MODEL_STRING = "Qwen/QwQ-32B"
OPENAI_MODEL_STRING = "gpt-4o-mini"
import numpy as np
from Simulation.agents.Farmer import Farmer
from Simulation.agents.Authority import NationalAuthority
from Simulation.agents.Fish import FishPopulation
from Simulation.solver import generate_dv_matrix, generate_cv_matrix, solve_game
import math
from together import Together
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import importlib.util
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define the schema for game theory decision
class GameDecision(BaseModel):
    strategy_choice: str = Field(
        description="Choose either 'L' for Low irrigation (conservative) or 'H' for High irrigation (aggressive)",
        pattern="^[LH]$"
    )
    fields_to_irrigate: int = Field(
        description="Number of fields to irrigate (integer between 0 and 10)",
        ge=0,
        le=10
    )
    reasoning: str = Field(
        description="Brief explanation for the irrigation strategy choice"
    )

# ---------------------------
# Parameters
# ---------------------------
MAX_FIELDS_DECENTRALIZED = 10
MAX_FIELDS_CENTRALIZED = 90
WATER_PER_FIELD = 50.0 # per month
FISH_INCOME_SCALE = 5
LARVAE_INFLOW_THRESHOLD = 2000 
AUTHORITY_INITIAL_BUDGET = 1800
CONSUMPTION_COST = 15 # annual
IRRIGATION_COST = 6
DEFAULT_TARGET_CATCH = 20

# ---------------------------

class WaterResource:
    def __init__(self, inflow_series, inflow_case="3"):
        self.inflow_series = inflow_series
        self.index = 0 # keep track of year sim is currently on
        self.inflow_case = inflow_case

    def next_year_inflow(self): # called once per year to get monthly inflow values
        if self.index < len(self.inflow_series):
            annual_inflow = self.inflow_series[self.index]
            self.index += 1
            
            if self.inflow_case == "4":
                # For case 4: start with annual_inflow/12 and gradually decline to 15000
                first_month_flow = annual_inflow / 12.0
                final_month_flow = 1250
                
                # Create a gradual decline with some randomness
                #np.random.seed(self.index)  # Use index for consistent randomness per year
                
                # Base linear decline
                base_decline = np.linspace(first_month_flow, final_month_flow, 12)

                # Add random variation (±20% of the decline amount)
                variation_factor = 0.2
                random_variations = np.random.normal(0, variation_factor, 12)
                decline_amount = first_month_flow - final_month_flow

                monthly_inflows = base_decline + (random_variations * decline_amount)
                return monthly_inflows
            else:
                # Original behavior for other cases
                monthly_inflows = np.full(12, annual_inflow / 12.0) #divide annual inflow across 12 months
                return monthly_inflows

class Simulation:
    def __init__(self, years=10, centralized=False, fishing_enabled=True, print_interval=1, memory_strength=0, use_cpr_game = False, use_static_game = False, generative_agent=False, llm_provider="together", use_fishing_cpr=False, simulate_tragedy=True):
        self.years = years
        self.farmers = [Farmer(location=i, memory_strength=1, min_income=30) for i in range(9)]
        for f in self.farmers:
            f.fishing_enabled = fishing_enabled
        self.print_interval = print_interval
        self.centralized = centralized
        self.authority = NationalAuthority(memory_strength=memory_strength) if centralized else None
        self.fish = FishPopulation()
        self.water = WaterResource(np.random.uniform(1000, 2500, years))
        self.fishing_enabled = fishing_enabled
        self.farmer_budget_history = [[] for _ in self.farmers]
        self.authority_budget_history = []
        self.annual_fish_totals = [] # total fish each year
        self.july_inflows = [] # inflow value for July (index 6)
        self.predicted_water_history = []
        self.fish_history = []
        self.use_cpr_game = use_cpr_game
        self.use_static_game = use_static_game
        self.generative_agent = generative_agent
        self.llm_provider = llm_provider
        self.use_fishing_cpr = use_fishing_cpr
        self.simulate_tragedy = simulate_tragedy

    def centralized_game(self, monthly_inflows):
            n_farmers = len(self.farmers)
            total_water = sum(monthly_inflows)
            s_threshold = int(total_water / (12 * WATER_PER_FIELD))

            # Build payoff matrix
            payoffs = generate_cv_matrix(
                n=10,
                m=1,
                water_field=WATER_PER_FIELD,
                total_water=total_water,
                yield_field=8,
                cost_per_field=IRRIGATION_COST,
                consumption_cost=CONSUMPTION_COST,
                stress_threshold=s_threshold,
                stressed_yield=3,
                authority_budget=self.authority.budget,
                n_farmers=n_farmers
            )

            equilibria = solve_game(payoffs, player_labels=("Authority", "Environment"))

            if equilibria:
                eq = np.random.choice(equilibria)

                auth_probs = eq["Authority"]
                if len(auth_probs) != 10:
                    auth_probs = np.ones(10) / 10.0  # fallback uniform

                # Authority strategy choice (game outcome)
                per_farmer_choice = int(np.random.choice(range(1, 11), p=auth_probs))

                # Compute feasibility once (same as in cv_irrigation)
                requested_total = per_farmer_choice * n_farmers
                affordable_total = max(0, self.authority.budget) // IRRIGATION_COST
                WATER_PER_FIELD_YEARLY = WATER_PER_FIELD * 12
                water_possible_total = total_water // WATER_PER_FIELD_YEARLY
                feasible_total = min(affordable_total, water_possible_total)

                # If feasible, assign equally; if not, cap to feasible
                if requested_total <= feasible_total:
                    final_alloc = per_farmer_choice
                else:
                    final_alloc = feasible_total // n_farmers

                for f in self.farmers:
                    f.irrigated_fields = final_alloc
                    
    def fishing_cpr_game(self):
        """
        Play fishing Common Pool Resource games between farmer pairs
        Returns a dictionary mapping farmer locations to target catch strategies
        """
        fishing_strategies = {}
        
        # Pair farmers and play CPR games
        for i in range(len(self.farmers) - 1):
            uf = self.farmers[i]
            df = self.farmers[i + 1]
            
            # Create fishing CPR payoff matrix
            # Strategies: different catch levels (e.g., 10, 15, 20, 25, 30)
            catch_levels = np.arange(0, DEFAULT_TARGET_CATCH+1, 5)
            payoffs = []
            payoff_scale_factor, congestion_factor = 3.0, 0.1
            for uf_catch in catch_levels:
                row = []
                for df_catch in catch_levels:
                    # Simple model: fish catch reduces fish population, affecting income
                    if not self.simulate_tragedy:
                        uf_utility = payoff_scale_factor * uf_catch - congestion_factor*(uf_catch + df_catch)*uf_catch
                        df_utility = payoff_scale_factor * df_catch - congestion_factor*(uf_catch + df_catch)*df_catch
                    else:
                        uf_utility = payoff_scale_factor * uf_catch
                        df_utility = payoff_scale_factor * df_catch

                    row.append((uf_utility, df_utility))
                payoffs.append(row)
            
            # Solve the fishing CPR game
            equilibria = solve_game(payoffs, player_labels=("UF_Fish", "DF_Fish"))
            
            if equilibria:
                eq = np.random.choice(equilibria)
                
                uf_choice = np.random.choice(catch_levels, p=eq["UF_Fish"])
                df_choice = np.random.choice(catch_levels, p=eq["DF_Fish"])
                
                fishing_strategies[uf.location] = uf_choice
                fishing_strategies[df.location] = df_choice
        
        return fishing_strategies

    def _make_llm_game_decisions(self, uf, df, uf_fish_income, df_fish_income, total_water):
        """
        Use LLM to make game theory decisions for two farmers.
        Reads pre-generated payoff matrices from file.
        Returns tuple of (uf_choice, df_choice) as field counts
        """
        # Load payoff matrices using DeepSeek-V3 function
        payoff_matrix_display = self._load_payoff_matrix_from_deepseek()
        
        # Make decisions for both farmers using the loaded matrix
        uf_choice = self._get_farmer_llm_decision(uf, df, uf_fish_income, total_water, payoff_matrix_display, "upstream")
        df_choice = self._get_farmer_llm_decision(df, uf, df_fish_income, total_water, payoff_matrix_display, "downstream")
        
        return uf_choice, df_choice
    
    def _load_payoff_matrix_from_deepseek(self):
        """
        Load irrigation payoff matrix using DeepSeek-V3 extract_upstream_downstream_matrix function.
        """
        try:
            # Import DeepSeek-V3 module dynamically
            deepseek_path = os.path.join(os.path.dirname(__file__), '..', '..', 'LLM_Prompting', 'Together', 'DeepSeek-V3.py')
            spec = importlib.util.spec_from_file_location("deepseek_v3", deepseek_path)
            deepseek_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(deepseek_module)
            
            # Use the new extract_upstream_downstream_matrix function
            matrices_file = os.path.join(os.path.dirname(__file__), '..', '..', 'LLM_Prompting', 'Txts', 'game_matrices.txt')
            upstream_downstream_scenario = deepseek_module.extract_upstream_downstream_matrix(file_path=matrices_file)
            
            print(f"Loaded Pydantic scenario: {upstream_downstream_scenario.title}")
            
            # Convert Pydantic model to display format
            return self._format_pydantic_payoff_matrix_for_display(upstream_downstream_scenario)
                
        except Exception as e:
            raise RuntimeError(f"Error loading payoff matrices using DeepSeek-V3: {e}")
    
    def _format_pydantic_payoff_matrix_for_display(self, action_situation):
        """
        Convert Pydantic ActionSituation model to display format for LLM prompts.
        """
        try:
            # Extract payoffs from Pydantic model
            payoffs = action_situation.payoff_matrix
            actions = action_situation.actions
            
            # Get action names (first two actions for each player)
            player1_actions = actions.player1[:2] if len(actions.player1) >= 2 else ["Low", "High"]
            player2_actions = actions.player2[:2] if len(actions.player2) >= 2 else ["Low", "High"]
            
            # Map to L/H for simplicity
            action1_short = "L"  # First action (typically conservative)
            action2_short = "H"  # Second action (typically aggressive)
            
            # Extract payoffs from Pydantic structure
            ll_payoff = payoffs.action1_action1  # Both choose first action
            lh_payoff = payoffs.action1_action2  # Player1 first, Player2 second
            hl_payoff = payoffs.action2_action1  # Player1 second, Player2 first
            hh_payoff = payoffs.action2_action2  # Both choose second action
            
            return f"""
        Payoff Matrix (Your Payoff, Other's Payoff):
        
                    Other Farmer
                    L     H
        You    L   {ll_payoff} {lh_payoff}
               H   {hl_payoff} {hh_payoff}
        
        L = {player1_actions[0]} (conservative)
        H = {player1_actions[1]} (aggressive)
        
        Scenario: {action_situation.title}
        Strategic Core: {action_situation.strategic_core}
        """
        except Exception as e:
            raise RuntimeError(f"Error formatting Pydantic payoff matrix: {e}")
    
    def _get_farmer_llm_decision(self, farmer, other_farmer, fish_income, total_water, payoff_matrix, position):
        """
        Get LLM decision for a single farmer in a game theory context
        """
        provider = self.llm_provider if hasattr(self, 'llm_provider') else "together"
        
        if provider.lower() == "openai":
            return self._get_farmer_decision_openai(farmer, other_farmer, fish_income, total_water, payoff_matrix, position)
        else:
            return self._get_farmer_decision_together(farmer, other_farmer, fish_income, total_water, payoff_matrix, position)
    
    def _get_farmer_decision_together(self, farmer, other_farmer, fish_income, total_water, payoff_matrix, position):
        """Use Together AI for game theory decision"""
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            print("TOGETHER_API_KEY not found, falling back to random choice.")
            return np.random.choice([6, 10])  # Random L or H choice
            
        client = Together(api_key=api_key)
        prompt = f"""You are Farmer {farmer.location} making an irrigation decision in a strategic setting with Farmer {other_farmer.location}.

            Context:
            - Your current budget: {farmer.budget:.2f}
            - Other farmer's budget: {other_farmer.budget:.2f}
            - Your fish income: {fish_income:.2f}
            - Total water available: {total_water:.2f}
            - Your position: {position} farmer
            - Irrigation cost per field: {IRRIGATION_COST}
            - You are in a strategic interaction where your choice affects both your payoff and the other farmer's payoff

            {payoff_matrix}

            Game Theory Context:
            This is a  interaction. First, You must choose between:
            - L (Low): Conservative strategy, potentially better for cooperation
            - H (High): Aggressive strategy, potentially better for competition

            Then based on your strategy, specify the exact number of fields between 0 (min) and 10 (max) you want to irrigate.

            Your task: Specify the exact number of fields (0-10) you want to irrigate.
            """

        try:
            response = client.chat.completions.create(
                model=TOGETHER_MODEL_STRING,
                messages=[
                    {
                        "role": "system", 
                        "content": SYSTEM_PROMPT_FARMER
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "schema": GameDecision.model_json_schema(),
                }
            )
            output = json.loads(response.choices[0].message.content)
            strategy = output.get("strategy_choice", "L")
            fields = output.get("fields_to_irrigate", 6)
            reasoning = output.get("reasoning", "No reasoning provided")
            fields = max(0, min(fields, 10))
            print(f"Farmer {farmer.location} ({position}): Strategy {strategy}, {fields} fields - {reasoning}")
            return fields
        except Exception as e:
            print(f"LLM call failed for farmer {farmer.location}: {e}")
            return np.random.choice([6, 10])  # Fallback to random L or H
    
    def _get_farmer_decision_openai(self, farmer, other_farmer, fish_income, total_water, payoff_matrix, position):
        """Use OpenAI for game theory decision"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not found, falling back to random choice.")
            return np.random.choice([6, 10])  # Random L or H choice
            
        client = OpenAI(api_key=api_key)
        
        prompt = f"""You are Farmer {farmer.location} making an irrigation decision in a strategic game with Farmer {other_farmer.location}.

                Context:
                - Your current budget: {farmer.budget:.2f}
                - Other farmer's budget: {other_farmer.budget:.2f}
                - Your fish income: {fish_income:.2f}
                - Total water available: {total_water:.2f}
                - Your position: {position} farmer
                - You are in a strategic interaction where your choice affects both your payoff and the other farmer's payoff

                {payoff_matrix}

                Game Theory Context:
                This is a classic strategic interaction. You must choose between:
                - L (Low): Conservative strategy, irrigate 6 fields, potentially better for cooperation
                - H (High): Aggressive strategy, irrigate 10 fields, potentially better for competition

                Consider:
                1. What strategy maximizes your expected payoff?
                2. What might the other farmer choose?
                3. Should you cooperate (both choose L) or compete (choose H)?

                Your task: Choose either L or H strategy, and specify the exact number of fields (0-10) you want to irrigate.
                """

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL_STRING,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a farmer making strategic irrigation decisions in a game theory context. Consider both cooperation and competition. Answer only in JSON format."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            output = json.loads(response.choices[0].message.content)
            strategy = output.get("strategy_choice", "L")
            fields = output.get("fields_to_irrigate", 6)
            reasoning = output.get("reasoning", "No reasoning provided")
            fields = max(0, min(fields, 10))
            print(f"Farmer {farmer.location} ({position}): Strategy {strategy}, {fields} fields - {reasoning}")
            return fields
        except Exception as e:
            print(f"LLM call failed for farmer {farmer.location}: {e}")
            return np.random.choice([6, 10])  # Fallback to random L or H

    def decentralized_game(self, monthly_inflows):
        remaining_water = sum(monthly_inflows)
        for i in range(len(self.farmers) - 1):
            uf = self.farmers[i]
            df = self.farmers[i + 1]

            if uf.budget <= 0:
                uf_choice = 0
                uf.possible_choices.append(uf_choice)
            if df.budget <= 0:
                df_choice = 0
                df.possible_choices.append(df_choice)

            total_water_for_this_game = remaining_water
            s_threshold = int(remaining_water / (12 * WATER_PER_FIELD))

            uf_fish_income = FISH_INCOME_SCALE * (np.mean(uf.catch_history[-3:]) if len(uf.catch_history) >= 3 else (uf.catch_history[-1] if uf.catch_history else 0))
            df_fish_income = FISH_INCOME_SCALE * (np.mean(df.catch_history[-3:]) if len(df.catch_history) >= 3 else (df.catch_history[-1] if df.catch_history else 0))

            if self.use_static_game:
                # LLM-based game theory decision
                uf_choice, df_choice = self._make_llm_game_decisions(uf, df, uf_fish_income, df_fish_income, total_water_for_this_game)
                uf.possible_choices.append(uf_choice)
                df.possible_choices.append(df_choice)
            else:
                payoffs = generate_dv_matrix(
                    n=10,
                    m=1,
                    water_field=WATER_PER_FIELD,
                    total_water=total_water_for_this_game,
                    yield_field=8,
                    cost_per_field=IRRIGATION_COST,
                    consumption_cost=CONSUMPTION_COST,
                    stress_threshold=s_threshold,
                    stressed_yield=3,
                    uf_budget=uf.budget,
                    df_budget=df.budget,
                    uf_fish_income=uf_fish_income,
                    df_fish_income=df_fish_income,
                    simulate_tragedy=self.simulate_tragedy
                )
                matrix_size = 10
                strategy_values = list(range(1, matrix_size + 1))

                equilibria = solve_game(payoffs, player_labels=("UF", "DF"))

                if equilibria:
                    eq = np.random.choice(equilibria)

                    if uf.budget > 0:
                        uf_choice = np.random.choice(strategy_values, p=eq["UF"])
                        uf.possible_choices.append(uf_choice)

                    if df.budget > 0:
                        df_choice = np.random.choice(strategy_values, p=eq["DF"])
                        df.possible_choices.append(df_choice)

            uf_water_used = uf_choice * WATER_PER_FIELD * 12
            remaining_water = max(0, remaining_water - uf_water_used)

        for farmer in self.farmers:
            if len(farmer.possible_choices) == 1:
                farmer.irrigated_fields = farmer.possible_choices[0]
            else:
                farmer.irrigated_fields = np.random.choice(farmer.possible_choices)
            

    def run(self):
        for year in range(self.years):
            monthly_inflows = self.water.next_year_inflow()
            july_inflow = monthly_inflows[6]
            print(f"\n--- Year {year + 1} ---")
            #---------------Decision logic----------------#

            if self.centralized and self.authority: # Centralized
                if self.use_cpr_game: # Game logic
                    self.centralized_game(monthly_inflows)
                else: # Heuristic logic
                    predicted = self.authority.predict_water()           
                    self.predicted_water_history.append(predicted)
                    self.authority.allocate_fields(self.farmers)

            for farmer in self.farmers:
                farmer.monthly_water_received = []
                farmer.possible_choices = []
            
            if not self.centralized: # Decentralized
                if self.use_cpr_game or self.use_static_game: # Game logic
                    self.decentralized_game(monthly_inflows)
                else: # Heuristic logic
                    for farmer in self.farmers:
                        farmer.predict_water()
                        if not self.generative_agent:
                            farmer.decide_irrigation()
                        else:
                            farmer.decide_irrigation_generative_agent(provider=self.llm_provider)
                            
            if year % self.print_interval == 0:
                # self.use_cpr_game == False and self.use_static_game == False and self.centralized == True:
                print(f"\nYear {year + 1} | Total Inflow: {sum(monthly_inflows):.2f}")
                lake = 0

            annual_usage = {f.location: 0.0 for f in self.farmers}
            annual_allocated = {f.location: 0.0 for f in self.farmers}
            runoff_factor = 1
            total_runoff = 0

            # -------------Water allocation-------------- #
            for month, inflow in enumerate(monthly_inflows): # allcoate water month by month
                if self.centralized: # Centralized
                    total_fields = sum(f.irrigated_fields for f in self.farmers)
                    monthly_demand = total_fields * WATER_PER_FIELD

                    water_remaining = inflow
                    if monthly_demand > 0:
                        for farmer in self.farmers:
                            demand = farmer.irrigated_fields * WATER_PER_FIELD
                            alloc = min(demand, water_remaining)
                            farmer.receive_water(alloc)
                            annual_usage[farmer.location] += alloc
                            annual_allocated[farmer.location] += alloc
                            water_remaining -= alloc
                    else:
                        for farmer in self.farmers:
                            farmer.receive_water(0)

                    monthly_used = sum(farmer.irrigated_fields * WATER_PER_FIELD for farmer in self.farmers)
                    monthly_runoff = max(0, inflow - monthly_used)
                    total_runoff += monthly_runoff

                else: # Decentralized
                    water_remaining = inflow
                    for farmer in sorted(self.farmers, key=lambda f: f.location):
                        received = farmer.irrigate(water_remaining)
                        water_remaining -= received
                        annual_usage[farmer.location] += received
                    monthly_runoff = max(0, water_remaining)
                    total_runoff += monthly_runoff

            # after allocation, record this year's inflow
            if self.centralized and self.authority:
                self.authority.july_memory.append(july_inflow)
                #if len(self.authority.july_memory) > 10:
                    #self.authority.july_memory.pop(0)

            for farmer in self.farmers:
                if len(farmer.monthly_water_received) >= 7:  # how much each farmer receives in july
                    farmer.july_memory.append(farmer.monthly_water_received[6])
                    #if len(farmer.july_memory) > 10: # if memory too long, pop 
                        #farmer.july_memory.pop(0)

            lake += total_runoff * runoff_factor

            if self.centralized:
                # All farmers are the same — print only once
                loc = self.farmers[0].location
                used = annual_usage[loc]
                allocated = annual_allocated.get(loc, used)
                if year % self.print_interval == 0:
                    #if self.use_cpr_game == False and self.use_static_game == False and self.centralized == True:
                    print(f"Per farmer: total water used = {used:.2f}, allocated = {allocated:.2f}, remaining inflow: {total_runoff:.2f}")


            # ---------------- Fish Dynamics ---------------- #
            may_inflow = monthly_inflows[3]
            self.fish.grow(lake_water=lake, may_inflow=may_inflow)

            total_fish = sum(self.fish.age_classes)
            adult_fish = sum(self.fish.age_classes[5:])
            juvenile_fish = sum(self.fish.age_classes[1:5])
            larvae = self.fish.age_classes[0]

            if year % self.print_interval == 0:
                #print(f"Lake Water = {lake:.2f}")
                #if self.use_cpr_game == False and self.use_static_game == False and self.centralized == True:
                print(f"Fish Status — Total: {total_fish}, Adults: {adult_fish}, Juveniles: {juvenile_fish}, Larvae: {larvae}")

            total_yield = 0
            total_irrigation = 0
            total_consumption = 0

            # Fishing logic - either CPR game or fixed target catch
            if self.use_fishing_cpr and self.fishing_enabled:
                # Play fishing CPR games between farmer pairs
                fishing_strategies = self.fishing_cpr_game()
                
                # Apply fishing strategies and harvest
                for farmer in sorted(self.farmers, key=lambda f: -f.location):
                    if getattr(farmer, 'fishing_enabled', True):
                        target_catch = fishing_strategies.get(farmer.location, DEFAULT_TARGET_CATCH)  # default if not in game
                        fish_catch = self.fish.harvest(target_catch)
                    else:
                        fish_catch = 0
                    y, ci, cc = farmer.update_budget_and_yield(fish_catch=fish_catch, centralized=self.centralized)
                    total_yield += y
                    total_irrigation += ci
                    total_consumption += cc
            else:
                # Original fishing logic - fixed target catch
                for farmer in sorted(self.farmers, key=lambda f: -f.location):
                    if getattr(farmer, 'fishing_enabled', True):
                        target_catch = DEFAULT_TARGET_CATCH
                        fish_catch = self.fish.harvest(target_catch)
                    else:
                        fish_catch = 0
                    y, ci, cc = farmer.update_budget_and_yield(fish_catch=fish_catch, centralized=self.centralized)
                    total_yield += y
                    total_irrigation += ci
                    total_consumption += cc

            for i, f in enumerate(self.farmers): # append budget
                    self.farmer_budget_history[i].append(f.budget)

            # --------------------budget calc----------------#
            if self.centralized and self.authority:
                self.authority.budget += total_yield - total_irrigation - total_consumption
                self.authority.net_returns = [total_yield, total_irrigation, total_consumption]
                self.authority_budget_history.append(self.authority.budget) # append budget
                if year % self.print_interval == 0:
                    #if self.use_cpr_game == False and self.use_static_game == False and self.centralized == True:
                    print(f"National Authority Budget = {self.authority.budget:.2f} "
                        f"(Income: {total_yield:.2f}, Irrigation Cost: {total_irrigation:.2f}, Consumption Cost: {total_consumption:.2f}) \n")

            for i, f in enumerate(self.farmers):
                if year % self.print_interval == 0:
                    #if self.use_cpr_game == False and self.use_static_game == False and self.centralized == True:
                    print(f"Farmer {i+1}: Fields={f.irrigated_fields}, Budget={f.budget:.2f}, "
                        f"Last Yield={f.yield_history[-1]:.2f}, Catch={int(f.catch_history[-1])}")
            
            self.annual_fish_totals.append(sum(self.fish.age_classes))
            self.july_inflows.append(self.water.inflow_series[year] / 12.0)  # July inflow assumed uniform
            self.fish_history.append(list(self.fish.age_classes))


