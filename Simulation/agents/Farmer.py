import numpy as np
from together import Together
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define the schema for irrigation decision
class IrrigationDecision(BaseModel):
    fields_to_irrigate: int = Field(
        description="Number of fields to irrigate this season (integer between 0 and 10)",
        ge=0,
        le=10
    )
    reasoning: str = Field(
        description="Brief explanation for the irrigation decision"
    )

# ---------------------------
# Parameters
# ---------------------------
MAX_FIELDS_DECENTRALIZED = 10
DEMAND_THRESHOLD = 0.9
WATER_PER_FIELD = 50.0 # per month
FISH_INCOME_SCALE = 5
FARMER_INITIAL_BUDGET = 350
CONSUMPTION_COST = 15
IRRIGATION_COST = 6

class Farmer:
    def __init__(self, location, memory_strength, min_income):
        self.location = location
        self.irrigated_fields = 10
        self.expected_water = 250 # per farmer per month
        self.memory = []
        self.memory_strength = memory_strength
        self.yield_history = []
        self.catch_history = []
        self.irrigated_fields_history = []
        self.budget = FARMER_INITIAL_BUDGET
        self.min_income = min_income
        self.planned_fields = self.irrigated_fields
        self.monthly_water_received = []
        self.july_memory = [] 

    def receive_water(self, amount):
            self.monthly_water_received.append(amount)

    def predict_water(self): # water prediction equation (eq1)
        if not self.july_memory:
            return self.expected_water
        if self.memory_strength == 0:
            if self.july_memory:
                return self.july_memory[-1]  # use last July's actual received value
        weights = np.array([self.memory_strength ** i for i in range(len(self.july_memory))])[::-1]
        self.expected_water = np.dot(weights, self.july_memory) / weights.sum()

        return self.expected_water # monthly value

    def decide_irrigation(self):
        if not self.yield_history:
            return

        # constraints
        expected_annual_water = self.predict_water() * 12 # expected water
        max_water_fields = int(expected_annual_water / (WATER_PER_FIELD * 12)) # water availability
        max_affordable_fields = max(0, int(self.budget // IRRIGATION_COST) - CONSUMPTION_COST)

        max_possible = min(max_water_fields, max_affordable_fields, MAX_FIELDS_DECENTRALIZED)
        
        last_yield = self.yield_history[-1] if self.yield_history else 0.0 
        last_satisfaction = self.memory[-1] if self.memory else 0.0 

        # Rule 1: Below minimum income → increase fields
        if last_yield < self.min_income:
            self.irrigated_fields = min(self.irrigated_fields + 1, max_possible)

        # Rule 2: Income satisfied but demand not met → keep fields safe for expected water
        elif last_yield >= self.min_income and last_satisfaction < DEMAND_THRESHOLD:
            self.irrigated_fields = min(max_water_fields, max_possible)

        # Rule 3: Otherwise → full use of constraints
        else:
            self.irrigated_fields = max(max_possible, 0)
    
    def decide_irrigation_generative_agent(self, num_farmers=9, provider="openai"):
        """
        Use LLM with structured output to decide irrigation based on current context
        Supports 'together' (Together AI) and 'openai' (OpenAI o4-mini) providers
        """
        if provider.lower() == "openai":
            return self._decide_with_openai(num_farmers)
        else:  # default to together
            return self._decide_with_together(num_farmers)
    
    def _decide_with_together(self, num_farmers=9):
        """Use Together AI for irrigation decision"""
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            print("TOGETHER_API_KEY not found, falling back to heuristic method.")
            self.decide_irrigation()
            return
            
        client = Together(api_key=api_key)
        
        # Get context information
        predicted_water = self.predict_water()
        water_received_last_year = self.july_memory[-1] if self.july_memory else 0
        max_fields = MAX_FIELDS_DECENTRALIZED
        print(f"Calling LLM for farmer at location {self.location}) ")
        # Determine relative position
        if self.location == 0:
            relative_position = "You are the most upstream farmer"
        elif self.location == num_farmers - 1:
            relative_position = "You are the most downstream farmer"
        else:
            relative_position = f"You have {num_farmers - self.location - 1} farmers downstream from you"
        
        # Create the prompt
        prompt = f"""You are a farmer agent deciding how many fields to irrigate this season.
Your decision affects not only your own harvest but also the availability of water for farmers located downstream along the river. The more water you use, the less water remains for others.

Context:

Your location along the river: {self.location}

Current budget: {self.budget:.2f}

Maximum number of fields possible: {max_fields}

Predicted water availability at your location: {predicted_water:.2f}

Observed water received last year: {water_received_last_year:.2f}

Total number of farmers along the river: {num_farmers}

Your position relative to downstream farmers: {relative_position}

Reminder:
Water is a common-pool resource. If you irrigate more fields, you increase your own potential yield but reduce water availability for downstream farmers.

Question:
Given this information, how many fields do you want to irrigate this season? Provide your decision and reasoning."""

        try:
            # Make API call with structured output
            response = client.chat.completions.create(
                model="Qwen/QwQ-32B",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a rational farmer making irrigation decisions. Consider both your own economic needs and the water needs of downstream farmers. Only answer in JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "schema": IrrigationDecision.model_json_schema(),
                }
            )
            
            # Parse the structured response
            output = json.loads(response.choices[0].message.content)
            fields_to_irrigate = output.get("fields_to_irrigate", 0)
            reasoning = output.get("reasoning", "No reasoning provided")
            
            # Ensure within valid bounds
            fields_to_irrigate = max(0, min(fields_to_irrigate, max_fields))
            self.irrigated_fields = fields_to_irrigate
            
            # Optional: print reasoning for debugging
            # print(f"Farmer {self.location}: {fields_to_irrigate} fields - {reasoning}")
                
        except Exception as e:
            # If API call fails, fallback to heuristic method
            print(f"LLM call failed for farmer {self.location}: {e}")
            self.decide_irrigation()

    def _decide_with_openai(self, num_farmers=9):
        """Use OpenAI o4-mini for irrigation decision"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not found, falling back to heuristic method.")
            self.decide_irrigation()
            return
            
        client = OpenAI(api_key=api_key)
        
        # Get context information
        predicted_water = self.predict_water()
        water_received_last_year = self.july_memory[-1] if self.july_memory else 0
        max_fields = MAX_FIELDS_DECENTRALIZED
        print(f"Calling OpenAI o4-mini for farmer at location {self.location}")
        
        # Determine relative position
        if self.location == 0:
            relative_position = "You are the most upstream farmer"
        elif self.location == num_farmers - 1:
            relative_position = "You are the most downstream farmer"
        else:
            relative_position = f"You have {num_farmers - self.location - 1} farmers downstream from you"
        
        # Create the prompt for OpenAI o4-mini
        prompt = f"""You are a farmer agent deciding how many fields to irrigate this season.
                    Your decision affects not only your own harvest but also the availability of water for farmers located downstream along the river. The more water you use, the less water remains for others.

                    Context:

                    Your location along the river: {self.location}

                    Current budget: {self.budget:.2f}

                    Maximum number of fields possible: {max_fields}

                    Predicted water availability at your location: {predicted_water:.2f}

                    Observed water received last year: {water_received_last_year:.2f}

                    Total number of farmers along the river: {num_farmers}

                    Your position relative to downstream farmers: {relative_position}

                    Reminder:
                    Water is a common-pool resource. If you irrigate more fields, you increase your own potential yield but reduce water availability for downstream farmers.

                    Question:
                    Given this information, how many fields do you want to irrigate this season? Please respond with a JSON object containing "fields_to_irrigate" (integer 0-10) and "reasoning" (string explanation)."""

        try:
            # Make API call to OpenAI o4-mini
            response = client.responses.create(
                model="o4-mini-2025-04-16",
                input=prompt
            )
            
            # Parse the response - o4-mini returns text that we need to parse as JSON
            response_text = response.output_text.strip()
            
            # Try to extract JSON from the response
            try:
                # Look for JSON-like content in the response
                if "{" in response_text and "}" in response_text:
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    json_str = response_text[json_start:json_end]
                    output = json.loads(json_str)
                    
                    fields_to_irrigate = output.get("fields_to_irrigate", 0)
                    reasoning = output.get("reasoning", "No reasoning provided")
                else:
                    # If no JSON found, try to extract number from text
                    import re
                    numbers = re.findall(r'\b(\d+)\b', response_text)
                    fields_to_irrigate = int(numbers[0]) if numbers else 5  # default to 5
                    reasoning = "Extracted from text response"
                    
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, extract number from text or use default
                import re
                numbers = re.findall(r'\b(\d+)\b', response_text)
                fields_to_irrigate = int(numbers[0]) if numbers else 5
                reasoning = "Fallback parsing from text"
            
            # Ensure within valid bounds
            fields_to_irrigate = max(0, min(fields_to_irrigate, max_fields))
            self.irrigated_fields = fields_to_irrigate
            
            # Optional: print reasoning for debugging
            # print(f"Farmer {self.location}: {fields_to_irrigate} fields - {reasoning}")
                
        except Exception as e:
            # If API call fails, fallback to heuristic method
            print(f"OpenAI call failed for farmer {self.location}: {e}")
            self.decide_irrigation()

    def irrigate(self, available_water): # return received water per month     
        self.planned_fields = self.irrigated_fields  # store for stress calc
        demand = self.irrigated_fields * (WATER_PER_FIELD)
        received = min(available_water, demand)
        self.monthly_water_received.append(received)
        return received

    def calculate_crop_stress(self): # irrigation equation (eq2)
        if len(self.monthly_water_received) < 12: # pad with zeros if incomplete
            self.monthly_water_received += [0] * (12 - len(self.monthly_water_received))

        stress_sum = 0
        for m in range(3, 9):  # April (3) to September (8)
            V_R = self.monthly_water_received[m]
            V_D = self.irrigated_fields * 50 # per month
            ratio = V_R / V_D if V_D > 0 else 0
            stress_sum += ratio

        average_stress = stress_sum / 6.0
        Y_jt = 10 * self.irrigated_fields * min(1.0, average_stress)
        return Y_jt

    def update_budget_and_yield(self, fish_catch, centralized=False): # budget equation (eq3)       
        field_yield = self.calculate_crop_stress()

        if centralized: # Farmers keep ONLY fishing income
            net_return = fish_catch * FISH_INCOME_SCALE # net return per farmer
            income = field_yield + fish_catch * FISH_INCOME_SCALE
            net_return = income - (self.irrigated_fields * IRRIGATION_COST) - CONSUMPTION_COST
            authority_yield = field_yield
            authority_irrigation_cost = self.irrigated_fields * IRRIGATION_COST
            authority_consumption_cost = CONSUMPTION_COST
        else: # Farmers pay for everything
            income = field_yield + fish_catch * FISH_INCOME_SCALE
            net_return = income - (self.irrigated_fields * IRRIGATION_COST) - CONSUMPTION_COST
            authority_yield = 0
            authority_irrigation_cost = 0
            authority_consumption_cost = 0

        self.budget += net_return
        self.yield_history.append(field_yield) # total yield (not per field)
        self.catch_history.append(fish_catch)
        self.irrigated_fields_history.append(self.irrigated_fields)

        # Update water satisfaction memory (average over the season)
        total_demand = self.irrigated_fields * 12
        total_received = sum(self.monthly_water_received)
        satisfaction = total_received / total_demand if total_demand > 0 else 0
        self.memory.append(satisfaction)
        if len(self.memory) > 5:
            self.memory.pop(0)
        self.monthly_water_received = []

        return authority_yield, authority_irrigation_cost, authority_consumption_cost
