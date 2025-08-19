import numpy as np

# ---------------------------
# Parameters
# ---------------------------
MAX_FIELDS_DECENTRALIZED = 10
DEMAND_THRESHOLD = 0.9
WATER_PER_FIELD = 50.0 # per month
FISH_INCOME_SCALE = 5
FARMER_INITIAL_BUDGET = 350
CONSUMPTION_COST = 20
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

        # Update water satisfaction memory (average over the season)
        total_demand = self.irrigated_fields * 12
        total_received = sum(self.monthly_water_received)
        satisfaction = total_received / total_demand if total_demand > 0 else 0
        self.memory.append(satisfaction)
        if len(self.memory) > 5:
            self.memory.pop(0)
        self.monthly_water_received = []

        return authority_yield, authority_irrigation_cost, authority_consumption_cost
