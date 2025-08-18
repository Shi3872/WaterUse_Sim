import numpy as np

# ---------------------------
# Parameters
# ---------------------------
MAX_FIELDS_CENTRALIZED = 90
WATER_PER_FIELD = 50.0 # per month
AUTHORITY_INITIAL_BUDGET = 1800
CONSUMPTION_COST = 20
IRRIGATION_COST = 6

class NationalAuthority:
    def __init__(self, memory_strength):
        self.expected_water = 25000
        self.memory = []
        self.memory_strength = memory_strength
        self.budget = AUTHORITY_INITIAL_BUDGET
        self.net_returns = [] # net returns per farmer
        self.july_memory = []
    
    def predict_water(self):
        if not self.july_memory:
            return self.expected_water
        if self.memory_strength == 0: 
            if self.july_memory:
                return self.july_memory[-1]  # use last July's actual received value
        weights = np.array([self.memory_strength ** i for i in range(len(self.july_memory))])[::-1]
        self.expected_water = np.dot(weights, self.july_memory) / weights.sum()
        return self.expected_water # monthly value

    def decide_total_fields(self, farmers):
        expected_annual_inflow = self.predict_water() * 12

        # constraints
        max_water_fields = int(expected_annual_inflow / (WATER_PER_FIELD * 12))
        max_affordable_fields = int((self.budget // (IRRIGATION_COST)) - CONSUMPTION_COST)
        
        max_possible = min(max_water_fields, max_affordable_fields, MAX_FIELDS_CENTRALIZED)

        return max_possible

    def allocate_fields(self, farmers):
        total_fields = self.decide_total_fields(farmers)                                                                                                            
        equal_fields = total_fields // len(farmers) if total_fields > 0 else 0
        for farmer in farmers:
            farmer.irrigated_fields = min(equal_fields, MAX_FIELDS_CENTRALIZED)

        #if self.budget < len(farmers) * (CONSUMPTION_COST + IRRIGATION_COST):
            #print("⚠️ Centralized authority has collapsed due to insufficient budget.")
            #raise RuntimeError("Centralized authority collapse.")
