class NationalAuthority:
    def __init__(self, memory_strength):
        self.expected_water = 40000
        self.memory = []
        self.memory_strength = memory_strength
        self.budget = AUTHORITY_INITIAL_BUDGET
        self.net_returns = [] # net returns per farmer
        self.july_memory = []
    
    def predict_water(self):
        if not self.july_memory:
            return self.expected_water
        weights = np.array([self.memory_strength ** i for i in range(len(self.july_memory))])[::-1]
        self.expected_water = np.dot(weights, self.july_memory) / weights.sum()
        return self.expected_water # monthly value

    def decide_total_fields(self, farmers):
        predicted_july_inflow = self.predict_water()
        predicted_annual_inflow = predicted_july_inflow * 12

        water_limit_fields = int(predicted_annual_inflow / WATER_PER_FIELD)
        budget_limit_fields = int(self.budget // (IRRIGATION_COST + CONSUMPTION_COST))

        conservative_limit = min(water_limit_fields, budget_limit_fields, MAX_FIELDS_CENTRALIZED)

        # Current field assignment
        prev_total_fields = sum(f.irrigated_fields for f in farmers)

        # max change in fields per year is ±1 per farmer
        max_growth = len(farmers)
        max_loss = len(farmers)

        if conservative_limit > prev_total_fields:
            total_fields = min(prev_total_fields + max_growth, conservative_limit)
        elif conservative_limit < prev_total_fields:
            total_fields = max(prev_total_fields - max_loss, conservative_limit)
        else:
            total_fields = conservative_limit

        return max(total_fields, len(farmers))  # ensure minimum of 1 field per farmer

    def allocate_fields(self, farmers):
        total_fields = self.decide_total_fields(farmers)                                                                                                                
        equal_fields = total_fields // len(farmers)
        for farmer in farmers:
            farmer.irrigated_fields = min(equal_fields, MAX_FIELDS_CENTRALIZED)

        if self.budget < len(farmers) * (CONSUMPTION_COST + IRRIGATION_COST):
            print("⚠️ Centralized authority has collapsed due to insufficient budget.")
            raise RuntimeError("Centralized authority collapse.")
