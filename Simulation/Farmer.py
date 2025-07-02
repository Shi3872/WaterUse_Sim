class Farmer:
    def __init__(self, location, memory_strength, min_income):
        self.location = location
        self.irrigated_fields = 3
        self.expected_water = 40000
        self.memory = []
        self.memory_strength = memory_strength
        self.yield_history = []
        self.catch_history = []
        self.budget = FARMER_INITIAL_BUDGET
        self.min_income = min_income
        self.planned_fields = self.irrigated_fields
        self.monthly_water_received = []
        self.july_memory = [] 
        self.collapsed = False

    def receive_water(self, amount):
            self.monthly_water_received.append(amount)

    def predict_water(self): # water prediction equation (eq1)
        if not self.july_memory:
            return self.expected_water
        weights = np.array([self.memory_strength ** i for i in range(len(self.july_memory))])[::-1]
        self.expected_water = np.dot(weights, self.july_memory) / weights.sum()
        return self.expected_water # monthly value

    def decide_irrigation(self):
        if getattr(self, "collapsed", False):
            return
        if not self.yield_history:
            return
        last_yield = self.yield_history[-1]
        last_satisfaction = self.memory[-1] if self.memory else 1.0

        # if in debt and have no income (from yields)
        if self.budget < -100 and self.yield_history and self.yield_history[-1] == 0:
            self.collapsed = True
            self.irrigated_fields = 0 
            return

        # decide number of fields based on income and demand
        if last_yield == 0:
            self.irrigated_fields = int(max(1, self.irrigated_fields - 1))
        elif last_yield < YIELD_THRESHOLD_COLLAPSE: # low income = take risk
            self.irrigated_fields = int(min(self.irrigated_fields + 1, MAX_FIELDS_DECENTRALIZED))
        elif last_satisfaction < 0.8: # income ok, but demand unmet = be cautious
            expected = self.predict_water() * 12
            max_fields = int(expected / WATER_PER_FIELD)
            self.irrigated_fields = int(min(max_fields, self.irrigated_fields))
        else: # demand and income met = +1 if can
            if self.irrigated_fields < MAX_FIELDS_DECENTRALIZED:
                self.irrigated_fields += 1

    def irrigate(self, available_water): # return received water per month
        if self.collapsed:
            return 0
        
        self.planned_fields = self.irrigated_fields  # store for stress calc
        demand = self.irrigated_fields * (WATER_PER_FIELD)
        received = min(available_water, demand)
        self.monthly_water_received.append(received)
        return received

    def calculate_crop_stress(self): # irrigation equation (eq2)
        if self.collapsed:
            return 0
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
        if self.collapsed:
            self.yield_history.append(0)
            self.catch_history.append(0)
            return 0, 0, 0
        field_yield = self.calculate_crop_stress()
        yield_factor = field_yield / self.irrigated_fields if self.irrigated_fields > 0 else 0

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
        self.yield_history.append(yield_factor)
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
