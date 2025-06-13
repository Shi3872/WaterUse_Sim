import numpy as np

# ---------------------------
# Parameters
# ---------------------------
MAX_FIELDS_DECENTRALIZED = 15
MAX_FIELDS_CENTRALIZED = 180
YIELD_THRESHOLD_COLLAPSE = 5
WATER_PER_FIELD = 50.0 # per month
FISH_INCOME_SCALE = 10
LARVAE_INFLOW_THRESHOLD = 2000 
FARMER_INITIAL_BUDGET = 200
AUTHORITY_INITIAL_BUDGET = 1800
CONSUMPTION_COST = 20
IRRIGATION_COST = 5

# ---------------------------
# Agents
# ---------------------------

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

class FishPopulation:
    def __init__(self):
        self.age_classes = [1000, 500] + [100] * 11 # age 0 to 12; 0 = larvae, 1–4 = juvenile, 5+ = adult
        self.larvae_survival_rate = 0.3
        self.adult_spawn_rate = 1.2
        self.carrying_capacity = 100000  # Max sustainable population
        self.mortality_density_independent = [0.4, 0.3, 0.2, 0.1, 0.05] + [0.1] * 8
        self.reproduction_rates = [0.0] * 5 + [0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.migration_rate_age_0 = 0.25

    def grow(self, lake_water, may_inflow): # leslie matrix equation
        new_population = [0] * 13
        adult_population = sum(self.age_classes[5:])
        juvenile_population = sum(self.age_classes[0:5])
        
        alpha = self.adult_spawn_rate # Birth rate
        sigma = 0.00001 # Density dependence strength
        gamma_juv = 0.00000000000001 # Juvenile crowding mortality

        I_t = int(may_inflow - LARVAE_INFLOW_THRESHOLD) if may_inflow > LARVAE_INFLOW_THRESHOLD else 0
        density_term = np.exp(-sigma * adult_population) # density dependent spawning
        larvae_from_adults = sum([ alpha * density_term * self.age_classes[i] for i in range(5, 13)])
        new_population[0] = int(I_t + larvae_from_adults)

        # age progression with mortality
        for age in range(12, 0, -1):  # age 12 down to age 1
            prev = self.age_classes[age - 1]
            mortality = self.mortality_density_independent[age - 1]
            survival = prev * (1 - mortality)

            if 1 <= age - 1 <= 4:
                crowding_penalty = gamma_juv * (juvenile_population ** 2)
                survival -= crowding_penalty
                survival = max(survival, 0)

            # Age 12 fish die
            new_population[age] = int(survival)

        if lake_water <= 100:
            new_population = [0 for _ in new_population]
        elif lake_water < 500:
            new_population = [int(n * 0.8) for n in new_population]
        
        self.age_classes = new_population

    def harvest(self, effort):
        adult_indices = range(5, 13)  # age classes 5 to 12
        adults = sum(self.age_classes[i] for i in adult_indices)
        
        if adults < 100 or effort <= 0:
            return 0

        harvest_rate = 0.03  # 3% of adults per unit effort
        max_catch = int(adults * harvest_rate * effort)
        catch = min(max_catch, adults)

        if catch <= 0:
            return 0

        # Proportional harvest from age classes
        weights = np.array([self.age_classes[i] for i in adult_indices])
        probabilities = weights / weights.sum()

        harvested = 0
        for _ in range(catch):
            chosen = np.random.choice(adult_indices, p=probabilities)
            if self.age_classes[chosen] > 0:
                self.age_classes[chosen] -= 1
                harvested += 1
                weights = np.array([self.age_classes[i] for i in adult_indices])
                if weights.sum() == 0:
                    break
                probabilities = weights / weights.sum()
        return harvested

class WaterResource:
    def __init__(self, inflow_series):
        self.inflow_series = inflow_series
        self.index = 0 # keep track of year sim is currently on

    def next_year_inflow(self): # called once per year to get monthly inflow values
        if self.index < len(self.inflow_series):
            annual_inflow = self.inflow_series[self.index]
            self.index += 1
            monthly_inflows = np.full(12, annual_inflow / 12.0) #divide annual inflow across 12 months
            return monthly_inflows
        return np.zeros(12) # return an array of 12 values
    
# ---------------------------


class Simulation:
    def __init__(self, years=10, centralized=False, fishing_enabled=True, print_interval=1):
        self.years = years
        self.farmers = [Farmer(location=i, memory_strength=0, min_income=50) for i in range(9)]
        for f in self.farmers:
            f.fishing_enabled = fishing_enabled
        self.print_interval = print_interval
        self.centralized = centralized
        self.authority = NationalAuthority(memory_strength=0) if centralized else None
        self.fish = FishPopulation()
        self.water = WaterResource(np.random.uniform(100, 250, years))
        self.fishing_enabled = fishing_enabled
        self.farmer_budget_history = [[] for _ in self.farmers]
        self.authority_budget_history = []
        self.annual_fish_totals = [] # total fish each year
        self.july_inflows = [] # inflow value for July (index 6)

    def run(self):
        for year in range(self.years):
            monthly_inflows = self.water.next_year_inflow()
            july_inflow = monthly_inflows[6]

            if self.centralized and self.authority and year > 0:
                self.authority.allocate_fields(self.farmers)

            for farmer in self.farmers:
                farmer.monthly_water_received = []
                farmer.july_memory.append(july_inflow)
                if len(farmer.july_memory) > 10:
                    farmer.july_memory.pop(0)

            if year % self.print_interval == 0:
                print(f"\nYear {year + 1} | Total Inflow: {sum(monthly_inflows):.2f}")
                lake = 0

            if self.centralized and self.authority:
                self.authority.july_memory.append(july_inflow)
                if len(self.authority.july_memory) > 10:
                    self.authority.july_memory.pop(0)
            else:
                for farmer in self.farmers:
                    farmer.predict_water()
                    farmer.decide_irrigation()

            annual_usage = {f.location: 0.0 for f in self.farmers}
            annual_allocated = {f.location: 0.0 for f in self.farmers}
            runoff_factor = 1

            total_runoff = 0
            for month, inflow in enumerate(monthly_inflows):
                if self.centralized:
                    total_fields = sum(f.irrigated_fields for f in self.farmers)
                    monthly_demand = total_fields * WATER_PER_FIELD

                    if monthly_demand > 0:
                        for farmer in self.farmers:
                            demand = farmer.irrigated_fields * WATER_PER_FIELD
                            alloc = min(inflow * (farmer.irrigated_fields / total_fields), demand)
                            farmer.receive_water(alloc)
                            annual_usage[farmer.location] += alloc
                            annual_allocated[farmer.location] += inflow * (farmer.irrigated_fields / total_fields)
                    else:
                        for farmer in self.farmers:
                            farmer.receive_water(0)

                    monthly_used = sum(farmer.irrigated_fields * WATER_PER_FIELD for farmer in self.farmers)
                    monthly_runoff = max(0, inflow - monthly_used)
                    total_runoff += monthly_runoff

                else:
                    water_remaining = inflow
                    for farmer in sorted(self.farmers, key=lambda f: f.location):
                        received = farmer.irrigate(water_remaining)
                        water_remaining -= received
                        annual_usage[farmer.location] += received
                    monthly_runoff = max(0, water_remaining)
                    total_runoff += monthly_runoff

            lake += total_runoff * runoff_factor

            if self.centralized:
                # All farmers are the same — print only once
                loc = self.farmers[0].location
                used = annual_usage[loc]
                allocated = annual_allocated.get(loc, used)
                if year % self.print_interval == 0:
                    print(f"Per farmer: total water used = {used:.2f}, allocated = {allocated:.2f}, remaining inflow: {total_runoff:.2f}")

            may_inflow = monthly_inflows[3]
            self.fish.grow(lake_water=lake, may_inflow=may_inflow)

            total_fish = sum(self.fish.age_classes)
            adult_fish = sum(self.fish.age_classes[5:])
            juvenile_fish = sum(self.fish.age_classes[1:5])
            larvae = self.fish.age_classes[0]

            if year % self.print_interval == 0:
                print(f"Lake Water = {lake:.2f}")
                print(f"Fish Status — Total: {total_fish}, Adults: {adult_fish}, Juveniles: {juvenile_fish}, Larvae: {larvae}")

            total_yield = 0
            total_irrigation = 0
            total_consumption = 0

            for farmer in sorted(self.farmers, key=lambda f: -f.location):
                if getattr(farmer, 'fishing_enabled', True):
                    fish_catch = self.fish.harvest(effort=2)
                else:
                    fish_catch = 0
                y, ci, cc = farmer.update_budget_and_yield(fish_catch=fish_catch, centralized=self.centralized)
                total_yield += y
                total_irrigation += ci
                total_consumption += cc

            for i, f in enumerate(self.farmers): # append budget
                    self.farmer_budget_history[i].append(f.budget)

            if self.centralized and self.authority:
                self.authority.budget += total_yield - total_irrigation - total_consumption
                self.authority.net_returns = [total_yield, total_irrigation, total_consumption]
                self.authority_budget_history.append(self.authority.budget) # append budget
                if year % self.print_interval == 0:
                    print(f"National Authority Budget = {self.authority.budget:.2f} "
                        f"(Income: {total_yield:.2f}, Irrigation Cost: {total_irrigation:.2f}, Consumption Cost: {total_consumption:.2f}) \n")

            for i, f in enumerate(self.farmers):
                if year % self.print_interval == 0:
                    print(f"Farmer {i+1}: Fields={f.irrigated_fields}, Budget={f.budget:.2f}, "
                        f"Last Yield={f.yield_history[-1]:.2f}, Catch={int(f.catch_history[-1])}")
            
            self.annual_fish_totals.append(sum(self.fish.age_classes))
            self.july_inflows.append(self.water.inflow_series[year] / 12.0)  # July inflow assumed uniform

def create_test_inflows(case):
    if case == "1": # ideal for centralized
        return np.array([60000.0] * 200)  # stable moderate inflow
    elif case == "2": # centralized collapse
        return np.array([20000.0] * 200)  # consistently low inflow
    else:
        return np.random.uniform(20000, 60000, 200)  # default random inflow

def run_and_collect_metrics(years=100, centralized=False, inflows=None, fishing_enabled=True, print_interval=1):
    if inflows is None:
        inflows = np.random.uniform(1000, 3000, years)
    sim = Simulation(years=years, centralized=centralized, fishing_enabled=fishing_enabled, print_interval=print_interval)
    sim.water = WaterResource(inflows)
    
    sim.run()
    avg_yield = np.mean([np.mean(f.yield_history) for f in sim.farmers])
    avg_catch = np.mean([np.mean(f.catch_history) for f in sim.farmers])
    avg_fish_per_year = np.mean(sim.annual_fish_totals) if sim.annual_fish_totals else 0

    if centralized:
        avg_farmer_budget = np.mean([np.mean(b) for b in sim.farmer_budget_history]) if sim.farmer_budget_history else 0
        avg_authority_budget = np.mean(sim.authority_budget_history) if sim.authority_budget_history else 0
        avg_budget = {
            "farmer": avg_farmer_budget,
            "authority": avg_authority_budget,
            "combined_per_farmer": (avg_authority_budget / len(sim.farmers) if sim.farmers else 0) + avg_farmer_budget
        }
    else:
        avg_budget = np.mean([np.mean(budget_list) for budget_list in sim.farmer_budget_history]) if sim.farmer_budget_history else 0

    return {
        "mode": "centralized" if centralized else "decentralized",
        "avg_yield": avg_yield,
        "avg_catch": avg_catch,
        "avg_budget": avg_budget,
        "avg_fish_per_year": avg_fish_per_year,
    }

if __name__ == "__main__":
    inflows = create_test_inflows("1")  # change case here

    #print("\n--- Running Decentralized Simulation WITHOUT Fishing---")
    #decentralized_no_fishing = run_and_collect_metrics(years=10, centralized=False, inflows=inflows, fishing_enabled=False, print_interval=10)

    print("\n--- Running Decentralized Simulation WITH Fishing---")
    decentralized_with_fishing = run_and_collect_metrics(years=100, centralized=False, inflows=inflows, fishing_enabled=True, print_interval=10)

    #print("\n=== Fishing Impact on Decentralized Performance ===")
    #keys = ["avg_yield", "avg_catch", "avg_budget", "avg_fish_per_year"]
    #for key in keys:
        #no_fish = decentralized_no_fishing[key]
        #with_fish = decentralized_with_fishing[key]
        #print(f"{key.replace('_', ' ').title()}:")
        #print(f"  No Fishing : {no_fish:.2f}")
        #print(f"  With Fishing: {with_fish:.2f}\n")

    #print("\n--- Running Centralized Simulation---")
    #centralized_results = run_and_collect_metrics(years=10, centralized=True, inflows=inflows, print_interval=1)

    print("\n=== Comparison Results ===")
    keys = ["avg_yield", "avg_catch", "avg_budget", "avg_fish_per_year"]

    for key in keys:
        d_val = decentralized_with_fishing[key]
        #c_val = centralized_results[key]
        print(f"{key.replace('_', ' ').title()}:")
        if key == "avg_budget":
            print(f"  Decentralized: {d_val:.2f}")
            #print(f"  Centralized Budgets: Authority: {c_val['authority']:.2f}, Avg Farmer: {c_val['farmer']:.2f},  Combined per Farmer: {c_val['combined_per_farmer']:.2f}")
        else:
            print(f"  Decentralized: {d_val:.2f}")
            #print(f"  Centralized  : {c_val:.2f}")