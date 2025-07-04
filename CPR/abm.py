import numpy as np
from Farmer import Farmer
from Authority import NationalAuthority
from Fish import FishPopulation
from solver import generate_matrix, solve_game, print_matrix

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

class WaterResource:
    def __init__(self, inflow_series):
        self.inflow_series = inflow_series
        self.index = 0 # keep track of year sim is currently on

    def next_year_inflow(self): # called once per year to get monthly inflow values
        if self.index < len(self.inflow_series): # if theres inflow values left (length of list not exceeded).
            annual_inflow = self.inflow_series[self.index]
            self.index += 1
            monthly_inflows = np.full(12, annual_inflow / 12.0) #divide annual inflow across 12 months
            return monthly_inflows
        return np.zeros(12) # return an array of 12 values


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

            if year % self.print_interval == 0:
                print(f"\nYear {year + 1} | Total Inflow: {sum(monthly_inflows):.2f}")
                lake = 0

            if self.centralized and self.authority:
                self.authority.july_memory.append(july_inflow)
                if len(self.authority.july_memory) > 10:
                    self.authority.july_memory.pop(0)
            else:
                for farmer in self.farmers:
                    farmer.possible_choices = []
                    farmer.predict_water()

                # --- CPR games ---
                remaining_water = sum(monthly_inflows)
                
                for i in range(len(self.farmers) - 1):
                    uf = self.farmers[i]
                    df = self.farmers[i + 1]

                    total_water_for_this_game = remaining_water
                    s_threshold = int(remaining_water / (12 * WATER_PER_FIELD))

                    uf_fish_income = FISH_INCOME_SCALE * (np.mean(uf.catch_history[-3:]) if len(uf.catch_history) >= 3 else (uf.catch_history[-1] if uf.catch_history else 0))
                    df_fish_income = FISH_INCOME_SCALE * (np.mean(df.catch_history[-3:]) if len(df.catch_history) >= 3 else (df.catch_history[-1] if df.catch_history else 0))

                    payoffs = generate_matrix(
                        n=12,
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
                        df_fish_income=df_fish_income
                    )

                    #print_matrix(payoffs, label_a="UF", label_b="DF")
                    equilibria = solve_game(payoffs, player_labels=("UF", "DF"))

                    if equilibria:
                        eq = np.random.choice(equilibria)

                        uf_choice = np.random.choice(range(1, 13), p=eq["UF"])
                        df_choice = np.random.choice(range(1, 13), p=eq["DF"])

                        uf.possible_choices.append(uf_choice)
                        df.possible_choices.append(df_choice)

                        uf_water_used = uf_choice * WATER_PER_FIELD * 12
                        remaining_water = max(0, remaining_water - uf_water_used) # remaining water amount

                for farmer in self.farmers:
                    if len(farmer.possible_choices) == 1:
                        farmer.irrigated_fields = farmer.possible_choices[0]
                    elif len(farmer.possible_choices) == 2:
                        farmer.irrigated_fields = np.random.choice(farmer.possible_choices)
                    else:
                        raise ValueError(f"Farmer {farmer.location} has unexpected number of choices: {len(farmer.possible_choices)}")

                #----------------------------------------------------------

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
                    monthly_runoff = max(0, water_remaining)
                    total_runoff += monthly_runoff

            for farmer in self.farmers:
                if len(farmer.monthly_water_received) >= 7:  # how much each farmer receives in july
                    farmer.july_memory.append(farmer.monthly_water_received[6])
                    if len(farmer.july_memory) > 10: # if memory too long, pop 
                        farmer.july_memory.pop(0)
                        
            lake += total_runoff * runoff_factor

            if self.centralized:
                # All farmers are the same — print only once
                loc = self.farmers[0].location
                used = annual_usage[loc]
                allocated = annual_allocated.get(loc, used)
                if year % self.print_interval == 0:
                    print(f"Per farmer: total water used = {used:.2f}, allocated = {allocated:.2f}, remaining inflow: {total_runoff:.2f}")

            may_inflow = monthly_inflows[4]
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
        return np.array([70000.0] * 200)  # stable moderate inflow
    elif case == "2": # centralized collapse
        return np.array([20000.0] * 200)  # consistently low inflow
    elif case == "3":
        return np.array([32000.0, 58000.0, 36000.0, 55000.0, 30000.0, 49000.0, 70000.0, 24000.0, 62000.0, 38000.0] * 200)
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
    inflows = create_test_inflows("3")  # change case here

    #print("\n--- Running Decentralized Simulation WITHOUT Fishing---")
    #decentralized_no_fishing = run_and_collect_metrics(years=10, centralized=False, inflows=inflows, fishing_enabled=False, print_interval=10)

    print("\n--- Running Decentralized Simulation WITH Fishing---")
    decentralized_with_fishing = run_and_collect_metrics(years=5, centralized=False, inflows=inflows, fishing_enabled=True, print_interval=1)

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