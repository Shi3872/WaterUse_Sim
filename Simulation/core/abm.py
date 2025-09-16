import numpy as np
from Simulation.agents.Farmer import Farmer
from Simulation.agents.Authority import NationalAuthority
from Simulation.agents.Fish import FishPopulation
from Simulation.solver import generate_dv_matrix, generate_cv_matrix, solve_game
import math

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

# ---------------------------

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

class Simulation:
    def __init__(self, years=10, centralized=False, fishing_enabled=True, print_interval=1, memory_strength=0, use_cpr_game = False, use_static_game = False, generative_agent=False, llm_provider="together"):
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
                per_farmer_choice = int(np.random.choice(list(range(1, 11)), p=eq["Authority"])) # pick per farmer count (1-10)

                for f in self.farmers: # equal allocation
                    f.irrigated_fields = per_farmer_choice

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
                payoffs = [
                    [(6, 6), (5, 7)],
                    [(9, 3), (5, 2)]
                ]
                matrix_size = 2
                strategy_values = [6, 10]
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
                    df_fish_income=df_fish_income
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
            elif len(farmer.possible_choices) == 2:
                farmer.irrigated_fields = np.random.choice(farmer.possible_choices)
            else:
                raise ValueError(f"Farmer {farmer.location} has unexpected number of choices: {len(farmer.possible_choices)}")

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
                if self.use_cpr_game: # Game logic
                    self.decentralized_game(monthly_inflows)
                else: # Heuristic logic
                    for farmer in self.farmers:
                        farmer.predict_water()
                        if not self.generative_agent:
                            farmer.decide_irrigation()
                        else:
                            if np.random.random() < 0.5: # 10% chance to use generative agent
                                farmer.decide_irrigation_generative_agent(provider=self.llm_provider)
                            else:
                                farmer.decide_irrigation()

            if year % self.print_interval == 0:
                if self.use_cpr_game == False and self.use_static_game == False and self.centralized == True:
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
                    if self.use_cpr_game == False and self.use_static_game == False and self.centralized == True:
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
                if self.use_cpr_game == False and self.use_static_game == False and self.centralized == True:
                    print(f"Fish Status — Total: {total_fish}, Adults: {adult_fish}, Juveniles: {juvenile_fish}, Larvae: {larvae}")

            total_yield = 0
            total_irrigation = 0
            total_consumption = 0

            for farmer in sorted(self.farmers, key=lambda f: -f.location):
                if getattr(farmer, 'fishing_enabled', True):
                    fish_catch = self.fish.harvest(target_catch=20)
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
                    if self.use_cpr_game == False and self.use_static_game == False and self.centralized == True:
                        print(f"National Authority Budget = {self.authority.budget:.2f} "
                            f"(Income: {total_yield:.2f}, Irrigation Cost: {total_irrigation:.2f}, Consumption Cost: {total_consumption:.2f}) \n")

            for i, f in enumerate(self.farmers):
                if year % self.print_interval == 0:
                    if self.use_cpr_game == False and self.use_static_game == False and self.centralized == True:
                        print(f"Farmer {i+1}: Fields={f.irrigated_fields}, Budget={f.budget:.2f}, "
                            f"Last Yield={f.yield_history[-1]:.2f}, Catch={int(f.catch_history[-1])}")
            
            self.annual_fish_totals.append(sum(self.fish.age_classes))
            self.july_inflows.append(self.water.inflow_series[year] / 12.0)  # July inflow assumed uniform
            self.fish_history.append(list(self.fish.age_classes))


