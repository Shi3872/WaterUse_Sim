import numpy as np
from Farmer import Farmer
from Authority import NationalAuthority
from Fish import FishPopulation
import os
import pickle
from plots import plot
import pandas as pd

# ---------------------------
# Parameters
# ---------------------------
MAX_FIELDS_DECENTRALIZED = 15
MAX_FIELDS_CENTRALIZED = 135
YIELD_THRESHOLD_COLLAPSE = 5
WATER_PER_FIELD = 50.0 # per month
FISH_INCOME_SCALE = 10
LARVAE_INFLOW_THRESHOLD = 2000 
AUTHORITY_INITIAL_BUDGET = 1800
CONSUMPTION_COST = 20
IRRIGATION_COST = 5

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
    def __init__(self, years=10, centralized=False, fishing_enabled=True, print_interval=1):
        self.years = years
        self.farmers = [Farmer(location=i, memory_strength=1, min_income=50) for i in range(9)]
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
        return np.array([50000.0] * 200)  # stable moderate inflow
    elif case == "2": # centralized collapse
        return np.array([12000.0] * 200)  # consistently low inflow
    elif case == "3":
        return np.array([
        18034.2, 32456.7, 36150.9, 6201.5, 25500.0, 3603.4, 17750.8, 8055.1,
        25999.9, 35770.0, 24920.0, 6601.7, 11999.8, 13015.3, 12050.7, 28450.0,
        17850.9, 32620.5, 55940.0, 5800.0, 15800.0, 4470.5, 27680.0, 7890.0,
        36120.0, 25400.1, 15010.2, 6450.0, 12010.0, 13100.0, 41980.0, 48610.9,
        28100.0, 32510.0, 56050.0, 4990.0, 15600.0, 3550.0, 27400.0, 7950.0,
        25980.0, 35550.0, 75050.0, 7550.0, 32100.0, 52200.0, 42030.0, 38520.0,
    ]* 100)
    else:
        return np.random.uniform(5000, 40000, 200)  # default random inflow

def run_and_collect_metrics(years=20, centralized=False, inflows=None, fishing_enabled=True, print_interval=1):
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

def run_multiple_sims(memory_strength=0, centralized = False):
    results = []
    inflows = create_test_inflows("")

    for _ in range(1):
        sim = Simulation(years=100, centralized=centralized, fishing_enabled=False, print_interval=1)
        for f in sim.farmers:
            f.memory_strength = memory_strength
        sim.water = WaterResource(inflows)
        sim.run()
        yields = [f.yield_history for f in sim.farmers]
        results.append(np.array(yields).T)

    avg_results = np.mean(results, axis=0)
    return avg_results
    

if __name__ == "__main__":
    inflows = create_test_inflows("1")  # change case here

    #print("\n--- Running Decentralized Simulation WITHOUT Fishing---")
    #decentralized_no_fishing = run_and_collect_metrics(years=10, centralized=False, inflows=inflows, fishing_enabled=False, print_interval=10)

    #print("\n--- Running Decentralized Simulation WITH Fishing---")
    #decentralized_with_fishing = run_and_collect_metrics(years=50, centralized=False, inflows=inflows, fishing_enabled=True, print_interval=1)

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

    #print("\n=== Comparison Results ===")
    #keys = ["avg_yield", "avg_catch", "avg_budget", "avg_fish_per_year"]

    #for key in keys:
        #d_val = decentralized_with_fishing[key]
        #c_val = centralized_results[key]
        #print(f"{key.replace('_', ' ').title()}:")
        #if key == "avg_budget":
            #print(f"  Decentralized: {d_val:.2f}")
            #print(f"  Centralized Budgets: Authority: {c_val['authority']:.2f}, Avg Farmer: {c_val['farmer']:.2f},  Combined per Farmer: {c_val['combined_per_farmer']:.2f}")
        #else:
            #print(f"  Decentralized: {d_val:.2f}")
            #print(f"  Centralized  : {c_val:.2f}")


    #heuristics_data = run_multiple_sims("Heuristics data")
    #os.makedirs("CPR/data", exist_ok=True)  # creates CPR/data/

    results_delta0 = run_multiple_sims(memory_strength=0)
    results_delta1 = run_multiple_sims(memory_strength=1.0)
    results_celta0 = run_multiple_sims(memory_strength=0, centralized=True)
    results_celta1 = run_multiple_sims(memory_strength=1.0, centralized=True)

    results = {
        "decentralized_delta0": results_delta0,
        "decentralized_delta1": results_delta1,
        "centralized_delta0": results_celta0,
        "centralized_delta1": results_celta1
    }

    plot(results)
