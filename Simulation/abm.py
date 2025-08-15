import numpy as np
from Farmer import Farmer
from Authority import NationalAuthority
from Fish import FishPopulation
import os
import pickle
from plots import water_plot, box_plot
import pandas as pd

# ---------------------------
# Parameters
# ---------------------------
MAX_FIELDS_DECENTRALIZED = 10
MAX_FIELDS_CENTRALIZED = 90
WATER_PER_FIELD = 50.0 # per month
FISH_INCOME_SCALE = 10
LARVAE_INFLOW_THRESHOLD = 2000 
AUTHORITY_INITIAL_BUDGET = 1800
CONSUMPTION_COST = 20 # annual
IRRIGATION_COST =6

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
    def __init__(self, years=10, centralized=False, fishing_enabled=True, print_interval=1, memory_strength=0):
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

    def run(self):
        for year in range(self.years):
            monthly_inflows = self.water.next_year_inflow()
            july_inflow = monthly_inflows[6]

           # predict water before this year's inflow is known 
            if self.centralized and self.authority:
                predicted = self.authority.predict_water()
                self.predicted_water_history.append(predicted)

                self.authority.allocate_fields(self.farmers)

            for farmer in self.farmers:
                farmer.monthly_water_received = []

            if year % self.print_interval == 0:
                print(f"\nYear {year + 1} | Total Inflow: {sum(monthly_inflows):.2f}")
                lake = 0

            if not self.centralized:
                for farmer in self.farmers:
                    farmer.predict_water()
                    farmer.decide_irrigation()

            annual_usage = {f.location: 0.0 for f in self.farmers}
            annual_allocated = {f.location: 0.0 for f in self.farmers}
            runoff_factor = 1
            total_runoff = 0

            # allcoate water month by month
            for month, inflow in enumerate(monthly_inflows):
                if self.centralized:
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

                else:
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
                if len(self.authority.july_memory) > 10:
                    self.authority.july_memory.pop(0)

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
        38441.4, 29340.4, 19380.8, 29041.0, 36600.0, 33140.8, 26109.6, 21661.2,
        31198.8, 42920.0, 29900.0, 27120.4, 31197.6, 39683.6, 26468.4, 34140.0,
        33420.8, 27146.0, 43128.0, 30960.0, 30960.0, 29366.0, 33210.0, 45460.0,
        38540.0, 30481.2, 18012.4, 31740.0, 26412.0, 27720.0, 38376.0, 46333.8,
        33700.0, 39010.0, 31260.0, 29880.0, 18200.0, 28600.0, 32800.0, 45500.0,
        31170.0, 42660.0, 30060.0, 33060.0, 38520.0, 38640.0, 14360.0, 34220.0,
    ]* 100)
    else:
        return np.random.uniform(5000, 40000, 200)  # default random inflow

def run_multiple_sims(memory_strength=0, centralized = False, return_sim = False):
    results = []
    inflows = create_test_inflows("3")

    for _ in range(1):
        sim = Simulation(years=100, centralized=centralized, fishing_enabled=False, print_interval=1, memory_strength=memory_strength)
        for f in sim.farmers:
            f.memory_strength = memory_strength
        sim.water = WaterResource(inflows)
        sim.run()
        yields = [f.yield_history for f in sim.farmers]
        results.append(np.array(yields).T)

    if return_sim:
        return sim
    
    avg_results = np.mean(results, axis=0)
    return avg_results
    

if __name__ == "__main__":
    inflows = create_test_inflows("1")  # change case here

    # water plot
    #sim_delta0 = run_multiple_sims(memory_strength=0, centralized=True, return_sim=True)
    #sim_delta1 = run_multiple_sims(memory_strength=1, centralized=True, return_sim=True)
    #water_plot(sim_delta0, sim_delta1)
    
    # box plot
    results_delta0 = run_multiple_sims(memory_strength=0)
    results_delta1 = run_multiple_sims(memory_strength=1.0)
    results_celta0 = run_multiple_sims(memory_strength=0, centralized=True)
    results_celta1 = run_multiple_sims(memory_strength=1.0, centralized=True)

    results = {
        "Decentralized Delta0": results_delta0,
        "Decentralized Delta1": results_delta1,
        "Centralized Delta0": results_celta0,
        "Centralized Delta1": results_celta1
    }

    box_plot(results)
