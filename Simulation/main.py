from abm import Simulation, WaterResource
import numpy as np
from plots import water_plot, box_plot, fish_plot, farmer_returns_plot

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

def run_multiple_sims(memory_strength=0, centralized = False, fishing_enabled=False, return_sim = False):
    results = []
    inflows = create_test_inflows("3")

    for _ in range(1):
        sim = Simulation(years=50, centralized=centralized, fishing_enabled=fishing_enabled, print_interval=1, memory_strength=memory_strength)
        for f in sim.farmers:
            f.memory_strength = memory_strength
        sim.water = WaterResource(inflows)
        sim.run()

    if return_sim:
        return sim
    
    # Crop yields
    yields = [f.yield_history for f in sim.farmers]
    results.append(np.array(yields).T)
    avg_results = np.mean(results, axis=0)

    # Fish
    adult_fish = [sum(ac[5:]) for ac in sim.fish_history] # adult classes 5+
    larvae_inflow = [ac[0] for ac in sim.fish_history] # larvae (class 0)
    avg_adults = np.mean(adult_fish)
    avg_larvae = np.mean(larvae_inflow)

    total_catch = [sum(f.catch_history) for f in sim.farmers]
    mean_catch = np.mean(total_catch) if fishing_enabled else 0

    # Farmer 9 returns
    farmer9_returns = sim.farmer_budget_history[8]

    return avg_results, avg_adults, avg_larvae, mean_catch, farmer9_returns
    
if __name__ == "__main__":
    inflows = create_test_inflows("1")  # change case here

    # water plot
    #sim_delta0 = run_multiple_sims(memory_strength=0, centralized=True, return_sim=True)
    #sim_delta1 = run_multiple_sims(memory_strength=1, centralized=True, return_sim=True)
    #water_plot(sim_delta0, sim_delta1)
    
    #fish plot
    '''deltas = np.linspace(0, 1, 11)

    central_adults, central_larvae = [], []
    dec_nf_adults, dec_nf_larvae = [], []
    dec_f_adults, dec_f_larvae, dec_f_catch = [], [], []

    for d in deltas:
        _, a, l, _ = run_multiple_sims(d, centralized=True, fishing_enabled=False)
        central_adults.append(a)
        central_larvae.append(l)

        _, a, l, _ = run_multiple_sims(d, centralized=False, fishing_enabled=False)
        dec_nf_adults.append(a)
        dec_nf_larvae.append(l)

        _, a, l, c = run_multiple_sims(d, centralized=False, fishing_enabled=True)
        dec_f_adults.append(a)
        dec_f_larvae.append(l)
        dec_f_catch.append(c)'''
    

    #fish_plot(deltas, central_adults, central_larvae, dec_nf_adults, dec_nf_larvae, dec_f_adults, dec_f_larvae, dec_f_catch)

    #farmer 9 plot
    deltas = [0, 1]
    results_by_delta = {}

    for d in deltas:
        _, _, _, _, farmer9_returns = run_multiple_sims(d, centralized=False, fishing_enabled=True)
        results_by_delta[d] = farmer9_returns 
        
    farmer_returns_plot(results_by_delta)


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
