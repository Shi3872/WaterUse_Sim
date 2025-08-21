from abm import Simulation, WaterResource
import numpy as np
from plots import water_plot, box_plot, box_plot_cv, box_plot_dv, fish_plot, farmer_returns_plot

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

def run_multiple_sims(memory_strength=0, centralized=False, fishing_enabled=False, return_sim = False, use_cpr_game=False, use_static_game=False):
    results = []
    inflows = create_test_inflows("3")

    for _ in range(1):
        sim = Simulation(years=100, centralized=centralized, fishing_enabled=fishing_enabled, print_interval=1,
                        memory_strength=memory_strength, use_cpr_game=use_cpr_game, use_static_game=use_static_game)
        for f in sim.farmers:
            f.memory_strength = memory_strength
        sim.water = WaterResource(inflows)
        sim.run()

    if return_sim:
        return sim
    
    # Crop yields
    yields = [f.yield_history for f in sim.farmers]
    results.append(np.array(yields).T)
    avg_yield = np.mean(results, axis=0)

    # Fish
    adult_fish = [sum(ac[5:]) for ac in sim.fish_history] # adult classes 5+
    larvae_inflow = [ac[0] for ac in sim.fish_history] # l 
    avg_adults = np.mean(adult_fish)
    avg_larvae = np.mean(larvae_inflow)

    total_catch = [sum(f.catch_history) for f in sim.farmers]
    mean_catch = np.mean(total_catch) if fishing_enabled else 0

    # Farmer 9 returns
    farmer9_returns = sim.farmer_budget_history[8]

    return avg_yield, avg_adults, avg_larvae, mean_catch, farmer9_returns
    
if __name__ == "__main__":
    inflows = create_test_inflows("1")  # change case here

    # box plot
    results_delta0 = run_multiple_sims(memory_strength=0, centralized= False)
    results_delta1 = run_multiple_sims(memory_strength=1)
    results_static_cpr = run_multiple_sims(use_cpr_game=True, use_static_game=True)
    results_complex_cpr = run_multiple_sims(use_cpr_game=True, use_static_game=False)


    results_decentralized = {
    "Heuristics delta 0": results_delta0[0],
    "Static CPR": results_static_cpr[0],
    "Heuristics delta 1": results_delta1[0],
    "Complex CPR": results_complex_cpr[0],
    }

    box_plot_dv(results_decentralized)
    
    # box plot
    results_delta0 = run_multiple_sims(memory_strength=0, centralized= True)
    results_delta1 = run_multiple_sims(memory_strength=1, centralized=True)
    results_complex_cpr = run_multiple_sims(centralized=True, use_cpr_game=True)


    results_centralized = {
    "Heuristics delta 0": results_delta0[0],
    "Heuristics delta 1": results_delta1[0],
    "Complex CPR": results_complex_cpr[0],
    }

    box_plot_cv(results_centralized)
