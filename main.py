from Simulation.core.abm import Simulation, WaterResource
import numpy as np
from Simulation.plots import water_plot, box_plot, box_plot_cv, box_plot_dv, fish_plot, farmer_returns_plot

def create_test_inflows(case):
    if case == "1": # ideal for centralized
        return np.array([50000.0] * 200)  # stable moderate inflow
    elif case == "2": # centralized collapse
        return np.array([12000.0] * 200)  # consistently low inflow
    elif case == "3":
        return np.array([
        22000.0, 27010.0, 38441.4, 25340.4, 18380.8, 16041.0, 22600.0, 33140.8, 26109.6, 20661.2,
        14198.8, 22920.0, 17021.0, 29900.0, 31120.4, 35197.6, 26683.6, 22468.4, 18140.0,
        23420.8, 27146.0, 43128.0, 33960.0, 30960.0, 26366.0, 22210.0, 19460.0,
        28540.0, 33481.2, 26012.4, 24740.0, 20412.0, 27720.0, 38376.0, 46333.8,
        32700.0, 25260.0, 18200.0, 28600.0, 32800.0, 45500.0,
        31170.0, 20301.0, 14360.0, 34220.0,
    ]* 100)
    else:
        return np.random.uniform(5000, 40000, 200)  # default random inflow

def run_multiple_sims(memory_strength=0, centralized=False, fishing_enabled=False, return_sim = False, use_cpr_game=False, use_static_game=False, generative_agent=True, llm_provider="together"):
    results = []
    inflows = create_test_inflows("3")

    for _ in range(1):
        sim = Simulation(years=20, centralized=centralized, fishing_enabled=fishing_enabled, print_interval=1,
                        memory_strength=memory_strength, use_cpr_game=use_cpr_game, use_static_game=use_static_game, 
                        generative_agent=generative_agent, llm_provider=llm_provider)
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
    '''
    #fish plot
    deltas = np.linspace(0, 1, 11)

    central_adults, central_larvae = [], []
    dec_nf_adults, dec_nf_larvae = [], []
    dec_f_adults, dec_f_larvae, dec_f_catch = [], [], []

    for d in deltas:
        print("\n-----------------------Centralized no fishing-----------------")
        _, a, l, _, _ = run_multiple_sims(d, centralized=True, fishing_enabled=False)
        central_adults.append(a)
        central_larvae.append(l)

        print("'\n -----------------------Decentralized with fishing-----------------")
        _, a, l, c, _ = run_multiple_sims(d, centralized=False, fishing_enabled=True)
        dec_f_adults.append(a)
        dec_f_larvae.append(l)
        dec_f_catch.append(c)

        print("\n -----------------------Decentralized no fishing-----------------")
        _, a, l, _, _ = run_multiple_sims(d, centralized=False, fishing_enabled=False)
        dec_nf_adults.append(a)
        dec_nf_larvae.append(l)

    fish_plot(deltas, central_adults, central_larvae, dec_nf_adults, dec_nf_larvae, dec_f_adults, dec_f_larvae, dec_f_catch)
    
    #farmer 9 plot
    deltas = [0, 1]
    results_by_delta = {}

    for d in deltas:
        _, _, _, _, farmer9_returns = run_multiple_sims(d, centralized=False, fishing_enabled=True)
        results_by_delta[d] = farmer9_returns 
        
    farmer_returns_plot(results_by_delta)
    
    
    sim_delta0 = run_multiple_sims(memory_strength=0, centralized=True, return_sim=True)
    sim_delta1 = run_multiple_sims(memory_strength=1, centralized=True, return_sim=True)
    water_plot(sim_delta0, sim_delta1)
    '''
    
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
    results_complex_cpr = run_multiple_sims(centralized=True, use_cpr_game=True)
    results_delta0 = run_multiple_sims(memory_strength=0, centralized= True)
    results_delta1 = run_multiple_sims(memory_strength=1, centralized=True)
    
    
    results_centralized = {
    "Heuristics delta 0": results_delta0[0],
    "Heuristics delta 1": results_delta1[0],
    "Complex CPR": results_complex_cpr[0],
    }

    box_plot_cv(results_centralized)
    
