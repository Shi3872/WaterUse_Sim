from Simulation.core.abm import Simulation, WaterResource
import numpy as np
from Simulation.core.plots import water_plot, box_plot, box_plot_cv, box_plot_dv, fish_plot, farmer_returns_plot
from config_loader import load_config
from csv_exporter import SimulationCSVExporter

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

def run_multiple_sims(memory_strength=0, centralized=False, fishing_enabled=False, return_sim = False, use_cpr_game=False, use_static_game=False, generative_agent=False, llm_provider="together", years=20, inflow_case="3", save_to_csv=False, scenario_name="default"):
    results = []
    inflows = create_test_inflows(inflow_case)

    for _ in range(1):
        sim = Simulation(years=years, centralized=centralized, fishing_enabled=fishing_enabled, print_interval=1,
                        memory_strength=memory_strength, use_cpr_game=use_cpr_game, use_static_game=use_static_game, 
                        generative_agent=generative_agent, llm_provider=llm_provider)
        for f in sim.farmers:
            f.memory_strength = memory_strength
        sim.water = WaterResource(inflows)
        sim.run()

    # Export to CSV if requested
    if save_to_csv:
        exporter = SimulationCSVExporter()
        config_params = {
            'memory_strength': memory_strength,
            'centralized': centralized,
            'fishing_enabled': fishing_enabled,
            'use_cpr_game': use_cpr_game,
            'use_static_game': use_static_game,
            'generative_agent': generative_agent,
            'llm_provider': llm_provider,
            'years': years,
            'inflow_case': inflow_case
        }
        csv_files = exporter.export_all_data(sim, scenario_name, config_params)
        print(f"CSV files saved to: {exporter.session_dir}")
        for data_type, file_path in csv_files.items():
            print(f"  - {data_type}: {file_path}")

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

def run_simulation_from_config(scenario="default", config_path="config.yaml", save_to_csv=None):
    """
    Run simulation using parameters from configuration file
    
    Args:
        scenario: Which scenario to run from config.yaml
        config_path: Path to configuration file
        save_to_csv: Whether to save detailed results to CSV files (if None, uses config setting)
        
    Returns:
        Simulation results and CSV file paths (if save_to_csv=True)
    """
    # Load configuration
    config = load_config(config_path)
    params = config.get_simulation_params(scenario)
    
    # Use config setting if save_to_csv not explicitly provided
    if save_to_csv is None:
        save_to_csv = params.get('save_csv', True)
    
    print(f"Running simulation with scenario: '{scenario}'")
    print(f"Parameters: {params}")
    print(f"CSV Export: {'Enabled' if save_to_csv else 'Disabled'}")
    
    # Run simulation with config parameters
    results = run_multiple_sims(
        memory_strength=params['memory_strength'],
        centralized=params['centralized'],
        fishing_enabled=params['fishing_enabled'],
        use_cpr_game=params['use_cpr_game'],
        use_static_game=params['use_static_game'],
        generative_agent=params['generative_agent'],
        llm_provider=params['llm_provider'],
        years=params['years'],
        inflow_case=params['inflow_case'],
        save_to_csv=save_to_csv,
        scenario_name=scenario
    )
    
    return results

def run_config_based_experiments(save_to_csv=None):
    """Run experiments based on different scenarios in config"""
    config = load_config()
    scenarios = config.list_scenarios()
    
    # Use config setting if save_to_csv not explicitly provided
    if save_to_csv is None:
        output_config = config.get_output_config()
        save_to_csv = output_config.get('save_csv', True)
    
    print("Available scenarios:", scenarios)
    print(f"CSV Export: {'Enabled' if save_to_csv else 'Disabled'}")
    
    all_results = {}
    
    # Run all available scenarios
    for scenario in scenarios:
        print(f"\n=== Running {scenario} scenario ===")
        results = run_simulation_from_config(scenario, save_to_csv=save_to_csv)
        all_results[scenario] = results
        print(f"Completed {scenario} scenario")
    
    return all_results
    
if __name__ == "__main__":
    # Configuration-based simulation runs
    print("=== Water Use Simulation with Configuration Management ===")
    
    # Load configuration
    config = load_config()
    print("Available scenarios:", config.list_scenarios())
    
    # Option 1: Run a specific scenario from config
    print("\n--- Running Default Scenario ---")
    results_default = run_simulation_from_config("default")
    
    # Option 2: Run multiple scenarios for comparison
    #print("\n--- Running Multiple Scenarios ---")
    #run_config_based_experiments()
    
    # Option 3: Traditional experiments (commented out for now)
    """
    # Uncomment this section to run the original experimental setup
    
    inflows = create_test_inflows("1")  # change case here

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
    #results_delta0 = run_multiple_sims(memory_strength=0, centralized= True)
    results_delta1 = run_multiple_sims(memory_strength=1, centralized=True)
    results_complex_cpr = run_multiple_sims(centralized=True, use_cpr_game=True)
    
    
    results_centralized = {
    #"Heuristics delta 0": results_delta0[0],
    "Heuristics delta 1": results_delta1[0],
    "Complex CPR": results_complex_cpr[0],
    }

    box_plot_cv(results_centralized)
    """
