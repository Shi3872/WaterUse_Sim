from Simulation.core.abm import Simulation, WaterResource
import numpy as np
import os
from Simulation.core.plots import water_plot, box_plot, box_plot_cv, box_plot_dv, fish_plot, farmer_returns_plot
from config_loader import load_config
from csv_exporter import SimulationCSVExporter
from csv_plots import CSVPlotter, plot_from_latest_results

def validate_config_parameters(params):
    """
    Validate configuration parameters according to specific conjunction rules.
    
    Args:
        params: Dictionary of simulation parameters
        
    Raises:
        ValueError: If configuration rules are violated
    """
    # Extract boolean flags from flat parameters (config loader returns flat structure)
    centralized = params.get('centralized', False)
    generative_agent = params.get('generative_agent', False)
    use_cpr_game = params.get('use_cpr_game', False)
    fishing_cpr = params.get('fishing_cpr', False)
    use_static_game = params.get('use_static_game', False)
    
    # Apply conjunction rules using boolean operators
    rule1_valid = not centralized or (not generative_agent and not use_cpr_game and not fishing_cpr and not use_static_game)
    rule2_valid = not generative_agent or (not centralized and not use_cpr_game and not fishing_cpr and not use_static_game)
    rule3_valid = not use_cpr_game or (not centralized and not generative_agent and not use_static_game and fishing_cpr)
    rule4_valid = not fishing_cpr or (not centralized and not generative_agent and not use_static_game and use_cpr_game)
    rule5_valid = not use_static_game or (not centralized and not generative_agent)
    
    # Check if all rules are satisfied
    if not (rule1_valid and rule2_valid and rule3_valid and rule4_valid and rule5_valid):
        failed_rules = []
        if not rule1_valid: failed_rules.append("rule1 (centralized conflicts)")
        if not rule2_valid: failed_rules.append("rule2 (generative_agent conflicts)")
        if not rule3_valid: failed_rules.append("rule3 (use_cpr_game conflicts)")
        if not rule4_valid: failed_rules.append("rule4 (fishing_cpr conflicts)")
        if not rule5_valid: failed_rules.append("rule5 (use_static_game conflicts)")
        
        raise ValueError(f"Configuration validation failed: Parameter combination violates conjunction rules. Failed: {', '.join(failed_rules)}")
        
    return True


def initialize_game_matrices(use_static_game=False, force_regenerate=False):
    """
    Initialize game matrices using DeepSeek-V3 if use_static_game is True.
    Store results in a text file for later use.
    
    Args:
        use_static_game (bool): Whether to generate game matrices
        force_regenerate (bool): Force regeneration even if file exists
        
    Returns:
        bool: True if matrices were generated/exist, False otherwise
    """
    matrices_file = "game_matrices.txt"
    
    if not use_static_game:
        print("Static game mode disabled, skipping matrix generation.")
        return False
    
    # Check if file already exists and is valid
    if os.path.exists(matrices_file) and not force_regenerate:
        try:
            with open(matrices_file, 'r') as f:
                content = f.read()
                if content.strip() and "VALIDATED_MATRICES" in content:
                    print(f"✓ Using existing game matrices from {matrices_file}")
                    return True
        except Exception as e:
            print(f"Error reading existing matrices file: {e}")
    
    print("Generating strategic game matrices using DeepSeek-V3...")
    
    try:
        # Import DeepSeek-V3 function
        import sys
        import importlib.util
        import json
        from datetime import datetime
        
        llm_path = os.path.join(os.path.dirname(__file__), 'LLM_Prompting', 'Together', 'DeepSeek-V3.py')
        spec = importlib.util.spec_from_file_location("deepseek_v3", llm_path)
        deepseek_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deepseek_module)
        
        # Generate all game scenarios including upstream-downstream (automatically saves to file)
        print("Generating action situations including upstream-downstream...")
        game_result = deepseek_module.generate_game_scenarios_and_payoffs(save_to_file=True, output_file=matrices_file)
        
        # Check if matrices were generated successfully
        if game_result and os.path.exists(matrices_file):
            print(f"✓ Generated action situations saved to {matrices_file}")
            return True
        else:
            print("✗ Failed to generate or save game matrices")
            return False
            
    except Exception as e:
        print(f"✗ Error generating game matrices: {e}")
        raise RuntimeError(f"Failed to generate game matrices: {e}")


real_inflows = np.array([
        22000.0, 27010.0, 38441.4, 25340.4, 18380.8, 16041.0, 22600.0, 33140.8, 26109.6, 20661.2,
        14198.8, 22920.0, 17021.0, 29900.0, 31120.4, 35197.6, 26683.6, 22468.4, 18140.0,
        23420.8, 27146.0, 43128.0, 33960.0, 30960.0, 26366.0, 22210.0, 19460.0,
        28540.0, 33481.2, 26012.4, 24740.0, 20412.0, 27720.0, 38376.0, 46333.8,
        32700.0, 25260.0, 18200.0, 28600.0, 32800.0, 45500.0,
        31170.0, 20301.0, 14360.0, 34220.0,
    ])
def create_test_inflows(case):
    if case == "1": # ideal for centralized
        return np.array([50000.0] * 200)  # stable moderate inflow
    elif case == "2": # centralized collapse
        return np.array([12000.0] * 200)  # consistently low inflow
    elif case == "3":
        return np.array(list(real_inflows) * 100)
    elif case == "4": 
        # random but with decreasing trend of inflow
        np.random.seed(42)  # For reproducible results
        years = 100
        
        # Interpolate real_inflows from 45 to 100 points
        original_indices = np.linspace(0, 1, len(real_inflows))
        new_indices = np.linspace(0, 1, years)
        interpolated_inflows = np.interp(new_indices, original_indices, real_inflows)
        
        return interpolated_inflows
        
    else:
        return np.random.uniform(16000, 46000, 100)  # default random inflow

def run_multiple_sims(memory_strength=0, centralized=False, fishing_enabled=False, return_sim = False, use_cpr_game=False, use_static_game=False, generative_agent=False, \
                      llm_provider="together", years=20, inflow_case="3", use_fishing_cpr=False, save_to_csv=False, scenario_name="default", number_of_runs=1, simulate_tragedy=True):
    """
    Run multiple simulation instances and collect results
    
    Args:
        number_of_runs: Number of simulation runs to execute (default: 1)
        Other parameters: Standard simulation parameters
        
    Returns:
        Results including aggregated data across all runs
    """
    all_simulations = []
    all_farmer_budgets = []  # Store budget histories across all runs
    

    for run_idx in range(number_of_runs):
        print(f"Running simulation {run_idx + 1}/{number_of_runs}")
        inflows = create_test_inflows(inflow_case)
        sim = Simulation(years=years, centralized=centralized, fishing_enabled=fishing_enabled, print_interval=1,
                        memory_strength=memory_strength, use_cpr_game=use_cpr_game, use_static_game=use_static_game, 
                        generative_agent=generative_agent, llm_provider=llm_provider, use_fishing_cpr=use_fishing_cpr, simulate_tragedy=simulate_tragedy)
        for f in sim.farmers:
            f.memory_strength = memory_strength
        sim.water = WaterResource(inflows, inflow_case)
        sim.run()
        
        all_simulations.append(sim)
        # Store farmer budget histories for this run
        run_budgets = sim.farmer_budget_history  # Use simulation's farmer budget history
        all_farmer_budgets.append(run_budgets)

    # Use the last simulation for most results (backward compatibility)
    last_sim = all_simulations[-1]
    
    yields = [f.yield_history for f in last_sim.farmers]
    results = [np.array(yields).T]
    avg_yield = np.mean(results, axis=0)

    # Fish data from last simulation
    adult_fish = [sum(ac[5:]) for ac in last_sim.fish_history]
    larvae_inflow = [ac[0] for ac in last_sim.fish_history]
    avg_adults = np.mean(adult_fish)
    avg_larvae = np.mean(larvae_inflow)

    total_catch = [sum(f.catch_history) for f in last_sim.farmers]
    mean_catch = np.mean(total_catch) if fishing_enabled else 0

    # Farmer 9 returns from last simulation
    farmer9_returns = last_sim.farmer_budget_history[8]
    
    # Export to CSV if requested - include multiple runs data
    session_dir = None
    if save_to_csv:
        exporter = SimulationCSVExporter()
        config_params = {
            'memory_strength': memory_strength,
            'centralized': centralized,
            'fishing_enabled': fishing_enabled,
            'use_cpr_game': use_cpr_game,
            'use_static_game': use_static_game,
            'fishing_cpr': use_fishing_cpr,
            'generative_agent': generative_agent,
            'llm_provider': llm_provider,
            'years': years,
            'inflow_case': inflow_case,
            'number_of_runs': number_of_runs
        }
        csv_files = exporter.export_multiple_runs_data(all_simulations, all_farmer_budgets, scenario_name, config_params)
        session_dir = exporter.session_dir
        print(f"CSV files saved to: {session_dir}")
        for data_type, file_path in csv_files.items():
            print(f"  - {data_type}: {file_path}")

    if return_sim:
        if save_to_csv:
            return last_sim, session_dir
        else:
            return last_sim
    
    # Return results with session directory if CSV was saved
    if save_to_csv:
        return avg_yield, avg_adults, avg_larvae, mean_catch, farmer9_returns, session_dir
    else:
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
    
    print(f"Loading scenario '{scenario}' with parameters:")
    print(f"  - centralized: {params.get('centralized', 'NOT SET')}")
    print(f"  - use_static_game: {params.get('use_static_game', 'NOT SET')}")
    print(f"  - use_cpr_game: {params.get('use_cpr_game', 'NOT SET')}")
    print(f"  - generative_agent: {params.get('generative_agent', 'NOT SET')}")
    print(f"  - fishing_cpr: {params.get('fishing_cpr', 'NOT SET')}")
    
    # Validate configuration parameters for conflicts
    validate_config_parameters(params)
    
    # Initialize game matrices if using static game mode
    if params.get('use_static_game', False):
        print(f"Scenario '{scenario}' requires static game matrices - initializing...")
        matrices_ready = initialize_game_matrices(use_static_game=True)
        if not matrices_ready:
            print("Warning: Failed to generate game matrices, simulation may use defaults")
    else:
        print(f"Scenario '{scenario}' does not use static game matrices")
    
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
        use_fishing_cpr=params['fishing_cpr'],
        save_to_csv=save_to_csv,
        scenario_name=scenario,
        number_of_runs=params.get('number_of_runs', 1),  # Default to 1 if not specified
        simulate_tragedy=params['simulate_tragedy']
    )
    
    # If CSV was saved, results will include the session directory as the last element
    if save_to_csv and len(results) == 6:
        # Extract session directory from results
        session_dir = results[-1]
        simulation_results = results[:-1]
        return simulation_results, session_dir
    else:
        return results, None

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

def plot_latest_results(show_dashboard=True, results_dir=None):
    """
    Generate plots from simulation results
    All plots are automatically saved to the results directory
    
    Args:
        show_dashboard: Whether to show the comprehensive dashboard (default: True)
                       If False, generates individual plots instead
        results_dir: Specific results directory to use. If None, uses latest directory
    """
    plotter = CSVPlotter()
    
    # Use provided results directory or find the latest one
    if results_dir is not None:
        latest_dir = results_dir
    else:
        latest_dir = plotter.get_latest_results_dir()
    
    if latest_dir is None:
        print("No results directory found! Run a simulation first.")
        return
    
    scenario_name = os.path.basename(latest_dir).split('_')[0]
    print(f"Generating plots from: {latest_dir}")
    print(f"Scenario: {scenario_name}")
    
    if show_dashboard:
        plotter.comprehensive_dashboard(latest_dir)
    else:
        # Generate individual plots (all auto-saved to results directory)
        plotter.farmer_returns_plot(latest_dir)
        plotter.water_plot(latest_dir)
        plotter.fish_plot(latest_dir)
        plotter.box_plot_yields(latest_dir)

def run_simulation_and_plot(scenario="default", save_csv=None, save_plots=True):
    """
    Convenience function to run simulation and immediately generate plots
    All plots are automatically saved to the results directory
    
    Args:
        scenario: Which scenario to run
        save_csv: Whether to save CSV data (uses config default if None)
        show_plots: Whether to generate plots after simulation (default: True)
    """
    print(f"=== Running Simulation and Plotting: {scenario} ===")
    
    # Run the simulation
    results, session_dir = run_simulation_from_config(scenario, save_to_csv=save_csv)
    
    if save_plots and save_csv != False:  # Only plot if we have CSV data
        print(f"\n=== Generating Plots ===")
        if session_dir:
            # Use the specific session directory created by the CSV exporter
            plot_latest_results(show_dashboard=True, results_dir=session_dir)
        else:
            # Fallback to finding latest directory
            plot_latest_results(show_dashboard=True)
    
    return results
    
if __name__ == "__main__":
    # Configuration-based simulation runs with CSV export and plotting
    print("=== Water Use Simulation with CSV Export and Plotting ===")
    
    # Load configuration
    config = load_config()
    print("Available scenarios:", config.list_scenarios())
    
    # Validate configuration for the default scenario
    default_params = config.get_simulation_params("default")
    print(f"Default scenario parameters: {default_params}")
    validate_config_parameters(default_params)
    
    # Note: Matrix initialization will be handled by run_simulation_from_config() 
    # when the scenario actually runs, ensuring proper parameter override handling
    
    # Option 1: Run simulation with automatic plotting
    print("\n--- Running Simulation with Plots ---")
    results = run_simulation_and_plot("default", save_plots=True)
    
    # Option 2: Generate plots from existing results
    #print("\n--- Generating Plots from Latest Results ---")
    #plot_latest_results(show_dashboard=True)
    
    # Option 3: Run specific scenario and plot
    #print("\n--- Running Centralized Fishing Scenario ---")
    #results_fishing = run_simulation_and_plot("centralized_fishing", save_plots=True)
    
    # Option 4: Run multiple scenarios for comparison
    #print("\n--- Running Multiple Scenarios ---")
    #run_config_based_experiments()
    
    print("\n--- Simulation and Plotting Complete ---")
    print("Check the current directory for saved plot files:")
    print("- dashboard_default.png")
    print("- farmer_returns_default.png") 
    print("- water_data_default.png")
    print("- fish_population_default.png")
    print("- yield_boxplot_default.png")
    