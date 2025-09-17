# CSV-Based Plotting for Water Use Simulation

This document explains how to generate plots from CSV simulation results, replacing the old implementation's direct plotting approach.

## Overview

The simulation now exports detailed CSV data and provides comprehensive plotting functionality to visualize results. This approach allows for:
- Data persistence and reproducibility
- Flexible post-processing and analysis
- Consistent plot generation across different runs

## Quick Start

### 1. Run Simulation with Automatic Plotting
```python
from main import run_simulation_and_plot

# Run simulation and generate plots automatically
results = run_simulation_and_plot("default", save_plots=True)
```

### 2. Generate Plots from Existing Results
```python
from main import plot_latest_results

# Plot from the most recent simulation
plot_latest_results(show_dashboard=True, save_plots=True)
```

### 3. Plot Specific Components
```python
from csv_plots import CSVPlotter

plotter = CSVPlotter()
latest_dir = plotter.get_latest_results_dir()

# Generate individual plots
plotter.farmer_returns_plot(latest_dir, farmer_id=9)
plotter.water_plot(latest_dir) 
plotter.fish_plot(latest_dir)
plotter.box_plot_yields(latest_dir)
```

## Available Plot Types

### 1. Farmer Returns Plot
- Shows budget evolution over time for a specific farmer
- Equivalent to original `farmer_returns_plot()`
- Default: Farmer 9 (can be customized)

### 2. Water Resource Plot  
- Displays annual and July inflow patterns
- Shows water availability trends over simulation period

### 3. Fish Population Plot
- Visualizes fish dynamics (adults, larvae, total population)
- Equivalent to original `fish_plot()` functionality

### 4. Yield Box Plot
- Distribution of crop yields across all farmers
- Equivalent to original `box_plot()` functions
- Shows variability and outliers

### 5. Comprehensive Dashboard
- Combined view of all key metrics
- Single-page summary of simulation results
- Includes summary statistics panel

## File Organization

### CSV Data Structure
```
results/
├── scenario_YYYYMMDD_HHMMSS/
│   ├── farmer_data.csv      # Farmer time series (yields, budgets, catches)
│   ├── fish_data.csv        # Fish population dynamics 
│   ├── water_data.csv       # Water resource data
│   ├── summary.csv          # Aggregated statistics
│   └── metadata.txt         # Run information
```

### Generated Plot Files
- `dashboard_[scenario].png` - Comprehensive dashboard
- `farmer_returns_[scenario].png` - Farmer budget evolution
- `water_data_[scenario].png` - Water inflow patterns
- `fish_population_[scenario].png` - Fish dynamics
- `yield_boxplot_[scenario].png` - Yield distributions

## Configuration

The plotting behavior is controlled through `config.yaml`:

```yaml
output:
  save_csv: true              # Enable CSV export
  results_dir: "results"      # Output directory
  include_metadata: true      # Generate metadata files
  export_components:
    farmer_data: true         # Export farmer time series
    fish_data: true           # Export fish population data
    water_data: true          # Export water resource data
    summary: true             # Export summary statistics
```

## Advanced Usage

### Plot from Specific Scenario
```python
from csv_plots import plot_from_scenario

# Plot results from a specific scenario
plot_from_scenario("centralized_fishing")
```

### Custom Plotting
```python
from csv_plots import CSVPlotter
import matplotlib.pyplot as plt

plotter = CSVPlotter()
data = plotter.load_csv_data(results_dir)

# Custom analysis using the CSV data
farmer_data = data['farmer_data']
# ... your custom plotting code
```

### Batch Plot Generation
```python
from main import run_config_based_experiments

# Run all scenarios and generate CSV data
results = run_config_based_experiments()

# Then generate plots for each scenario
scenarios = ["default", "centralized_fishing", "llm_together"]
for scenario in scenarios:
    plot_from_scenario(scenario)
```

## Migration from Old Implementation

### Old Approach
```python
# Old: Direct plotting during simulation
results = run_multiple_sims(...)
box_plot(results)
fish_plot(deltas, adults, larvae, ...)
farmer_returns_plot(results_by_delta)
```

### New Approach  
```python
# New: CSV-based plotting
run_simulation_and_plot("default", save_plots=True)
# or
results = run_simulation_from_config("default")
plot_latest_results(save_plots=True)
```

## Benefits

1. **Data Persistence**: All simulation data saved for future analysis
2. **Reproducibility**: Exact same plots can be regenerated anytime
3. **Flexibility**: Easy to create custom plots and analyses
4. **Organization**: Timestamped results prevent data loss
5. **Sharing**: CSV files can be easily shared and analyzed by others
6. **Automation**: Consistent plot generation across different scenarios

## Troubleshooting

### No Results Found
```python
# Check if results exist
plotter = CSVPlotter()
print(plotter.get_latest_results_dir())
```

### Missing Data
- Ensure `save_csv=True` when running simulations
- Check that all CSV files exist in the results directory
- Verify the simulation completed successfully

### Plot Display Issues
```python
# For non-interactive environments
import matplotlib
matplotlib.use('Agg')
```

This new plotting system provides the same visualization capabilities as the original implementation while adding data persistence and improved flexibility for analysis.