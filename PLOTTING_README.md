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

## Individual Plot Extraction Utility

**NEW**: Extract individual plots from the comprehensive dashboard as separate PDF or PNG files.

### Interactive Mode (Recommended)

Run the interactive plot extractor:
```bash
python csv_plots.py
```

This launches a user-friendly interface with guided prompts:

```
CSV Plotter Utility
====================
1. Interactive plot extractor
2. Show help  
3. List available plots
4. Quick test (extract farmer_budgets from most recent folder)

Enter your choice: 1

📁 Available results folders (5 found):
  1. generative agent (qwen-32B)
  2. centralized
  3. cpr_pigouvian  
  4. procedural_ABM
  5. cpr_tragedy

🔍 Select a results folder: cpr

📂 Found matching folder: cpr_tragedy

📊 Available plot types:
  1. farmer_budgets - Farmer Budget Returns
  2. water_inflows - Water Inflows  
  3. fish_population - Fish Population
  4. activity_breakdown - Activity Breakdown
  5. all - Extract all plots

🎨 Select plot type(s): all

💾 Choose output format:
  1. PDF (high quality, recommended)
  2. PNG (web-friendly)
Your choice: 1

🚀 Starting extraction...
📊 Extracting farmer_budgets... ✅
📊 Extracting water_inflows... ✅  
📊 Extracting fish_population... ✅
📊 Extracting activity_breakdown... ✅

🎉 Successfully extracted 7/7 plots!
📁 Files saved to: results/cpr_tragedy/
```

### Command Line Interface

For programmatic use:
```python
from csv_plots import extract_plot, list_available_plots

# List all available plot types
list_available_plots()

# Extract specific plots
extract_plot("cpr_tragedy", "farmer_budgets", save_as_pdf=True)
extract_plot("centralized", "activity_breakdown", save_as_pdf=False)
extract_plot("scenario_folder", "water_inflows")

# Quick extraction of all plots
plot_types = ['farmer_budgets', 'water_inflows', 'fish_population', 
              'yield_distribution', 'budget_inequality', 'activity_breakdown', 'summary_stats']
              
for plot_type in plot_types:
    extract_plot("my_scenario", plot_type, save_as_pdf=True)
```

### Available Plot Types for Extraction

1. **`farmer_budgets`** - Farmer Budget Returns (with confidence intervals if multi-run data available)
2. **`water_inflows`** - Water Inflows (annual and July)
3. **`fish_population`** - Fish Population (adults and larvae)
4. **`yield_distribution`** - Yield Distribution (box plot)
5. **`budget_inequality`** - Budget Inequality (Theil Index with confidence intervals)
6. **`activity_breakdown`** - Activity Breakdown (stacked bar chart showing farmer activity distribution)
7. **`summary_stats`** - Summary Statistics (text summary)

### Output Features

- **Smart folder matching**: Use partial folder names (e.g., "cpr" matches "cpr_tragedy")
- **Multiple formats**: PDF (high quality) or PNG (web-friendly)
- **Descriptive filenames**: `farmer_budgets_scenario.pdf`, `activity_breakdown_scenario.png`
- **Automatic titles**: Plot titles include the scenario name
- **File size reporting**: Shows created file sizes for verification

### Use Cases

- **Research presentations**: Extract high-quality PDFs for papers and presentations
- **Web content**: Generate PNG files for websites and reports
- **Comparative analysis**: Extract the same plot type from multiple scenarios
- **Custom dashboards**: Build focused visualizations with specific plots
- **Data sharing**: Share individual charts without full dashboard complexity

### Quick Commands

```bash
# Interactive mode (recommended for first-time users)
python csv_plots.py

# Help and documentation
echo "2" | python csv_plots.py

# Quick test
echo "4" | python csv_plots.py

# List available plots
echo "3" | python csv_plots.py
```

The extraction utility preserves all styling, confidence intervals, and data relationships from the original dashboard while providing focused, publication-ready individual plots.