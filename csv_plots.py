"""
CSV-based plotting utilities for Water Use Simulation
Reads CSV data from results folder and generates plots like the original implementation

New Feature: Individual Plot Extraction
=====================================
You can now extract individual plots from the comprehensive dashboard as separate PDF or PNG files.

Usage Examples:
--------------
from csv_plots import extract_plot, list_available_plots

# List all available plot types
list_available_plots()

# Extract farmer budget plot as PDF
extract_plot("cpr_tragedy", "farmer_budgets", save_as_pdf=True)

# Extract activity breakdown as PNG  
extract_plot("scenario_folder", "activity_breakdown", save_as_pdf=False)

# Extract any other plot type
extract_plot("folder_name", "water_inflows")
extract_plot("folder_name", "fish_population") 
extract_plot("folder_name", "yield_distribution")
extract_plot("folder_name", "budget_inequality")
extract_plot("folder_name", "summary_stats")

Available Plot Types:
--------------------
- farmer_budgets: Farmer Budget Returns (with confidence intervals if multi-run data available)
- water_inflows: Water Inflows (annual and July)
- fish_population: Fish Population (adults and larvae)
- yield_distribution: Yield Distribution (box plot)
- budget_inequality: Budget Inequality (Theil Index with confidence intervals)
- activity_breakdown: Activity Breakdown (stacked bar chart showing farmer activity distribution)
- summary_stats: Summary Statistics (text summary)

The extracted plots will be saved in the same results folder with descriptive filenames like:
- farmer_budgets_scenario.pdf
- activity_breakdown_scenario.png
- etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import Dict, List, Optional, Tuple
import math

class CSVPlotter:
    """Class to generate plots from CSV simulation results"""
    
    def __init__(self, results_base_dir: str = "results"):
        """
        Initialize CSV plotter
        
        Args:
            results_base_dir: Base directory containing simulation results
        """
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.results_base_dir = os.path.join(project_root, results_base_dir)
        
    def get_latest_results_dir(self) -> Optional[str]:
        """Get the most recent results directory"""
        if not os.path.exists(self.results_base_dir):
            return None
            
        result_dirs = [d for d in os.listdir(self.results_base_dir) 
                      if os.path.isdir(os.path.join(self.results_base_dir, d))]
        
        if not result_dirs:
            return None
            
        # Sort by timestamp (assuming format: scenario_YYYYMMDD_HHMMSS)
        latest_dir = sorted(result_dirs)[-1]
        return os.path.join(self.results_base_dir, latest_dir)
    
    def get_results_dir_by_scenario(self, scenario_name: str) -> Optional[str]:
        """Get the most recent results directory for a specific scenario"""
        if not os.path.exists(self.results_base_dir):
            return None
            
        result_dirs = [d for d in os.listdir(self.results_base_dir) 
                      if os.path.isdir(os.path.join(self.results_base_dir, d)) 
                      and d.startswith(scenario_name)]
        
        if not result_dirs:
            return None
            
        # Sort by timestamp and get the latest
        latest_dir = sorted(result_dirs)[-1]
        return os.path.join(self.results_base_dir, latest_dir)
    
    def load_csv_data(self, results_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from a results directory
        
        Args:
            results_dir: Path to the results directory
            
        Returns:
            Dictionary mapping data type to DataFrame
        """
        data = {}
        
        csv_files = {
            'farmer_data': 'farmer_data.csv',
            'fish_data': 'fish_data.csv',
            'water_data': 'water_data.csv',
            'summary': 'summary.csv',
            'multi_run_budgets': 'farmer_budgets_all_runs.csv'  # Add multi-run budget data
        }
        
        for data_type, filename in csv_files.items():
            file_path = os.path.join(results_dir, filename)
            if os.path.exists(file_path):
                data[data_type] = pd.read_csv(file_path)
            else:
                print(f"Warning: {filename} not found in {results_dir}")
                
        return data
    
    def farmer_returns_plot(self, results_dir: str = None, farmer_id: int = 9, 
                          title: str = None, save_path: str = None):
        """
        Plot farmer returns over time (equivalent to original farmer_returns_plot)
        
        Args:
            results_dir: Path to results directory (uses latest if None)
            farmer_id: ID of farmer to plot (default: 9)
            title: Custom title for the plot
            save_path: Path to save the plot (auto-generates in results dir if None)
        """
        if results_dir is None:
            results_dir = self.get_latest_results_dir()
            
        if results_dir is None:
            print("No results directory found!")
            return
            
        data = self.load_csv_data(results_dir)
        
        if 'farmer_data' not in data:
            print("Farmer data not found!")
            return
            
        farmer_data = data['farmer_data']
        
        # Filter data for the specific farmer
        farmer_budget = farmer_data[farmer_data['farmer_id'] == farmer_id]
        
        if farmer_budget.empty:
            print(f"No data found for farmer {farmer_id}")
            return
            
        plt.figure(figsize=(8, 6))
        
        years = farmer_budget['year'].values
        budgets = farmer_budget['budget'].values
        
        plt.plot(years, budgets, 'o-', label=f'Farmer {farmer_id}')
        
        plt.xlabel("Year")
        plt.ylabel(f"Farmer {farmer_id} Budget")
        plt.title(title or f"Farmer {farmer_id} Returns Over Time")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        
        # Auto-generate save path in results directory if not provided
        if save_path is None:
            scenario_name = os.path.basename(results_dir).split('_')[0]
            save_path = os.path.join(results_dir, f"farmer_returns_{scenario_name}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Farmer returns plot saved to: {save_path}")
    
    def water_plot(self, results_dir: str = None, title: str = None, save_path: str = None):
        """
        Plot water inflow data over time
        
        Args:
            results_dir: Path to results directory (uses latest if None)
            title: Custom title for the plot
            save_path: Path to save the plot (auto-generates in results dir if None)
        """
        if results_dir is None:
            results_dir = self.get_latest_results_dir()
            
        if results_dir is None:
            print("No results directory found!")
            return
            
        data = self.load_csv_data(results_dir)
        
        if 'water_data' not in data:
            print("Water data not found!")
            return
            
        water_data = data['water_data']
        
        plt.figure(figsize=(10, 6))
        
        years = water_data['year'].values
        annual_inflow = water_data['annual_inflow'].values
        july_inflow = water_data['july_inflow'].values
        
        plt.subplot(2, 1, 1)
        plt.plot(years, annual_inflow, 'b-o', label='Annual Inflow')
        plt.ylabel("Annual Inflow")
        plt.title(title or "Water Resource Data Over Time")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        
        plt.subplot(2, 1, 2)
        plt.plot(years, july_inflow, 'r-s', label='July Inflow')
        plt.xlabel("Year")
        plt.ylabel("July Inflow")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        
        # Auto-generate save path in results directory if not provided
        if save_path is None:
            scenario_name = os.path.basename(results_dir).split('_')[0]
            save_path = os.path.join(results_dir, f"water_data_{scenario_name}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Water plot saved to: {save_path}")
    
    def fish_plot(self, results_dir: str = None, title: str = None, save_path: str = None):
        """
        Plot fish population dynamics over time
        
        Args:
            results_dir: Path to results directory (uses latest if None)
            title: Custom title for the plot
            save_path: Path to save the plot (auto-generates in results dir if None)
        """
        if results_dir is None:
            results_dir = self.get_latest_results_dir()
            
        if results_dir is None:
            print("No results directory found!")
            return
            
        data = self.load_csv_data(results_dir)
        
        if 'fish_data' not in data:
            print("Fish data not found!")
            return
            
        fish_data = data['fish_data']
        
        plt.figure(figsize=(12, 5))
        
        years = fish_data['year'].values
        adult_fish = fish_data['adult_fish'].values
        larvae = fish_data['larvae'].values
        total_fish = fish_data['total_fish'].values
        
        # Panel (a) – Adults and Total Fish
        plt.subplot(1, 2, 1)
        plt.plot(years, adult_fish, 'ko-', label='Adult Fish')
        plt.plot(years, total_fish, 'b--', label='Total Fish')
        plt.xlabel("Year")
        plt.ylabel("Fish Population")
        plt.title("Fish Population - Adults vs Total")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        
        # Panel (b) – Larvae
        plt.subplot(1, 2, 2)
        plt.plot(years, larvae, 'r-s', label='Larvae')
        plt.xlabel("Year")
        plt.ylabel("Larvae Population")
        plt.title("Fish Population - Larvae")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        
        plt.suptitle(title or "Fish Population Dynamics Over Time")
        plt.tight_layout()
        
        # Auto-generate save path in results directory if not provided
        if save_path is None:
            scenario_name = os.path.basename(results_dir).split('_')[0]
            save_path = os.path.join(results_dir, f"fish_population_{scenario_name}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Fish plot saved to: {save_path}")
    
    def box_plot_yields(self, results_dir: str = None, title: str = None, save_path: str = None):
        """
        Create box plots of farmer yields (equivalent to original box_plot functions)
        
        Args:
            results_dir: Path to results directory (uses latest if None)
            title: Custom title for the plot
            save_path: Path to save the plot (auto-generates in results dir if None)
        """
        if results_dir is None:
            results_dir = self.get_latest_results_dir()
            
        if results_dir is None:
            print("No results directory found!")
            return
            
        data = self.load_csv_data(results_dir)
        
        if 'farmer_data' not in data:
            print("Farmer data not found!")
            return
            
        farmer_data = data['farmer_data']
        
        # Pivot data to get yields by farmer and year
        yield_pivot = farmer_data.pivot(index='year', columns='farmer_id', values='yield')
        
        plt.figure(figsize=(10, 6))
        
        # Create box plot
        box_data = [yield_pivot[col].dropna().values for col in yield_pivot.columns]
        farmer_labels = [f'Farmer {col}' for col in yield_pivot.columns]
        
        plt.boxplot(box_data, labels=farmer_labels)
        plt.xlabel("Farmers")
        plt.ylabel("Yield")
        plt.title(title or "Distribution of Yields by Farmer")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            # Auto-generate save path in results directory
            scenario_name = os.path.basename(results_dir).split('_')[0]
            save_path = os.path.join(results_dir, f"yield_boxplot_{scenario_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Box plot saved to: {save_path}")
        
        plt.close()  # Close the figure to free memory
    
    def comprehensive_dashboard(self, results_dir: str = None, save_path: str = None):
        """
        Create a comprehensive dashboard with all plots
        
        Args:
            results_dir: Path to results directory (uses latest if None)
            save_path: Path to save the plot (auto-generates in results dir if None)
        """
        if results_dir is None:
            results_dir = self.get_latest_results_dir()
            
        if results_dir is None:
            print("No results directory found!")
            return
            
        data = self.load_csv_data(results_dir)
        
        # Get scenario name from directory
        scenario_name = os.path.basename(results_dir).split('_')[0]
        
        fig = plt.figure(figsize=(20, 16))  # Expanded figure size for additional plot
        
        # Farmer returns with confidence intervals (top left)
        plt.subplot(3, 3, 1)
        if 'multi_run_budgets' in data and not data['multi_run_budgets'].empty:
            # Plot confidence intervals across multiple runs
            multi_run_data = data['multi_run_budgets']
            
            # Calculate mean and confidence intervals for each farmer by year
            farmer_ids = sorted(multi_run_data['farmer_id'].unique())
            years = sorted(multi_run_data['year'].unique())
            
            for farmer_id in farmer_ids:
                farmer_multi_data = multi_run_data[multi_run_data['farmer_id'] == farmer_id]
                
                means = []
                ci_lower = []
                ci_upper = []
                
                for year in years:
                    year_data = farmer_multi_data[farmer_multi_data['year'] == year]['budget']
                    if len(year_data) > 0:
                        mean_budget = np.mean(year_data)
                        std_budget = np.std(year_data)
                        n = len(year_data)
                        
                        # Calculate 95% confidence interval
                        ci_margin = 1.96 * (std_budget / np.sqrt(n)) if n > 1 else 0
                        
                        means.append(mean_budget)
                        ci_lower.append(mean_budget - ci_margin)
                        ci_upper.append(mean_budget + ci_margin)
                    else:
                        means.append(0)
                        ci_lower.append(0)
                        ci_upper.append(0)
                
                # Plot confidence interval
                color = plt.cm.tab10(farmer_id - 1)  # Use consistent colors
                plt.fill_between(years, ci_lower, ci_upper, alpha=0.2, color=color)
                
                # Plot mean line as continuous
                plt.plot(years, means, '-', color=color, label=f'Farmer {farmer_id}', linewidth=1.5)
            
            plt.xlabel("Year")
            plt.ylabel("Budget")
            plt.title("All Farmers' Budget Returns (with 95% CI)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.grid(True, alpha=0.3)
            
        elif 'farmer_data' in data:
            # Fallback to single run data if multi-run data not available
            farmer_data = data['farmer_data']
            farmer_ids = sorted(farmer_data['farmer_id'].unique())
            for farmer_id in farmer_ids:
                farmer_budget = farmer_data[farmer_data['farmer_id'] == farmer_id]
                if not farmer_budget.empty:
                    color = plt.cm.tab10(farmer_id - 1)
                    
                    # Plot continuous line
                    plt.plot(farmer_budget['year'], farmer_budget['budget'], '-', 
                            color=color, label=f'Farmer {farmer_id}', linewidth=1)
            plt.xlabel("Year")
            plt.ylabel("Budget")
            plt.title("All Farmers' Budget Returns (Single Run)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.grid(True, alpha=0.3)
        
        # Water inflows (top middle)
        plt.subplot(3, 3, 2)
        if 'water_data' in data:
            water_data = data['water_data']
            plt.plot(water_data['year'], water_data['annual_inflow'], 'b-o', label='Annual')
            plt.plot(water_data['year'], water_data['july_inflow'], 'r-s', label='July')
            plt.xlabel("Year")
            plt.ylabel("Inflow")
            plt.title("Water Inflows")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Fish populations (top right)
        plt.subplot(3, 3, 3)
        if 'fish_data' in data:
            fish_data = data['fish_data']
            plt.plot(fish_data['year'], fish_data['adult_fish'], 'ko-', label='Adults')
            plt.plot(fish_data['year'], fish_data['larvae'], 'r-s', label='Larvae')
            plt.xlabel("Year")
            plt.ylabel("Fish Count")
            plt.title("Fish Population")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Yield box plot (middle left)
        plt.subplot(3, 3, 4)
        if 'farmer_data' in data:
            farmer_data = data['farmer_data']
            yield_pivot = farmer_data.pivot(index='year', columns='farmer_id', values='yield')
            box_data = [yield_pivot[col].dropna().values for col in yield_pivot.columns]
            plt.boxplot(box_data, labels=[f'F{col}' for col in yield_pivot.columns])
            plt.xlabel("Farmers")
            plt.ylabel("Yield")
            plt.title("Yield Distribution")
            plt.xticks(rotation=45)
        
        # Budget over time (middle center)
        plt.subplot(3, 3, 5)
        if 'multi_run_budgets' in data and not data['multi_run_budgets'].empty:
            # Calculate Theil Index with confidence intervals across multiple runs
            multi_run_data = data['multi_run_budgets']
            years = sorted(multi_run_data['year'].unique())
            runs = sorted(multi_run_data['run'].unique())
            
            def theil_index(values):
                """Calculate Theil Index (T) for inequality measurement"""
                values = np.array(values)
                # Replace negative values with 0.1 to avoid log issues
                values = np.where(values < 0, 0.1, values)
                # Remove only zero values to avoid log issues
                values = values[values > 0]
                if len(values) == 0:
                    return 0
                mean_val = np.mean(values)
                if mean_val == 0:
                    return 0
                # Theil Index formula: T = (1/n) * sum((x_i/μ) * ln(x_i/μ))
                ratios = values / mean_val
                theil = np.mean(ratios * np.log(ratios))
                return theil
            
            # Calculate Theil Index for each run and year
            theil_data = []
            for run in runs:
                run_data = multi_run_data[multi_run_data['run'] == run]
                theil_values = []
                for year in years:
                    year_budgets = run_data[run_data['year'] == year]['budget'].values
                    if len(year_budgets) > 0:
                        theil_val = theil_index(year_budgets)
                        theil_values.append(theil_val)
                    else:
                        theil_values.append(0)
                theil_data.append(theil_values)
            
            # Convert to numpy array for easier calculation
            theil_array = np.array(theil_data)  # Shape: (runs, years)
            
            # Calculate mean and confidence intervals
            theil_means = np.mean(theil_array, axis=0)
            theil_stds = np.std(theil_array, axis=0)
            n_runs = len(runs)
            
            # Calculate 95% confidence intervals
            ci_margins = 1.96 * (theil_stds / np.sqrt(n_runs)) if n_runs > 1 else np.zeros_like(theil_stds)
            ci_lower = theil_means - ci_margins
            ci_upper = theil_means + ci_margins
            
            # Plot mean line and confidence interval
            plt.plot(years, theil_means, 'g-o', linewidth=2, label='Mean Theil Index')
            plt.fill_between(years, ci_lower, ci_upper, alpha=0.2, color='green', label='95% CI')
            
            plt.xlabel("Year")
            plt.ylabel("Theil Index")
            plt.title("Budget Inequality (Theil Index with 95% CI)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        elif 'farmer_data' in data:
            # Fallback to single run data if multi-run data not available
            farmer_data = data['farmer_data']
            
            def theil_index(values):
                """Calculate Theil Index (T) for inequality measurement"""
                values = np.array(values)
                # Replace negative values with 0.1 to avoid log issues
                values = np.where(values < 0, 0.1, values)
                # Remove only zero values to avoid log issues
                values = values[values > 0]
                if len(values) == 0:
                    return 0
                mean_val = np.mean(values)
                if mean_val == 0:
                    return 0
                # Theil Index formula: T = (1/n) * sum((x_i/μ) * ln(x_i/μ))
                ratios = values / mean_val
                theil = np.mean(ratios * np.log(ratios))
                return theil
            
            theil_by_year = farmer_data.groupby('year')['budget'].apply(theil_index)
            plt.plot(theil_by_year.index, theil_by_year.values, 'g-o')
            plt.xlabel("Year")
            plt.ylabel("Theil Index")
            plt.title("Budget Inequality (Theil Index - Single Run)")
            plt.grid(True, alpha=0.3)
        
        # Activity breakdown by year (middle right)
        plt.subplot(3, 3, 6)
        if 'farmer_data' in data:
            farmer_data = data['farmer_data']
            years = sorted(farmer_data['year'].unique())
            
            # Calculate activity percentages for each year
            both_activities = []
            irrigation_only = []
            fishing_only = []
            no_activity = []
            
            for year in years:
                year_data = farmer_data[farmer_data['year'] == year]
                total_farmers = len(year_data)
                
                if total_farmers > 0:
                    both_count = len(year_data[(year_data['irrigated_fields'] > 0) & (year_data['catch'] > 0)])
                    irrigation_count = len(year_data[(year_data['irrigated_fields'] > 0) & (year_data['catch'] == 0)])
                    fishing_count = len(year_data[(year_data['irrigated_fields'] == 0) & (year_data['catch'] > 0)])
                    none_count = len(year_data[(year_data['irrigated_fields'] == 0) & (year_data['catch'] == 0)])
                    
                    # Convert to percentages
                    both_activities.append(both_count / total_farmers * 100)
                    irrigation_only.append(irrigation_count / total_farmers * 100)
                    fishing_only.append(fishing_count / total_farmers * 100)
                    no_activity.append(none_count / total_farmers * 100)
                else:
                    both_activities.append(0)
                    irrigation_only.append(0)
                    fishing_only.append(0)
                    no_activity.append(0)
            
            # Create stacked bar chart
            width = 0.8
            plt.bar(years, no_activity, width, label='No Activity', color='lightgray', alpha=0.7)
            plt.bar(years, fishing_only, width, bottom=no_activity, label='Fishing Only', color='red', alpha=0.7)
            
            bottom_irrigation = [na + fo for na, fo in zip(no_activity, fishing_only)]
            plt.bar(years, irrigation_only, width, bottom=bottom_irrigation, label='Irrigation Only', color='blue', alpha=0.7)
            
            bottom_both = [bi + io for bi, io in zip(bottom_irrigation, irrigation_only)]
            plt.bar(years, both_activities, width, bottom=bottom_both, label='Both Activities', color='purple', alpha=0.7)
            
            plt.xlabel("Year")
            plt.ylabel("Percentage of Farmers (%)")
            plt.title("Farmer Activity Distribution by Year")
            plt.legend(loc='upper right', fontsize=8)
            plt.grid(True, alpha=0.3, axis='y')
            plt.ylim(0, 100)
        
        # Summary stats (bottom left)
        plt.subplot(3, 3, 7)
        if 'summary' in data:
            summary = data['summary'].iloc[0]
            
            # Parse config_params from string representation
            try:
                import ast
                config_params = ast.literal_eval(summary['config_params'])
                memory_strength = config_params.get('memory_strength', 'N/A')
                use_cpr_game = config_params.get('use_cpr_game', 'N/A')
            except (ValueError, SyntaxError):
                memory_strength = 'N/A'
                use_cpr_game = 'N/A'
            stats_text = f"""
            Scenario: {summary['scenario']}
            Years: {summary['years_simulated']}
            Farmers: {summary['num_farmers']}
            Centralized: {summary['centralized']}
            Fishing: {summary['fishing_enabled']}
            Memory Strength: {memory_strength}
            CPR Game: {use_cpr_game}

            Total Yield: {summary['total_yield']:.1f}
            Avg Final Budget: {summary['avg_final_budget']:.1f}
            Final Fish: {summary['final_fish_total']:.0f}
            """
            plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='center')
            plt.axis('off')
            plt.title("Summary Statistics")
        
        plt.suptitle(f"Simulation Dashboard - {scenario_name}", fontsize=16)
        plt.tight_layout()
        
        # Auto-generate save path in results directory if not provided
        if save_path is None:
            save_path = os.path.join(results_dir, f"dashboard_{scenario_name}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Dashboard saved to: {save_path}")
    
    def extract_individual_plot(self, folder_name: str, plot_name: str, save_as_pdf: bool = True):
        """
        Extract an individual plot from the dashboard and save as a separate file
        
        Args:
            folder_name: Name of the results folder to use
            plot_name: Name of the plot to extract. Options:
                - 'farmer_budgets': Farmer Budget Returns (with CI if available)
                - 'water_inflows': Water Inflows (annual and July)
                - 'fish_population': Fish Population (adults and larvae)  
                - 'yield_distribution': Yield Distribution (box plot)
                - 'budget_inequality': Budget Inequality (Theil Index with CI)
                - 'activity_breakdown': Activity Breakdown (stacked bar chart)
                - 'summary_stats': Summary Statistics (text summary)
            save_as_pdf: If True, saves as PDF; if False, saves as PNG
        """
        # Find the results directory
        results_dir = None
        if os.path.exists(self.results_base_dir):
            for d in os.listdir(self.results_base_dir):
                dir_path = os.path.join(self.results_base_dir, d)
                if os.path.isdir(dir_path) and folder_name in d:
                    results_dir = dir_path
                    break
        
        if results_dir is None:
            print(f"No results directory found containing: {folder_name}")
            return
            
        # Load data
        data = self.load_csv_data(results_dir)
        
        # Get scenario name from directory for title
        scenario_name = os.path.basename(results_dir).split('_')[0]
        title = f"{scenario_name} - {plot_name.replace('_', ' ').title()}"
        
        # Create individual plot based on plot_name
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if plot_name == 'farmer_budgets':
            self._plot_farmer_budgets(data, ax, title)
        elif plot_name == 'water_inflows':
            self._plot_water_inflows(data, ax, title)
        elif plot_name == 'fish_population':
            self._plot_fish_population(data, ax, title)
        elif plot_name == 'yield_distribution':
            self._plot_yield_distribution(data, ax, title)
        elif plot_name == 'budget_inequality':
            self._plot_budget_inequality(data, ax, title)
        elif plot_name == 'activity_breakdown':
            self._plot_activity_breakdown(data, ax, title)
        elif plot_name == 'summary_stats':
            self._plot_summary_stats(data, ax, title)
        else:
            print(f"Unknown plot name: {plot_name}")
            print("Available options: farmer_budgets, water_inflows, fish_population, yield_distribution, budget_inequality, activity_breakdown, summary_stats")
            plt.close()
            return
        
        # Save the plot
        file_extension = 'pdf' if save_as_pdf else 'png'
        save_path = os.path.join(results_dir, f"{plot_name}_{scenario_name}.{file_extension}")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual plot saved to: {save_path}")
    
    def _plot_farmer_budgets(self, data, ax, title):
        """Helper method to plot farmer budgets"""
        if 'multi_run_budgets' in data and not data['multi_run_budgets'].empty:
            # Plot confidence intervals across multiple runs
            multi_run_data = data['multi_run_budgets']
            
            # Calculate mean and confidence intervals for each farmer by year
            farmer_ids = sorted(multi_run_data['farmer_id'].unique())
            years = sorted(multi_run_data['year'].unique())
            
            for farmer_id in farmer_ids:
                farmer_multi_data = multi_run_data[multi_run_data['farmer_id'] == farmer_id]
                
                means = []
                ci_lower = []
                ci_upper = []
                
                for year in years:
                    year_data = farmer_multi_data[farmer_multi_data['year'] == year]['budget']
                    if len(year_data) > 0:
                        mean_budget = np.mean(year_data)
                        std_budget = np.std(year_data)
                        n = len(year_data)
                        
                        # Calculate 95% confidence interval
                        ci_margin = 1.96 * (std_budget / np.sqrt(n)) if n > 1 else 0
                        
                        means.append(mean_budget)
                        ci_lower.append(mean_budget - ci_margin)
                        ci_upper.append(mean_budget + ci_margin)
                    else:
                        means.append(0)
                        ci_lower.append(0)
                        ci_upper.append(0)
                
                # Plot confidence interval
                color = plt.cm.tab10(farmer_id - 1)  # Use consistent colors
                ax.fill_between(years, ci_lower, ci_upper, alpha=0.2, color=color)
                
                # Plot mean line as continuous
                ax.plot(years, means, '-', color=color, label=f'Farmer {farmer_id}', linewidth=1.5)
            
            ax.set_xlabel("Year")
            ax.set_ylabel("Budget")
            ax.set_title(title)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
        elif 'farmer_data' in data:
            # Fallback to single run data if multi-run data not available
            farmer_data = data['farmer_data']
            farmer_ids = sorted(farmer_data['farmer_id'].unique())
            for farmer_id in farmer_ids:
                farmer_budget = farmer_data[farmer_data['farmer_id'] == farmer_id]
                if not farmer_budget.empty:
                    color = plt.cm.tab10(farmer_id - 1)
                    
                    # Plot continuous line
                    ax.plot(farmer_budget['year'], farmer_budget['budget'], '-', 
                            color=color, label=f'Farmer {farmer_id}', linewidth=1)
            ax.set_xlabel("Year")
            ax.set_ylabel("Budget")
            ax.set_title(title)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    def _plot_water_inflows(self, data, ax, title):
        """Helper method to plot water inflows"""
        if 'water_data' in data:
            water_data = data['water_data']
            ax.plot(water_data['year'], water_data['annual_inflow'], 'b-o', label='Annual')
            ax.plot(water_data['year'], water_data['july_inflow'], 'r-s', label='July')
            ax.set_xlabel("Year")
            ax.set_ylabel("Inflow")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_fish_population(self, data, ax, title):
        """Helper method to plot fish population"""
        if 'fish_data' in data:
            fish_data = data['fish_data']
            ax.plot(fish_data['year'], fish_data['adult_fish'], 'ko-', label='Adults')
            ax.plot(fish_data['year'], fish_data['larvae'], 'r-s', label='Larvae')
            ax.set_xlabel("Year")
            ax.set_ylabel("Fish Count")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_yield_distribution(self, data, ax, title):
        """Helper method to plot yield distribution"""
        if 'farmer_data' in data:
            farmer_data = data['farmer_data']
            yield_pivot = farmer_data.pivot(index='year', columns='farmer_id', values='yield')
            box_data = [yield_pivot[col].dropna().values for col in yield_pivot.columns]
            ax.boxplot(box_data, labels=[f'F{col}' for col in yield_pivot.columns])
            ax.set_xlabel("Farmers")
            ax.set_ylabel("Yield")
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_budget_inequality(self, data, ax, title):
        """Helper method to plot budget inequality (Theil Index)"""
        if 'multi_run_budgets' in data and not data['multi_run_budgets'].empty:
            # Calculate Theil Index with confidence intervals across multiple runs
            multi_run_data = data['multi_run_budgets']
            years = sorted(multi_run_data['year'].unique())
            runs = sorted(multi_run_data['run'].unique())
            
            def theil_index(values):
                """Calculate Theil Index (T) for inequality measurement"""
                values = np.array(values)
                # Replace negative values with 0.1 to avoid log issues
                values = np.where(values < 0, 0.1, values)
                # Remove only zero values to avoid log issues
                values = values[values > 0]
                if len(values) == 0:
                    return 0
                mean_val = np.mean(values)
                if mean_val == 0:
                    return 0
                # Theil Index formula: T = (1/n) * sum((x_i/μ) * ln(x_i/μ))
                ratios = values / mean_val
                theil = np.mean(ratios * np.log(ratios))
                return theil
            
            # Calculate Theil Index for each run and year
            theil_data = []
            for run in runs:
                run_data = multi_run_data[multi_run_data['run'] == run]
                theil_values = []
                for year in years:
                    year_budgets = run_data[run_data['year'] == year]['budget'].values
                    if len(year_budgets) > 0:
                        theil_val = theil_index(year_budgets)
                        theil_values.append(theil_val)
                    else:
                        theil_values.append(0)
                theil_data.append(theil_values)
            
            # Convert to numpy array for easier calculation
            theil_array = np.array(theil_data)  # Shape: (runs, years)
            
            # Calculate mean and confidence intervals
            theil_means = np.mean(theil_array, axis=0)
            theil_stds = np.std(theil_array, axis=0)
            n_runs = len(runs)
            
            # Calculate 95% confidence intervals
            ci_margins = 1.96 * (theil_stds / np.sqrt(n_runs)) if n_runs > 1 else np.zeros_like(theil_stds)
            ci_lower = theil_means - ci_margins
            ci_upper = theil_means + ci_margins
            
            # Plot mean line and confidence interval
            ax.plot(years, theil_means, 'g-o', linewidth=2, label='Mean Theil Index')
            ax.fill_between(years, ci_lower, ci_upper, alpha=0.2, color='green', label='95% CI')
            
            ax.set_xlabel("Year")
            ax.set_ylabel("Theil Index")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif 'farmer_data' in data:
            # Fallback to single run data if multi-run data not available
            farmer_data = data['farmer_data']
            
            def theil_index(values):
                """Calculate Theil Index (T) for inequality measurement"""
                values = np.array(values)
                # Replace negative values with 0.1 to avoid log issues
                values = np.where(values < 0, 0.1, values)
                # Remove only zero values to avoid log issues
                values = values[values > 0]
                if len(values) == 0:
                    return 0
                mean_val = np.mean(values)
                if mean_val == 0:
                    return 0
                # Theil Index formula: T = (1/n) * sum((x_i/μ) * ln(x_i/μ))
                ratios = values / mean_val
                theil = np.mean(ratios * np.log(ratios))
                return theil
            
            theil_by_year = farmer_data.groupby('year')['budget'].apply(theil_index)
            ax.plot(theil_by_year.index, theil_by_year.values, 'g-o')
            ax.set_xlabel("Year")
            ax.set_ylabel("Theil Index")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
    
    def _plot_activity_breakdown(self, data, ax, title):
        """Helper method to plot activity breakdown"""
        if 'farmer_data' in data:
            farmer_data = data['farmer_data']
            years = sorted(farmer_data['year'].unique())
            
            # Calculate activity percentages for each year
            both_activities = []
            irrigation_only = []
            fishing_only = []
            no_activity = []
            
            for year in years:
                year_data = farmer_data[farmer_data['year'] == year]
                total_farmers = len(year_data)
                
                if total_farmers > 0:
                    both_count = len(year_data[(year_data['irrigated_fields'] > 0) & (year_data['catch'] > 0)])
                    irrigation_count = len(year_data[(year_data['irrigated_fields'] > 0) & (year_data['catch'] == 0)])
                    fishing_count = len(year_data[(year_data['irrigated_fields'] == 0) & (year_data['catch'] > 0)])
                    none_count = len(year_data[(year_data['irrigated_fields'] == 0) & (year_data['catch'] == 0)])
                    
                    # Convert to percentages
                    both_activities.append(both_count / total_farmers * 100)
                    irrigation_only.append(irrigation_count / total_farmers * 100)
                    fishing_only.append(fishing_count / total_farmers * 100)
                    no_activity.append(none_count / total_farmers * 100)
                else:
                    both_activities.append(0)
                    irrigation_only.append(0)
                    fishing_only.append(0)
                    no_activity.append(0)
            
            # Create stacked bar chart
            width = 0.8
            ax.bar(years, no_activity, width, label='No Activity', color='lightgray', alpha=0.7)
            ax.bar(years, fishing_only, width, bottom=no_activity, label='Fishing Only', color='red', alpha=0.7)
            
            bottom_irrigation = [na + fo for na, fo in zip(no_activity, fishing_only)]
            ax.bar(years, irrigation_only, width, bottom=bottom_irrigation, label='Irrigation Only', color='blue', alpha=0.7)
            
            bottom_both = [bi + io for bi, io in zip(bottom_irrigation, irrigation_only)]
            ax.bar(years, both_activities, width, bottom=bottom_both, label='Both Activities', color='purple', alpha=0.7)
            
            ax.set_xlabel("Year")
            ax.set_ylabel("Percentage of Farmers (%)")
            ax.set_title(title)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 100)
    
    def _plot_summary_stats(self, data, ax, title):
        """Helper method to plot summary statistics"""
        if 'summary' in data:
            summary = data['summary'].iloc[0]
            
            # Parse config_params from string representation
            try:
                import ast
                config_params = ast.literal_eval(summary['config_params'])
                memory_strength = config_params.get('memory_strength', 'N/A')
                use_cpr_game = config_params.get('use_cpr_game', 'N/A')
            except (ValueError, SyntaxError):
                memory_strength = 'N/A'
                use_cpr_game = 'N/A'
            stats_text = f"""
            Scenario: {summary['scenario']}
            Years: {summary['years_simulated']}
            Farmers: {summary['num_farmers']}
            Centralized: {summary['centralized']}
            Fishing: {summary['fishing_enabled']}
            Memory Strength: {memory_strength}
            CPR Game: {use_cpr_game}

            Total Yield: {summary['total_yield']:.1f}
            Avg Final Budget: {summary['avg_final_budget']:.1f}
            Final Fish: {summary['final_fish_total']:.0f}
            """
            ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, 
                    fontsize=12, verticalalignment='center')
            ax.axis('off')
            ax.set_title(title)

def plot_from_latest_results():
    """Convenience function to plot from the latest results"""
    plotter = CSVPlotter()
    plotter.comprehensive_dashboard()

def plot_from_scenario(scenario_name: str):
    """Convenience function to plot from a specific scenario"""
    plotter = CSVPlotter()
    results_dir = plotter.get_results_dir_by_scenario(scenario_name)
    if results_dir:
        plotter.comprehensive_dashboard(results_dir)
    else:
        print(f"No results found for scenario: {scenario_name}")

def extract_plot(folder_name: str, plot_name: str, save_as_pdf: bool = True):
    """
    Convenience function to extract an individual plot as a separate file
    
    Args:
        folder_name: Name of the results folder to use (partial match)
        plot_name: Name of the plot to extract. Options:
            - 'farmer_budgets': Farmer Budget Returns (with CI if available)
            - 'water_inflows': Water Inflows (annual and July)
            - 'fish_population': Fish Population (adults and larvae)  
            - 'yield_distribution': Yield Distribution (box plot)
            - 'budget_inequality': Budget Inequality (Theil Index with CI)
            - 'activity_breakdown': Activity Breakdown (stacked bar chart)
            - 'summary_stats': Summary Statistics (text summary)
        save_as_pdf: If True, saves as PDF; if False, saves as PNG
    
    Example:
        extract_plot("default_20241221", "farmer_budgets", save_as_pdf=True)
        extract_plot("scenario1", "activity_breakdown", save_as_pdf=False)
    """
    plotter = CSVPlotter()
    plotter.extract_individual_plot(folder_name, plot_name, save_as_pdf)

def list_available_plots():
    """List all available plot types that can be extracted"""
    plots = [
        'farmer_budgets - Farmer Budget Returns (with confidence intervals if multi-run)',
        'water_inflows - Water Inflows (annual and July)',
        'fish_population - Fish Population (adults and larvae)',
        'yield_distribution - Yield Distribution (box plot)',
        'budget_inequality - Budget Inequality (Theil Index with confidence intervals)',
        'activity_breakdown - Activity Breakdown (stacked bar chart)',
        'summary_stats - Summary Statistics (text summary)'
    ]
    
    print("Available plot types:")
    for plot in plots:
        print(f"  - {plot}")
    
    print("\nUsage example:")
    print("  extract_plot('folder_name', 'plot_name', save_as_pdf=True)")
    print("  extract_plot('default_20241221', 'farmer_budgets', save_as_pdf=True)")

if __name__ == "__main__":
    def interactive_plot_extractor():
        """Interactive utility for extracting individual plots"""
        print("CSV Plotter Utility - Interactive Plot Extractor")
        print("=" * 50)
        
        # Check if results directory exists
        plotter = CSVPlotter()
        if not os.path.exists(plotter.results_base_dir):
            print("❌ No results directory found. Please run a simulation first.")
            return
        
        # Get available results folders
        result_dirs = [d for d in os.listdir(plotter.results_base_dir) 
                      if os.path.isdir(os.path.join(plotter.results_base_dir, d)) and not d.startswith('.')]
        
        if not result_dirs:
            print("❌ No simulation results found. Please run a simulation first.")
            return
        
        # Show available folders
        print(f"\n📁 Available results folders ({len(result_dirs)} found):")
        for i, folder in enumerate(result_dirs, 1):
            print(f"  {i}. {folder}")
        
        # Get folder selection
        print("\n🔍 Select a results folder:")
        print("  - Enter the number (1, 2, 3, etc.)")
        print("  - Or enter part of the folder name (e.g., 'cpr', 'centralized')")
        print("  - Or press Enter to use the most recent folder")
        
        folder_choice = input("\nYour choice: ").strip()
        
        # Determine selected folder
        selected_folder = None
        if not folder_choice:  # Use most recent
            selected_folder = sorted(result_dirs)[-1]
            print(f"📂 Using most recent folder: {selected_folder}")
        elif folder_choice.isdigit():  # Number selection
            idx = int(folder_choice) - 1
            if 0 <= idx < len(result_dirs):
                selected_folder = result_dirs[idx]
                print(f"📂 Selected folder: {selected_folder}")
            else:
                print("❌ Invalid number selection.")
                return
        else:  # Partial name match
            matches = [d for d in result_dirs if folder_choice.lower() in d.lower()]
            if len(matches) == 1:
                selected_folder = matches[0]
                print(f"📂 Found matching folder: {selected_folder}")
            elif len(matches) > 1:
                print(f"🔍 Multiple matches found: {matches}")
                print("Please be more specific.")
                return
            else:
                print(f"❌ No folders found matching '{folder_choice}'")
                return
        
        # Show available plot types
        plot_types = [
            ('farmer_budgets', 'Farmer Budget Returns (with confidence intervals)'),
            ('water_inflows', 'Water Inflows (annual and July)'),
            ('fish_population', 'Fish Population (adults and larvae)'),
            ('yield_distribution', 'Yield Distribution (box plot)'),
            ('budget_inequality', 'Budget Inequality (Theil Index)'),
            ('activity_breakdown', 'Activity Breakdown (stacked bar chart)'),
            ('summary_stats', 'Summary Statistics (text summary)')
        ]
        
        print(f"\n📊 Available plot types:")
        for i, (plot_key, plot_desc) in enumerate(plot_types, 1):
            print(f"  {i}. {plot_key} - {plot_desc}")
        print(f"  {len(plot_types) + 1}. all - Extract all plots")
        
        # Get plot selection
        print("\n🎨 Select plot type(s) to extract:")
        print("  - Enter the number (1, 2, 3, etc.)")
        print("  - Or enter the plot name (e.g., 'farmer_budgets')")
        print(f"  - Or enter 'all' to extract all {len(plot_types)} plots")
        
        plot_choice = input("\nYour choice: ").strip()
        
        # Get format preference
        print("\n💾 Choose output format:")
        print("  1. PDF (high quality, recommended)")
        print("  2. PNG (web-friendly)")
        
        format_choice = input("Your choice (1 for PDF, 2 for PNG, or press Enter for PDF): ").strip()
        save_as_pdf = format_choice != '2'
        format_name = "PDF" if save_as_pdf else "PNG"
        
        # Process plot selection
        selected_plots = []
        
        if plot_choice.lower() == 'all' or plot_choice == str(len(plot_types) + 1):
            selected_plots = [plot_key for plot_key, _ in plot_types]
            print(f"📈 Will extract all {len(selected_plots)} plots as {format_name}")
        elif plot_choice.isdigit():
            idx = int(plot_choice) - 1
            if 0 <= idx < len(plot_types):
                selected_plots = [plot_types[idx][0]]
                print(f"📈 Will extract '{plot_types[idx][0]}' as {format_name}")
            else:
                print("❌ Invalid number selection.")
                return
        else:
            # Try to match plot name
            matches = [(k, d) for k, d in plot_types if plot_choice.lower() in k.lower()]
            if len(matches) == 1:
                selected_plots = [matches[0][0]]
                print(f"📈 Will extract '{matches[0][0]}' as {format_name}")
            elif len(matches) > 1:
                print(f"🔍 Multiple plot types match '{plot_choice}': {[m[0] for m in matches]}")
                print("Please be more specific.")
                return
            else:
                print(f"❌ No plot type found matching '{plot_choice}'")
                return
        
        # Extract the plots
        print(f"\n🚀 Starting extraction...")
        print("-" * 30)
        
        success_count = 0
        for plot_name in selected_plots:
            try:
                print(f"📊 Extracting {plot_name}...", end=" ")
                extract_plot(selected_folder, plot_name, save_as_pdf=save_as_pdf)
                print("✅")
                success_count += 1
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Summary
        print("-" * 30)
        if success_count > 0:
            print(f"🎉 Successfully extracted {success_count}/{len(selected_plots)} plots!")
            results_path = os.path.join(plotter.results_base_dir, selected_folder)
            print(f"📁 Files saved to: {results_path}")
            
            # List the created files
            file_pattern = "*.pdf" if save_as_pdf else "*.png"
            import glob
            created_files = glob.glob(os.path.join(results_path, file_pattern))
            recent_files = [f for f in created_files if any(plot in os.path.basename(f) for plot in selected_plots)]
            
            if recent_files:
                print(f"📄 Created files:")
                for file_path in recent_files:
                    file_name = os.path.basename(file_path)
                    file_size = os.path.getsize(file_path)
                    print(f"   - {file_name} ({file_size:,} bytes)")
        else:
            print("❌ No plots were successfully extracted.")
    
    def show_help():
        """Show help information"""
        print("CSV Plotter Utility - Help")
        print("=" * 30)
        print()
        print("This utility allows you to extract individual plots from")
        print("the comprehensive simulation dashboard as separate files.")
        print()
        print("Available plot types:")
        list_available_plots()
        print()
        print("Usage modes:")
        print("1. Interactive mode: python csv_plots.py")
        print("2. Command line: from csv_plots import extract_plot")
        print("                 extract_plot('folder_name', 'plot_name')")
    
    # Main entry point
    print("CSV Plotter Utility")
    print("=" * 20)
    print("1. Interactive plot extractor")
    print("2. Show help")
    print("3. List available plots")
    print("4. Quick test (extract farmer_budgets from most recent folder)")
    
    choice = input("\nEnter your choice (1, 2, 3, 4, or press Enter for interactive mode): ").strip()
    
    if choice == '2':
        show_help()
    elif choice == '3':
        list_available_plots()
    elif choice == '4':
        # Quick test option
        print("\n🚀 Quick Test - Extracting farmer_budgets from most recent folder...")
        plotter = CSVPlotter()
        if os.path.exists(plotter.results_base_dir):
            result_dirs = [d for d in os.listdir(plotter.results_base_dir) 
                          if os.path.isdir(os.path.join(plotter.results_base_dir, d)) and not d.startswith('.')]
            if result_dirs:
                latest_folder = sorted(result_dirs)[-1]
                try:
                    extract_plot(latest_folder, "farmer_budgets", save_as_pdf=True)
                    print("✅ Quick test completed successfully!")
                except Exception as e:
                    print(f"❌ Quick test failed: {e}")
            else:
                print("❌ No results folders found.")
        else:
            print("❌ No results directory found.")
    else:  # Default to interactive mode
        interactive_plot_extractor()