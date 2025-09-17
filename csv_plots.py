"""
CSV-based plotting utilities for Water Use Simulation
Reads CSV data from results folder and generates plots like the original implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import Dict, List, Optional, Tuple

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
            'summary': 'summary.csv'
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
        
        fig = plt.figure(figsize=(16, 12))
        
        # Farmer returns (top left)
        plt.subplot(2, 3, 1)
        if 'farmer_data' in data:
            farmer_data = data['farmer_data']
            farmer_9 = farmer_data[farmer_data['farmer_id'] == 9]
            if not farmer_9.empty:
                plt.plot(farmer_9['year'], farmer_9['budget'], 'o-')
                plt.xlabel("Year")
                plt.ylabel("Budget")
                plt.title("Farmer 9 Returns")
                plt.grid(True, alpha=0.3)
        
        # Water inflows (top middle)
        plt.subplot(2, 3, 2)
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
        plt.subplot(2, 3, 3)
        if 'fish_data' in data:
            fish_data = data['fish_data']
            plt.plot(fish_data['year'], fish_data['adult_fish'], 'ko-', label='Adults')
            plt.plot(fish_data['year'], fish_data['larvae'], 'r-s', label='Larvae')
            plt.xlabel("Year")
            plt.ylabel("Fish Count")
            plt.title("Fish Population")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Yield box plot (bottom left)
        plt.subplot(2, 3, 4)
        if 'farmer_data' in data:
            farmer_data = data['farmer_data']
            yield_pivot = farmer_data.pivot(index='year', columns='farmer_id', values='yield')
            box_data = [yield_pivot[col].dropna().values for col in yield_pivot.columns]
            plt.boxplot(box_data, labels=[f'F{col}' for col in yield_pivot.columns])
            plt.xlabel("Farmers")
            plt.ylabel("Yield")
            plt.title("Yield Distribution")
            plt.xticks(rotation=45)
        
        # Budget over time (bottom middle)
        plt.subplot(2, 3, 5)
        if 'farmer_data' in data:
            farmer_data = data['farmer_data']
            # Plot average budget over time
            avg_budget = farmer_data.groupby('year')['budget'].mean()
            plt.plot(avg_budget.index, avg_budget.values, 'g-o')
            plt.xlabel("Year")
            plt.ylabel("Average Budget")
            plt.title("Average Farmer Budget")
            plt.grid(True, alpha=0.3)
        
        # Summary stats (bottom right)
        plt.subplot(2, 3, 6)
        if 'summary' in data:
            summary = data['summary'].iloc[0]
            stats_text = f"""
            Scenario: {summary['scenario']}
            Years: {summary['years_simulated']}
            Farmers: {summary['num_farmers']}
            Centralized: {summary['centralized']}
            Fishing: {summary['fishing_enabled']}
            
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