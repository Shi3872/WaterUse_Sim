"""
CSV Export Utilities for Water Use Simulation
Handles saving simulation results to CSV files for analysis
"""

import csv
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

class SimulationCSVExporter:
    """Class to export simulation results to CSV files"""
    
    def __init__(self, base_results_dir: str = "results"):
        """
        Initialize CSV exporter
        
        Args:
            base_results_dir: Base directory for saving results (relative to project root)
        """
        # Get the directory where this script is located (project root)
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.base_results_dir = os.path.join(project_root, base_results_dir)
        self.session_dir = None
        
    def create_session_directory(self, scenario_name: str = "default") -> str:
        """
        Create a timestamped directory for this simulation session
        
        Args:
            scenario_name: Name of the scenario being run
            
        Returns:
            Path to the created session directory
        """
        # Ensure base results directory exists
        os.makedirs(self.base_results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.base_results_dir, f"{scenario_name}_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        return self.session_dir
    
    def export_farmer_data(self, simulation, scenario_name: str = "default") -> str:
        """
        Export farmer-level data to CSV
        
        Args:
            simulation: The completed simulation object
            scenario_name: Name of the scenario
            
        Returns:
            Path to the created CSV file
        """
        if not self.session_dir:
            self.create_session_directory(scenario_name)
            
        # Prepare farmer data
        farmer_data = []
        
        # Get the maximum number of years from any farmer's history
        max_years = max(len(f.yield_history) for f in simulation.farmers) if simulation.farmers else 0
        
        for year in range(max_years):
            for farmer_idx, farmer in enumerate(simulation.farmers):
                row = {
                    'scenario': scenario_name,
                    'year': year + 1,
                    'farmer_id': farmer_idx + 1,
                    'location': farmer.location,
                    'irrigated_fields': farmer.irrigated_fields,
                    'yield': farmer.yield_history[year] if year < len(farmer.yield_history) else 0,
                    'catch': farmer.catch_history[year] if year < len(farmer.catch_history) else 0,
                    'budget': simulation.farmer_budget_history[farmer_idx][year] if year < len(simulation.farmer_budget_history[farmer_idx]) else 0,
                    'min_income': farmer.min_income
                }
                farmer_data.append(row)
        
        # Save to CSV
        farmer_csv_path = os.path.join(self.session_dir, "farmer_data.csv")
        with open(farmer_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['scenario', 'year', 'farmer_id', 'location', 'irrigated_fields', 
                         'yield', 'catch', 'budget', 'min_income']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(farmer_data)
            
        return farmer_csv_path
    
    def export_fish_data(self, simulation, scenario_name: str = "default") -> str:
        """
        Export fish population data to CSV
        
        Args:
            simulation: The completed simulation object
            scenario_name: Name of the scenario
            
        Returns:
            Path to the created CSV file
        """
        if not self.session_dir:
            self.create_session_directory(scenario_name)
            
        fish_data = []
        
        for year, age_classes in enumerate(simulation.fish_history):
            row = {
                'scenario': scenario_name,
                'year': year + 1,
                'age_class_0': age_classes[0] if len(age_classes) > 0 else 0,
                'age_class_1': age_classes[1] if len(age_classes) > 1 else 0,
                'age_class_2': age_classes[2] if len(age_classes) > 2 else 0,
                'age_class_3': age_classes[3] if len(age_classes) > 3 else 0,
                'age_class_4': age_classes[4] if len(age_classes) > 4 else 0,
                'age_class_5_plus': sum(age_classes[5:]) if len(age_classes) > 5 else 0,
                'total_fish': sum(age_classes),
                'adult_fish': sum(age_classes[5:]) if len(age_classes) > 5 else 0,
                'larvae': age_classes[0] if len(age_classes) > 0 else 0
            }
            fish_data.append(row)
        
        # Save to CSV
        fish_csv_path = os.path.join(self.session_dir, "fish_data.csv")
        with open(fish_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['scenario', 'year', 'age_class_0', 'age_class_1', 'age_class_2', 
                         'age_class_3', 'age_class_4', 'age_class_5_plus', 'total_fish', 
                         'adult_fish', 'larvae']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fish_data)
            
        return fish_csv_path
    
    def export_water_data(self, simulation, scenario_name: str = "default") -> str:
        """
        Export water resource data to CSV
        
        Args:
            simulation: The completed simulation object
            scenario_name: Name of the scenario
            
        Returns:
            Path to the created CSV file
        """
        if not self.session_dir:
            self.create_session_directory(scenario_name)
            
        water_data = []
        
        # Get inflow data from the water resource
        inflows = simulation.water.inflow_series if hasattr(simulation.water, 'inflow_series') else []
        july_inflows = simulation.july_inflows if hasattr(simulation, 'july_inflows') else []
        
        # Handle numpy arrays
        if hasattr(inflows, '__len__') and len(inflows) > 0:
            inflow_len = len(inflows)
        else:
            inflow_len = 0
            
        if hasattr(july_inflows, '__len__') and len(july_inflows) > 0:
            july_len = len(july_inflows)
        else:
            july_len = 0
        
        max_years = max(inflow_len, july_len) if inflow_len > 0 or july_len > 0 else 0
        
        for year in range(max_years):
            row = {
                'scenario': scenario_name,
                'year': year + 1,
                'annual_inflow': inflows[year] if year < len(inflows) else 0,
                'july_inflow': july_inflows[year] if year < len(july_inflows) else 0
            }
            water_data.append(row)
        
        # Save to CSV
        water_csv_path = os.path.join(self.session_dir, "water_data.csv")
        with open(water_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['scenario', 'year', 'annual_inflow', 'july_inflow']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(water_data)
            
        return water_csv_path
    
    def export_summary_data(self, simulation, scenario_name: str = "default", config_params: Optional[Dict] = None) -> str:
        """
        Export summary statistics to CSV
        
        Args:
            simulation: The completed simulation object
            scenario_name: Name of the scenario
            config_params: Configuration parameters used for this run
            
        Returns:
            Path to the created CSV file
        """
        if not self.session_dir:
            self.create_session_directory(scenario_name)
            
        # Calculate summary statistics
        total_yield = np.sum([np.sum(f.yield_history) for f in simulation.farmers])
        avg_yield_per_farmer = total_yield / len(simulation.farmers) if simulation.farmers else 0
        avg_yield_per_year = total_yield / len(simulation.farmers[0].yield_history) if simulation.farmers and simulation.farmers[0].yield_history else 0
        
        total_catch = np.sum([np.sum(f.catch_history) for f in simulation.farmers])
        avg_catch_per_farmer = total_catch / len(simulation.farmers) if simulation.farmers else 0
        
        final_fish_total = simulation.annual_fish_totals[-1] if simulation.annual_fish_totals else 0
        avg_fish_population = np.mean(simulation.annual_fish_totals) if simulation.annual_fish_totals else 0
        
        final_budgets = [simulation.farmer_budget_history[i][-1] if simulation.farmer_budget_history[i] else 0 
                        for i in range(len(simulation.farmers))]
        avg_final_budget = np.mean(final_budgets) if final_budgets else 0
        
        summary_data = [{
            'scenario': scenario_name,
            'timestamp': datetime.now().isoformat(),
            'num_farmers': len(simulation.farmers),
            'years_simulated': len(simulation.farmers[0].yield_history) if simulation.farmers and simulation.farmers[0].yield_history else 0,
            'centralized': getattr(simulation, 'centralized', False),
            'fishing_enabled': getattr(simulation, 'fishing_enabled', False),
            'total_yield': total_yield,
            'avg_yield_per_farmer': avg_yield_per_farmer,
            'avg_yield_per_year': avg_yield_per_year,
            'total_catch': total_catch,
            'avg_catch_per_farmer': avg_catch_per_farmer,
            'final_fish_total': final_fish_total,
            'avg_fish_population': avg_fish_population,
            'avg_final_budget': avg_final_budget,
            'config_params': str(config_params) if config_params else ""
        }]
        
        # Save to CSV
        summary_csv_path = os.path.join(self.session_dir, "summary.csv")
        with open(summary_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['scenario', 'timestamp', 'num_farmers', 'years_simulated', 
                         'centralized', 'fishing_enabled', 'total_yield', 'avg_yield_per_farmer',
                         'avg_yield_per_year', 'total_catch', 'avg_catch_per_farmer',
                         'final_fish_total', 'avg_fish_population', 'avg_final_budget', 'config_params']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
            
        return summary_csv_path
    
    def export_all_data(self, simulation, scenario_name: str = "default", config_params: Optional[Dict] = None) -> Dict[str, str]:
        """
        Export all simulation data to CSV files
        
        Args:
            simulation: The completed simulation object
            scenario_name: Name of the scenario
            config_params: Configuration parameters used for this run
            
        Returns:
            Dictionary mapping data type to CSV file path
        """
        if not self.session_dir:
            self.create_session_directory(scenario_name)
            
        results = {
            'farmer_data': self.export_farmer_data(simulation, scenario_name),
            'fish_data': self.export_fish_data(simulation, scenario_name),
            'water_data': self.export_water_data(simulation, scenario_name),
            'summary': self.export_summary_data(simulation, scenario_name, config_params)
        }
        
        # Create a metadata file
        metadata_path = os.path.join(self.session_dir, "metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Simulation Export Metadata\n")
            f.write(f"========================\n")
            f.write(f"Scenario: {scenario_name}\n")
            f.write(f"Export Time: {datetime.now().isoformat()}\n")
            f.write(f"Session Directory: {self.session_dir}\n")
            f.write(f"Files Created:\n")
            for data_type, file_path in results.items():
                f.write(f"  - {data_type}: {os.path.basename(file_path)}\n")
            if config_params:
                f.write(f"\nConfiguration Parameters:\n")
                for key, value in config_params.items():
                    f.write(f"  {key}: {value}\n")
        
        results['metadata'] = metadata_path
        return results