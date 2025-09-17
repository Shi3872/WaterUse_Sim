"""
Configuration loader for Water Use Simulation
Reads parameters from config.yaml and provides them to the simulation
"""

import yaml
import os
from typing import Dict, Any

class SimulationConfig:
    """Class to load and manage simulation configuration"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to the configuration YAML file. If None, looks for config.yaml 
                        in the same directory as this script.
        """
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")
        
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        return config
    
    def get_simulation_params(self, scenario: str = "default") -> Dict[str, Any]:
        """
        Get simulation parameters for a specific scenario
        
        Args:
            scenario: Name of the scenario to load from config
            
        Returns:
            Dictionary containing all simulation parameters
        """
        # Start with base parameters
        params = {
            # Simulation setup
            'years': self.config['simulation']['years'],
            'print_interval': self.config['simulation']['print_interval'],
            'num_farmers': self.config['simulation']['num_farmers'],
            
            # System configuration
            'centralized': self.config['system']['centralized'],
            'fishing_enabled': self.config['system']['fishing_enabled'],
            
            # Agent behavior
            'memory_strength': self.config['agents']['memory_strength'],
            
            # Game theory
            'use_cpr_game': self.config['games']['use_cpr_game'],
            'use_static_game': self.config['games']['use_static_game'],
            
            # LLM integration
            'generative_agent': self.config['llm']['generative_agent'],
            'llm_provider': self.config['llm']['provider'],
            
            # Water inflow
            'inflow_case': self.config['water']['inflow_case'],
            
            # Output configuration
            'save_csv': self.config.get('output', {}).get('save_csv', True),
            'results_dir': self.config.get('output', {}).get('results_dir', 'results'),
        }
        
        # Override with scenario-specific parameters if specified
        if scenario != "default" and scenario in self.config['scenarios']:
            scenario_params = self.config['scenarios'][scenario]
            params.update(scenario_params)
        
        return params
    
    def get_economic_params(self) -> Dict[str, float]:
        """Get economic parameters"""
        return self.config['economy']
    
    def get_field_params(self) -> Dict[str, Any]:
        """Get field constraint parameters"""
        return self.config['fields']
    
    def get_water_params(self) -> Dict[str, Any]:
        """Get water-related parameters"""
        return self.config['water']
    
    def get_fish_params(self) -> Dict[str, Any]:
        """Get fish population parameters"""
        return self.config['fish']
    
    def list_scenarios(self) -> list:
        """List all available scenarios"""
        return list(self.config['scenarios'].keys())
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        Get output and CSV export configuration
        
        Returns:
            Dictionary containing output configuration parameters
        """
        return self.config.get('output', {
            'save_csv': True,
            'results_dir': 'results',
            'include_metadata': True,
            'export_components': {
                'farmer_data': True,
                'fish_data': True,
                'water_data': True,
                'summary': True
            }
        })
    
    def update_config(self, section: str, key: str, value: Any):
        """
        Update a configuration value and save to file
        
        Args:
            section: Configuration section (e.g., 'simulation', 'system')
            key: Parameter key
            value: New value
        """
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            self._save_config()
        else:
            raise KeyError(f"Configuration key not found: {section}.{key}")
    
    def _save_config(self):
        """Save current configuration back to YAML file"""
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)

# Convenience function to load configuration
def load_config(config_path: str = "config.yaml") -> SimulationConfig:
    """Load simulation configuration from file"""
    return SimulationConfig(config_path)