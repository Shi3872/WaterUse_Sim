#!/usr/bin/env python3
"""
Example script demonstrating how to use the configuration system
for Water Use Simulation experiments
"""

from config_loader import load_config
from main import run_simulation_from_config

def main():
    """Run different simulation scenarios using configuration"""
    
    # Load configuration
    config = load_config()
    
    print("=== Water Use Simulation Configuration Demo ===")
    print(f"Available scenarios: {config.list_scenarios()}")
    
    # Example 1: Run default scenario
    print("\n1. Running default scenario...")
    results_default = run_simulation_from_config("default")
    print(f"Default results shape: {results_default[0].shape}")
    
    # Example 2: Run centralized fishing scenario
    print("\n2. Running centralized fishing scenario...")
    results_centralized = run_simulation_from_config("centralized_fishing")
    print(f"Centralized fishing results shape: {results_centralized[0].shape}")
    
    # Example 3: Test LLM scenarios (comment out if no API keys)
    # print("\n3. Running LLM scenario (Together AI)...")
    # results_llm_together = run_simulation_from_config("llm_together")
    # print(f"LLM Together results shape: {results_llm_together[0].shape}")
    
    # Example 4: Update configuration programmatically
    print("\n4. Updating configuration...")
    config.update_config("simulation", "years", 10)
    print("Updated simulation years to 10")
    
    # Example 5: Run with updated configuration
    print("\n5. Running with updated config...")
    results_updated = run_simulation_from_config("default")
    print(f"Updated results shape: {results_updated[0].shape}")
    
    print("\n=== Configuration Demo Complete ===")

if __name__ == "__main__":
    main()