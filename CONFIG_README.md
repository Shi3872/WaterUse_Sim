# Water Use Simulation Configuration System

This directory contains a comprehensive configuration system for the Water Use Simulation that allows you to easily modify simulation parameters without changing code.

## Files

- `config.yaml` - Main configuration file containing all simulation parameters
- `config_loader.py` - Python module for loading and managing configuration
- `example_config_usage.py` - Example script demonstrating configuration usage
- `main.py` - Updated main simulation script with configuration support

## Configuration File Structure

The `config.yaml` file is organized into sections:

### Simulation Setup
- `years`: Number of years to simulate
- `print_interval`: Print output frequency
- `num_farmers`: Number of farmers in simulation

### Economic Parameters  
- `farmer_initial_budget`: Starting budget for farmers
- `authority_initial_budget`: Starting budget for authority
- `consumption_cost`: Annual consumption cost
- `irrigation_cost`: Cost per irrigated field
- `fish_income_scale`: Fish catch income multiplier

### System Configuration
- `centralized`: Use centralized vs decentralized management
- `fishing_enabled`: Enable/disable fishing activities

### LLM Integration
- `generative_agent`: Use LLM for farmer decisions
- `provider`: LLM provider ("together" or "openai")

### Scenarios
Pre-defined parameter combinations for different experiments:
- `default`: Basic simulation setup
- `centralized_fishing`: Centralized management with fishing
- `llm_together`: LLM-based agents using Together AI
- `llm_openai`: LLM-based agents using OpenAI

## Usage Examples

### 1. Basic Usage

```python
from config_loader import load_config
from main import run_simulation_from_config

# Run default scenario
results = run_simulation_from_config("default")

# Run specific scenario
results = run_simulation_from_config("centralized_fishing")
```

### 2. Listing Available Scenarios

```python
config = load_config()
scenarios = config.list_scenarios()
print("Available scenarios:", scenarios)
```

### 3. Updating Configuration

```python
config = load_config()

# Update a parameter
config.update_config("simulation", "years", 50)

# Update system configuration
config.update_config("system", "centralized", True)
```

### 4. Running with Custom Parameters

Before running, edit `config.yaml`:

```yaml
simulation:
  years: 100                 # Change simulation length
  
system:
  centralized: true          # Switch to centralized management
  fishing_enabled: true      # Enable fishing
  
llm:
  generative_agent: true     # Use LLM agents
  provider: "together"       # Choose LLM provider
```

### 5. Running the Simulation

```bash
# Run with default configuration
python main.py

# Run example scenarios
python example_config_usage.py
```

## Key Benefits

1. **Easy Parameter Changes**: Modify `config.yaml` instead of code
2. **Scenario Management**: Pre-defined scenarios for common experiments
3. **Reproducible Research**: Save configurations for different experiments
4. **No Code Changes**: Change parameters without touching Python files
5. **Parameter Validation**: Configuration loader validates parameters

## Scenario Definitions

### Default Scenario
- Decentralized management
- No fishing
- Heuristic-based farmer decisions
- 20-year simulation

### Centralized Fishing
- Centralized management
- Fishing enabled
- Memory-based water prediction
- 20-year simulation

### LLM Scenarios
- Generative AI-based farmer decisions
- Shorter simulation (5 years for testing)
- Choice of Together AI or OpenAI providers

## Configuration Tips

1. **Start Small**: Use shorter simulations (5-10 years) when testing LLM scenarios
2. **Parameter Validation**: The system validates parameters and provides helpful error messages
3. **Backup Configs**: Save different configuration files for different experiments
4. **Documentation**: Add comments to your config files to remember parameter purposes

## API Keys for LLM Scenarios

To use LLM scenarios, set up your API keys in `.env`:

```
TOGETHER_API_KEY=your_together_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Troubleshooting

- **File Not Found**: Ensure `config.yaml` is in the same directory as your script
- **Invalid Parameters**: Check the error message for parameter validation issues
- **LLM Errors**: Verify API keys are set correctly in `.env` file
- **Import Errors**: Ensure PyYAML is installed: `pip install PyYAML`