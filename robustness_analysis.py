"""
Robustness Analysis Module for Water Use Simulation
Generates data tables for analyzing model performance across different configurations
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Optional


def prompt_user_for_model_string() -> str:
    """
    Prompt the user to enter a model string for analysis.
    
    Returns:
        str: The model string entered by the user
    """
    print("Available model types in results directory:")
    results_dir = "results"
    
    # Get all subdirectories to show available model types
    if os.path.exists(results_dir):
        subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        model_types = set()
        
        for subdir in subdirs:
            if '_' in subdir:
                model_type = subdir.split('_')[0]
                model_types.add(model_type)
        
        print("Available model types:")
        for model_type in sorted(model_types):
            print(f"  - {model_type}")
    
    model_string = input("\nEnter the model string for analysis (e.g., 'generative_agent', 'llm-egt'): ").strip()
    return model_string


def find_matching_directories(model_string: str, results_dir: str = "results") -> List[str]:
    """
    Find all subdirectories that match the pattern {model_string}_*
    
    Args:
        model_string: The model string to search for
        results_dir: The results directory path
        
    Returns:
        List of matching directory names
    """
    pattern = f"{model_string}_*"
    matching_dirs = []
    
    if os.path.exists(results_dir):
        all_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        for dir_name in all_dirs:
            if dir_name.startswith(model_string + "_"):
                matching_dirs.append(dir_name)
    
    return sorted(matching_dirs)


def parse_directory_name(dir_name: str, model_string: str) -> Tuple[str, str]:
    """
    Parse directory name to extract LLM model and behavioral induction factor.
    
    Args:
        dir_name: Directory name like "generative_agent_ds_altruistic"
        model_string: The model string like "generative_agent"
        
    Returns:
        Tuple of (llm_model, behavioral_induction_factor)
    """
    # Remove the model_string prefix and the first underscore
    remainder = dir_name[len(model_string) + 1:]
    
    # Split by underscore and assume last part is behavioral factor
    parts = remainder.split('_')
    if len(parts) >= 2:
        llm_model = '_'.join(parts[:-1])
        behavioral_factor = parts[-1]
    else:
        llm_model = parts[0] if parts else "unknown"
        behavioral_factor = "unknown"
    
    return llm_model, behavioral_factor


def calculate_budget_stats_year_100(farmer_data_path: str) -> Tuple[float, float, float, float]:
    """
    Calculate mean, standard deviation, min, and max of budget across all farmers for year 100.
    
    Args:
        farmer_data_path: Path to farmer_data.csv file
        
    Returns:
        Tuple of (mean_budget, std_budget, min_budget, max_budget) for year 100
    """
    try:
        df = pd.read_csv(farmer_data_path)
        year_100_data = df[df['year'] == 100]
        
        if len(year_100_data) == 0:
            print(f"Warning: No data found for year 100 in {farmer_data_path}")
            return np.nan, np.nan, np.nan, np.nan
        
        budgets = year_100_data['budget']
        mean_budget = budgets.mean()
        std_budget = budgets.std()
        min_budget = budgets.min()
        max_budget = budgets.max()
        
        return mean_budget, std_budget, min_budget, max_budget
    
    except Exception as e:
        print(f"Error processing {farmer_data_path}: {e}")
        return np.nan, np.nan, np.nan, np.nan


def calculate_activity_breakdown_percentage(farmer_data_path: str) -> Dict[str, float]:
    """
    Calculate aggregate percentage of activity breakdown over 100 years.
    Based on the logic from csv_plots.py _plot_activity_breakdown function.
    
    Args:
        farmer_data_path: Path to farmer_data.csv file
        
    Returns:
        Dictionary with activity percentages
    """
    try:
        df = pd.read_csv(farmer_data_path)
        
        # Initialize counters for all activities across all years
        total_farmer_years = 0
        both_activities_count = 0
        irrigation_only_count = 0
        fishing_only_count = 0
        no_activity_count = 0
        
        # Process each year
        years = sorted(df['year'].unique())
        for year in years:
            year_data = df[df['year'] == year]
            year_farmers = len(year_data)
            total_farmer_years += year_farmers
            
            if year_farmers > 0:
                # Count activities based on irrigated_fields and catch
                both_count = len(year_data[(year_data['irrigated_fields'] > 0) & (year_data['catch'] > 0)])
                irrigation_count = len(year_data[(year_data['irrigated_fields'] > 0) & (year_data['catch'] == 0)])
                fishing_count = len(year_data[(year_data['irrigated_fields'] == 0) & (year_data['catch'] > 0)])
                none_count = len(year_data[(year_data['irrigated_fields'] == 0) & (year_data['catch'] == 0)])
                
                both_activities_count += both_count
                irrigation_only_count += irrigation_count
                fishing_only_count += fishing_count
                no_activity_count += none_count
        
        # Calculate percentages
        if total_farmer_years > 0:
            activity_breakdown = {
                'both_activities': (both_activities_count / total_farmer_years) * 100,
                'irrigation_only': (irrigation_only_count / total_farmer_years) * 100,
                'fishing_only': (fishing_only_count / total_farmer_years) * 100,
                'no_activity': (no_activity_count / total_farmer_years) * 100
            }
        else:
            activity_breakdown = {
                'both_activities': 0.0,
                'irrigation_only': 0.0,
                'fishing_only': 0.0,
                'no_activity': 0.0
            }
        
        return activity_breakdown
    
    except Exception as e:
        print(f"Error processing {farmer_data_path}: {e}")
        return {
            'both_activities': np.nan,
            'irrigation_only': np.nan,
            'fishing_only': np.nan,
            'no_activity': np.nan
        }


def analyze_robustness(model_string: str, results_dir: str = "results") -> pd.DataFrame:
    """
    Main function to analyze robustness across different model configurations.
    
    Args:
        model_string: The model string to analyze
        results_dir: The results directory path
        
    Returns:
        DataFrame with robustness analysis results
    """
    matching_dirs = find_matching_directories(model_string, results_dir)
    
    if not matching_dirs:
        print(f"No directories found matching pattern '{model_string}_*'")
        return pd.DataFrame()
    
    print(f"Found {len(matching_dirs)} matching directories for model '{model_string}'")
    
    results = []
    
    for dir_name in matching_dirs:
        print(f"Processing directory: {dir_name}")
        
        # Parse directory name
        llm_model, behavioral_factor = parse_directory_name(dir_name, model_string)
        
        # Construct path to farmer_data.csv
        farmer_data_path = os.path.join(results_dir, dir_name, "farmer_data.csv")
        
        if not os.path.exists(farmer_data_path):
            print(f"  Warning: farmer_data.csv not found in {dir_name}")
            continue
        
        # Calculate budget statistics for year 100
        mean_budget, std_budget, min_budget, max_budget = calculate_budget_stats_year_100(farmer_data_path)
        
        # Calculate activity breakdown percentages
        activity_breakdown = calculate_activity_breakdown_percentage(farmer_data_path)
        
        # Store results
        result = {
            'model_string': model_string,
            'llm_model': llm_model,
            'behavioral_factor': behavioral_factor,
            'directory_name': dir_name,
            'mean_budget_year_100': mean_budget,
            'std_budget_year_100': std_budget,
            'min_budget_year_100': min_budget,
            'max_budget_year_100': max_budget,
            'both_activities_pct': activity_breakdown['both_activities'],
            'irrigation_only_pct': activity_breakdown['irrigation_only'],
            'fishing_only_pct': activity_breakdown['fishing_only'],
            'no_activity_pct': activity_breakdown['no_activity']
        }
        
        results.append(result)
        print(f"  Completed: mean={mean_budget:.2f}, std={std_budget:.2f}, min={min_budget:.2f}, max={max_budget:.2f}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if not df.empty:
        # Sort by llm_model and behavioral_factor for better readability
        df = df.sort_values(['llm_model', 'behavioral_factor'])
        df = df.reset_index(drop=True)
    
    return df


def save_robustness_results(df: pd.DataFrame, model_string: str, output_dir: str = "results") -> str:
    """
    Save robustness analysis results to CSV file.
    
    Args:
        df: DataFrame with robustness results
        model_string: The model string analyzed
        output_dir: Directory to save the output file
        
    Returns:
        Path to the saved file
    """
    if df.empty:
        print("No data to save")
        return ""
    
    filename = f"robustness_analysis_{model_string}.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"Robustness analysis results saved to: {filepath}")
    
    return filepath


def display_summary_table(df: pd.DataFrame):
    """
    Display a formatted summary table of the robustness analysis results.
    
    Args:
        df: DataFrame with robustness results
    """
    if df.empty:
        print("No data to display")
        return
    
    print("\n" + "="*120)
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print("="*120)
    
    # Display budget statistics
    print("\nBUDGET STATISTICS (Year 100):")
    print("-" * 100)
    budget_summary = df[['llm_model', 'behavioral_factor', 'mean_budget_year_100', 'std_budget_year_100', 'min_budget_year_100', 'max_budget_year_100']].copy()
    budget_summary['mean_budget_year_100'] = budget_summary['mean_budget_year_100'].round(2)
    budget_summary['std_budget_year_100'] = budget_summary['std_budget_year_100'].round(2)
    budget_summary['min_budget_year_100'] = budget_summary['min_budget_year_100'].round(2)
    budget_summary['max_budget_year_100'] = budget_summary['max_budget_year_100'].round(2)
    print(budget_summary.to_string(index=False))
    
    # Display activity breakdown
    print("\n\nACTIVITY BREAKDOWN PERCENTAGES (Aggregate over 100 years):")
    print("-" * 100)
    activity_summary = df[['llm_model', 'behavioral_factor', 'both_activities_pct', 
                          'irrigation_only_pct', 'fishing_only_pct', 'no_activity_pct']].copy()
    
    # Round percentages to 1 decimal place
    pct_cols = ['both_activities_pct', 'irrigation_only_pct', 'fishing_only_pct', 'no_activity_pct']
    for col in pct_cols:
        activity_summary[col] = activity_summary[col].round(1)
    
    print(activity_summary.to_string(index=False))
    
    print("\n" + "="*120)


def run_robustness_analysis():
    """
    Main function to run the complete robustness analysis workflow.
    """
    print("Water Use Simulation - Robustness Analysis")
    print("=" * 50)
    
    # Prompt user for model string
    model_string = prompt_user_for_model_string()
    
    if not model_string:
        print("No model string provided. Exiting...")
        return
    
    print(f"\nAnalyzing robustness for model: '{model_string}'")
    print("-" * 50)
    
    # Run analysis
    results_df = analyze_robustness(model_string)
    
    if results_df.empty:
        print("No results generated. Please check your model string and data availability.")
        return
    
    # Display summary
    display_summary_table(results_df)
    
    # Save results
    output_file = save_robustness_results(results_df, model_string)
    
    print(f"\nAnalysis complete! Results saved to: {output_file}")
    
    return results_df


if __name__ == "__main__":
    # Run the analysis when script is executed directly
    run_robustness_analysis()