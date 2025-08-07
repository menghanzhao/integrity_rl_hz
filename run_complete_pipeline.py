#!/usr/bin/env python3
"""
Complete pipeline for LDP Maintenance Reinforcement Learning
Runs training, simulation, and visualization in sequence
"""

import os
import sys
from datetime import datetime
import traceback

def main():
    """Run the complete RL pipeline"""
    
    print("="*80)
    print("LDP MAINTENANCE REINFORCEMENT LEARNING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config_file = "RL work scope.xlsx"
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"ERROR: Configuration file '{config_file}' not found!")
        print("Please ensure the Excel file is in the current directory.")
        return False
    
    try:
        # Step 1: Train the agent
        print("\nStep 1: Training the RL agent...")
        print("-" * 40)
        
        from train_agent import train_agent
        agent, episode_rewards = train_agent(config_file, n_episodes=100)
        
        print(f"Training completed. Final average reward: {sum(episode_rewards[-100:]) / 100:.2f}")
        
        # Step 2: Run 5-year simulation
        print("\nStep 2: Running 5-year simulation...")
        print("-" * 40)
        
        from simulate_and_visualize import simulate_5_years, create_visualizations
        
        agent_path = "trained_agent.pkl"
        results_df, simulation_data = simulate_5_years(agent_path, config_file)
        
        # Step 3: Create visualizations
        print("\nStep 3: Creating visualizations...")
        print("-" * 40)
        
        create_visualizations(results_df, simulation_data)
        
        print("\nPipeline completed successfully!")
        print("\nGenerated files:")
        print("- trained_agent.pkl (trained RL agent)")
        print("- training_progress.png (training plots)")
        print("- simulation_results.csv (simulation data)")
        print("- simulation_data.pkl (detailed simulation data)")
        print("- 5_year_simulation_results.png (comprehensive results)")
        print("- thickness_analysis.png (detailed thickness analysis)")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required packages are installed:")
        print("pip install numpy pandas matplotlib seaborn openpyxl")
        return False
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Please check that all required files are present.")
        return False
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False
    
    finally:
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)