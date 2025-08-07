import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
from LDPMaintenanceEnv import LDPMaintenanceEnv
from train_agent import QLearningAgent

def simulate_5_years(agent_path, config_file, save_results=True):
    """Run a 5-year simulation with the trained agent"""
    
    # Load trained agent
    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)
    
    # Set agent to exploitation mode (no exploration)
    agent.epsilon = 0.0
    
    # Initialize environment
    env = LDPMaintenanceEnv(config_file)
    
    # Tracking variables
    simulation_data = {
        'dates': [],
        'daily_rewards': [],
        'cumulative_rewards': [],
        'failures': [],
        'actions_taken': [],
        'thickness_history': [],
        'valid_action_days': []
    }
    
    state = env.reset()
    cumulative_reward = 0
    
    print("Running 5-year simulation...")
    
    step_count = 0
    while True:
        # Get valid spools for today
        valid_spools = env.get_valid_actions_for_day()
        
        # Choose actions
        actions = agent.choose_action(state, valid_spools)
        
        # Take step
        next_state, reward, done, info = env.step(actions)
        
        # Record data
        simulation_data['dates'].append(env.simulation_date - timedelta(days=1))
        simulation_data['daily_rewards'].append(reward)
        cumulative_reward += reward
        simulation_data['cumulative_rewards'].append(cumulative_reward)
        simulation_data['failures'].append(len(env.failures))
        simulation_data['actions_taken'].append(sum(1 for a in actions.values() if a != 0))
        simulation_data['thickness_history'].append(env.thickness_array.copy())
        simulation_data['valid_action_days'].append(len(valid_spools) > 0)
        
        state = next_state
        step_count += 1
        
        if step_count % 365 == 0:
            print(f"Year {step_count // 365} completed. Cumulative reward: {cumulative_reward:,.0f}")
        
        if done:
            break
    
    print(f"Simulation completed. Final cumulative reward: {cumulative_reward:,.0f}")
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame({
        'date': simulation_data['dates'],
        'daily_reward': simulation_data['daily_rewards'],
        'cumulative_reward': simulation_data['cumulative_rewards'],
        'failures': simulation_data['failures'],
        'actions_taken': simulation_data['actions_taken'],
        'valid_action_day': simulation_data['valid_action_days']
    })
    
    if save_results:
        results_df.to_csv('simulation_results.csv', index=False)
        with open('simulation_data.pkl', 'wb') as f:
            pickle.dump(simulation_data, f)
    
    return results_df, simulation_data

def create_visualizations(results_df, simulation_data):
    """Create comprehensive visualizations of the simulation results"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Cumulative Reward Over Time
    plt.subplot(3, 3, 1)
    plt.plot(results_df['date'], results_df['cumulative_reward'], linewidth=2)
    plt.title('Cumulative Reward Over 5 Years', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Reward')
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='plain', axis='y')
    
    # 2. Daily Rewards Distribution
    plt.subplot(3, 3, 2)
    plt.hist(results_df['daily_reward'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Daily Rewards', fontsize=14, fontweight='bold')
    plt.xlabel('Daily Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 3. Failures Over Time
    plt.subplot(3, 3, 3)
    plt.plot(results_df['date'], results_df['failures'], color='red', linewidth=2)
    plt.title('Daily Failure Count', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Number of Failures')
    plt.grid(True, alpha=0.3)
    
    # 4. Actions Taken Over Time
    plt.subplot(3, 3, 4)
    plt.plot(results_df['date'], results_df['actions_taken'], color='green', linewidth=2)
    plt.title('Actions Taken Per Day', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Number of Actions')
    plt.grid(True, alpha=0.3)
    
    # 5. Monthly Reward Analysis
    plt.subplot(3, 3, 5)
    results_df['month'] = results_df['date'].dt.month
    monthly_rewards = results_df.groupby('month')['daily_reward'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(range(1, 13), monthly_rewards.values, color='skyblue', edgecolor='black')
    plt.title('Average Daily Reward by Month', fontsize=14, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Average Daily Reward')
    plt.xticks(range(1, 13), month_names)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 6. Valid Action Days
    plt.subplot(3, 3, 6)
    valid_days = results_df['valid_action_day'].sum()
    total_days = len(results_df)
    plt.pie([valid_days, total_days - valid_days], 
            labels=[f'Valid Action Days ({valid_days})', f'No Action Days ({total_days - valid_days})'],
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    plt.title('Action Opportunity Distribution', fontsize=14, fontweight='bold')
    
    # 7. Yearly Performance
    plt.subplot(3, 3, 7)
    results_df['year'] = results_df['date'].dt.year
    yearly_rewards = results_df.groupby('year')['daily_reward'].sum()
    plt.bar(yearly_rewards.index, yearly_rewards.values, color='orange', edgecolor='black')
    plt.title('Total Reward by Year', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 8. Thickness Evolution (Average across all spools)
    plt.subplot(3, 3, 8)
    avg_thickness_over_time = []
    for thickness_array in simulation_data['thickness_history'][::30]:  # Sample every 30 days
        # thickness_array is now a numpy array [spool_idx, quadrant]
        avg_thickness = thickness_array.mean()
        avg_thickness_over_time.append(avg_thickness)
    
    sample_dates = results_df['date'][::30][:len(avg_thickness_over_time)]
    plt.plot(sample_dates, avg_thickness_over_time, color='purple', linewidth=2)
    plt.title('Average Thickness Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Average Thickness (mm)')
    plt.grid(True, alpha=0.3)
    
    # 9. Action Type Analysis
    plt.subplot(3, 3, 9)
    # This would need to be implemented based on action tracking
    action_types = ['Do Nothing', 'Rotate', 'Repair', 'Replace']
    action_counts = [0, 0, 0, 0]  # Placeholder - would need actual tracking
    plt.bar(action_types, action_counts, color=['gray', 'blue', 'yellow', 'red'], edgecolor='black')
    plt.title('Action Type Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Action Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('5_year_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional detailed plots
    create_detailed_analysis(results_df, simulation_data)

def create_detailed_analysis(results_df, simulation_data):
    """Create additional detailed analysis plots"""
    
    # Detailed thickness analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Track thickness for each quadrant over time
    quadrant_names = ['Quadrant 1 (Top)', 'Quadrant 2 (Right)', 'Quadrant 3 (Bottom)', 'Quadrant 4 (Left)']
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (quadrant, color) in enumerate(zip(['Quadrant 1', 'Quadrant 2', 'Quadrant 3', 'Quadrant 4'], colors)):
        ax = axes[i//2, i%2]
        
        # Sample thickness data every 30 days
        thickness_over_time = []
        for thickness_array in simulation_data['thickness_history'][::30]:
            # thickness_array is [spool_idx, quadrant_idx] - get average for this quadrant
            avg_thickness = thickness_array[:, i].mean()  # i is the quadrant index (0-3)
            thickness_over_time.append(avg_thickness)
        
        sample_dates = results_df['date'][::30][:len(thickness_over_time)]
        ax.plot(sample_dates, thickness_over_time, color=color, linewidth=2)
        ax.set_title(f'Average {quadrant_names[i]} Thickness', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Thickness (mm)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('thickness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual spool thickness charts
    create_individual_spool_charts(results_df, simulation_data)
    
def create_individual_spool_charts(results_df, simulation_data):
    """Create detailed thickness charts for each spool across all quadrants"""
    
    # Get environment to access spool information
    from LDPMaintenanceEnv import LDPMaintenanceEnv
    env = LDPMaintenanceEnv("RL work scope.xlsx")
    
    # Sample data every 30 days for better performance
    sample_indices = range(0, len(simulation_data['thickness_history']), 30)
    sample_dates = [results_df['date'].iloc[i] for i in sample_indices if i < len(results_df)]
    
    quadrant_names = ['Top', 'Right', 'Bottom', 'Left']
    colors = ['red', 'blue', 'green', 'orange']
    
    # Create 4 separate figures (one for each quadrant)
    for q_idx, (quad_name, color) in enumerate(zip(quadrant_names, colors)):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{quad_name} Quadrant Thickness Over 5 Years - All Spools', fontsize=16, fontweight='bold')
        
        # Plot each spool's thickness for this quadrant
        for spool_idx, spool in enumerate(env.spools):
            ax = axes[spool_idx // 2, spool_idx % 2]
            
            # Extract thickness data for this spool and quadrant
            thickness_data = []
            for i in sample_indices:
                if i < len(simulation_data['thickness_history']):
                    thickness_array = simulation_data['thickness_history'][i]
                    thickness_data.append(thickness_array[spool_idx, q_idx])
            
            # Truncate dates to match data length
            plot_dates = sample_dates[:len(thickness_data)]
            
            # Plot the thickness evolution
            ax.plot(plot_dates, thickness_data, color=color, linewidth=2, alpha=0.8)
            ax.set_title(f'Spool {spool}', fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Thickness (mm)')
            ax.grid(True, alpha=0.3)
            
            # Add critical threshold line
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Critical (1mm)')
            ax.axhline(y=5.0, color='orange', linestyle='--', alpha=0.5, label='Warning (5mm)')
            
            # Set y-axis limits for better visibility
            ax.set_ylim(0, max(25, max(thickness_data) * 1.1))
            
            # Add legend only to first subplot
            if spool_idx == 0:
                ax.legend()
            
            # Format x-axis dates
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'spool_thickness_{quad_name.lower()}_quadrant.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"\nGenerated individual spool thickness charts:")
    for quad_name in quadrant_names:
        print(f"- spool_thickness_{quad_name.lower()}_quadrant.png")

    # Performance summary
    print("\n" + "="*80)
    print("5-YEAR SIMULATION PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total Simulation Days: {len(results_df):,}")
    print(f"Final Cumulative Reward: {results_df['cumulative_reward'].iloc[-1]:,.0f}")
    print(f"Average Daily Reward: {results_df['daily_reward'].mean():.2f}")
    print(f"Total Days with Failures: {(results_df['failures'] > 0).sum():,}")
    print(f"Total Actions Taken: {results_df['actions_taken'].sum():,}")
    print(f"Days with Valid Actions: {results_df['valid_action_day'].sum():,} ({100 * results_df['valid_action_day'].mean():.1f}%)")
    print(f"Average Failures per Day: {results_df['failures'].mean():.3f}")
    print(f"Maximum Daily Failures: {results_df['failures'].max()}")
    print(f"Worst Single Day Reward: {results_df['daily_reward'].min():,.0f}")
    print(f"Best Single Day Reward: {results_df['daily_reward'].max():,.0f}")
    print("="*80)

if __name__ == "__main__":
    # Run simulation and create visualizations
    config_file = "RL work scope.xlsx"
    agent_path = "trained_agent.pkl"
    
    try:
        results_df, simulation_data = simulate_5_years(agent_path, config_file)
        create_visualizations(results_df, simulation_data)
    except FileNotFoundError:
        print("Trained agent not found. Please run train_agent.py first.")
    except Exception as e:
        print(f"Error during simulation: {e}")
        print("Please ensure the config file exists and the environment is properly set up.")