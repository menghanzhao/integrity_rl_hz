# LDP Maintenance Reinforcement Learning

This project implements a reinforcement learning solution for optimizing maintenance actions on slurry transport line spools to minimize failures and costs over a 5-year period.

## Problem Description

The system manages two slurry transport lines (Line 1 and Line 2), each with 2 spools. Each spool has measurements in 4 quadrants and is subject to erosion over time. The goal is to optimize maintenance actions while respecting operational constraints.

### Key Constraints:
- Actions can only be taken during scheduled outages:
  - Line 1: Jan 1-3, Apr 1-3, Jul 1-3, Oct 1-3
  - Line 2: Mar 1-3, Jun 1-3, Sep 1-3, Dec 1-3
- Actions can also be taken during unscheduled outages (when failures occur)
- Different seasonal penalties apply when both lines are down

### Actions Available:
0. Do nothing (cost: 0)
1. Rotate spool 90° clockwise (cost: -200)
2. Repair bottom quadrant (+4mm thickness) and rotate (cost: -1000)  
3. Replace spool (reset to 20.6mm thickness) (cost: -23000)

## Files Description

### Core Implementation
- `LDPMaintenanceEnv.py`: Main environment class implementing the RL environment
- `utils.py`: Utility functions including Weibull failure probability calculation
- `RL work scope.xlsx`: Configuration file with initial thickness, outage schedules, and erosion rates

### Training and Simulation
- `train_agent.py`: Q-learning agent implementation and training script
- `simulate_and_visualize.py`: 5-year simulation runner and visualization generator
- `run_complete_pipeline.py`: Main execution script that runs the entire pipeline

## Usage

### Quick Start
Run the complete pipeline:
```bash
python run_complete_pipeline.py
```

### Individual Components

1. **Train the agent only:**
```bash
python train_agent.py
```

2. **Run simulation with trained agent:**
```bash
python simulate_and_visualize.py
```

## Output Files

After running the pipeline, you'll get:

- `trained_agent.pkl`: Serialized trained RL agent
- `training_progress.png`: Training progress visualization
- `simulation_results.csv`: Daily simulation results
- `simulation_data.pkl`: Detailed simulation data
- `5_year_simulation_results.png`: Comprehensive 9-panel results visualization
- `thickness_analysis.png`: Detailed thickness evolution analysis

## Key Features

### Environment Features
- **Realistic Failure Modeling**: Uses Weibull distribution based on projected thickness
- **Seasonal Penalties**: Different costs for winter vs summer outages
- **Constraint Enforcement**: Only allows actions during valid outage periods
- **Erosion Simulation**: Realistic thickness degradation over time

### Agent Features
- **Q-Learning Algorithm**: Tabular reinforcement learning approach
- **State Discretization**: Efficient state representation for learning
- **Constraint-Aware Actions**: Respects operational constraints during action selection
- **Exploration Strategy**: Epsilon-greedy with decay for balanced exploration/exploitation

### Visualization Features
- Cumulative reward tracking
- Daily reward distribution analysis
- Failure pattern analysis
- Action frequency analysis  
- Seasonal performance patterns
- Thickness evolution tracking
- Comprehensive performance metrics

## Algorithm Details

The solution uses tabular Q-learning with:
- **State Space**: Discretized thickness levels, failure count, and date information
- **Action Space**: 4 actions per spool, constrained by operational schedules
- **Reward Function**: Combines action costs, failure penalties, and seasonal multipliers
- **Learning Parameters**: α=0.1, γ=0.99, ε-decay from 0.1 to 0.01

## Performance Metrics

The system tracks:
- Total cumulative reward over 5 years
- Number of failures prevented
- Cost efficiency of different action types
- Seasonal performance variations
- Thickness degradation patterns

## Requirements

```
numpy
pandas  
matplotlib
seaborn
openpyxl
```

Install with: `pip install numpy pandas matplotlib seaborn openpyxl`

## Notes

- The simulation runs for exactly 5 years (1825 days)
- All thickness measurements are in millimeters
- Costs are in arbitrary units (negative rewards)
- The critical failure threshold is 1.0mm thickness
- Net Operating Hours (NOH) assumes 80% uptime