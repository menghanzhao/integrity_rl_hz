import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import *
import random


class LDPMaintenanceEnv:
    def __init__(self, config_file_path):
        # Read data once and convert to efficient formats
        thickness_df = pd.read_excel(config_file_path, sheet_name="current thickness", header=0)
        self.outage_schedule = pd.read_excel(config_file_path, sheet_name="outage schedule", header=0)
        self.erosion_rate = np.array(pd.read_excel(config_file_path, sheet_name="erosion rate", header=0)['erosion rate (mm/1000NOH)'])
        
        # Convert thickness data to efficient numpy arrays and dictionaries
        self.spools = thickness_df['Spool'].unique()
        self.spool_to_idx = {spool: i for i, spool in enumerate(self.spools)}
        
        # Store thickness as numpy array [spool_idx, quadrant] 
        self.initial_thickness = np.zeros((len(self.spools), 4))
        self.thickness_array = np.zeros((len(self.spools), 4))
        
        for _, row in thickness_df.iterrows():
            spool_idx = self.spool_to_idx[row['Spool']]
            quadrant_idx = int(row['Relative Quadrant']) - 1
            thickness = row['Current Thickness']
            self.initial_thickness[spool_idx, quadrant_idx] = thickness
            self.thickness_array[spool_idx, quadrant_idx] = thickness
        
        self.simulation_date = datetime(2025, 1, 1)
        self.failures = set()  # Use set for faster lookup
        self.total_reward = 0
        
        # Pre-compute failure functions
        self.bottom_failure_prob_func = weibull_failure_probability(0.1, 20.6)
        self.left_right_failure_prob_func = weibull_failure_probability(0.2, 20.6)
        self.top_failure_prob_func = weibull_failure_probability(0.3, 20.6)

    def get_state(self):
        # Convert back to DataFrame format for compatibility
        thickness_data = []
        for i, spool in enumerate(self.spools):
            for q in range(4):
                thickness_data.append({
                    'Spool': spool,
                    'Relative Quadrant': q + 1,
                    'Current Thickness': self.thickness_array[i, q]
                })
        
        return {
            'day': self.simulation_date,
            'spool_thickness': pd.DataFrame(thickness_data),
            'failures': list(self.failures)
        }
    
    def is_outage(self, line):
        outage_list = self.outage_schedule[f'{line} outage list']
        # if the spool of a particular line is in fature list, then it is in outage
        
        return self.simulation_date in outage_list.values
    
    def calculate_projected_thickness(self, spool_idx, relative_quadrant, noh):
        current_thickness = self.thickness_array[spool_idx, relative_quadrant - 1]
        
        if relative_quadrant == 1: # top
            noise = random.uniform(-0.5, 0.5)
        elif relative_quadrant in [2, 4]:
            noise = random.uniform(-0.8, 0.8)
        else:
            noise = random.uniform(-3, 1.5)
        
        wear = (self.erosion_rate[relative_quadrant - 1] - noise) * noh / 1000
        return current_thickness - wear
    
    def is_failure(self, spool_idx, relative_quad, noh):
        projected_thickness = self.calculate_projected_thickness(spool_idx, relative_quad, noh)
        
        if relative_quad == 1:
            prob = self.top_failure_prob_func(projected_thickness)
        elif relative_quad in [2, 4]:
            prob = self.left_right_failure_prob_func(projected_thickness)
        else:
            prob = self.bottom_failure_prob_func(projected_thickness)
        
        return prob > random.random()
        
    def simulate_failure(self, noh):
        self.failures = set()
        reward = 0
        
        for spool_idx in range(len(self.spools)):
            for quadrant in range(1, 5):
                if self.is_failure(spool_idx, quadrant, noh):
                    spool = self.spools[spool_idx]
                    if spool not in self.failures:
                        reward += self.apply_failure(spool)
                        self.failures.add(spool)
                    break  # Only need one quadrant to fail for spool to fail
        return reward
    
    def apply_action(self, spool, action):
        spool_idx = self.spool_to_idx[spool]
        
        if action == 0: # do nothing
            return 0
        elif action == 1: # rotate 90 degrees clockwise
            # Rotate thickness values
            old_values = self.thickness_array[spool_idx].copy()
            self.thickness_array[spool_idx, 0] = old_values[3]  # top = left
            self.thickness_array[spool_idx, 1] = old_values[0]  # right = top
            self.thickness_array[spool_idx, 2] = old_values[1]  # bottom = right
            self.thickness_array[spool_idx, 3] = old_values[2]  # left = bottom
            return -200 
        elif action == 2: # repair (grow 4mm in the bottom quadrant, rotate that quadrant to the top)
            # Repair bottom quadrant (index 2)
            self.thickness_array[spool_idx, 2] += 4
            # Then rotate
            old_values = self.thickness_array[spool_idx].copy()
            self.thickness_array[spool_idx, 0] = old_values[3]  # top = left
            self.thickness_array[spool_idx, 1] = old_values[0]  # right = top
            self.thickness_array[spool_idx, 2] = old_values[1]  # bottom = right
            self.thickness_array[spool_idx, 3] = old_values[2]  # left = bottom
            return -1000
        elif action == 3: # replace
            # Reset all quadrants to nominal thickness
            self.thickness_array[spool_idx, :] = 20.6
            return -23000
        
    def get_valid_actions_for_day(self):
        """Return list of spools that can have actions taken on them today"""
        valid_spools = []
        
        # Check scheduled outages
        if self.is_outage('L1'):
            # L1 spools: all spools with '1' in name (line 1)
            for spool in self.spools:
                if '1' in spool:
                    valid_spools.append(spool)
        
        if self.is_outage('L2'):
            # L2 spools: all spools with '2' in name (line 2)  
            for spool in self.spools:
                if '2' in spool:
                    valid_spools.append(spool)
        
        # Check unscheduled outages (when failures occur)
        for failed_spool in self.failures:
            # Extract line from spool name and add all spools from that line
            if '1' in failed_spool:  # Line 1 failure
                for spool in self.spools:
                    if '1' in spool and spool not in valid_spools and spool not in self.failures:
                        valid_spools.append(spool)
            elif '2' in failed_spool:  # Line 2 failure
                for spool in self.spools:
                    if '2' in spool and spool not in valid_spools and spool not in self.failures:
                        valid_spools.append(spool)

        self.failures = set()
        
        return valid_spools

    def get_both_down_penalty(self):
        if len(self.failures) > 1:
            # if both lines in failure, then apply penalty
            lines = set()
            for spool in self.failures:
                if '1' in spool:
                    lines.add('1')
                elif '2' in spool:
                    lines.add('2')
            
            if len(lines) > 1:
                if self.simulation_date.month in [4,5,6,7,8,9,10,11]:
                    return -5000000
                elif self.simulation_date.month in [1,2,3,12]:
                    return -9800000
        elif len(self.failures) == 1:
            failed_spool = list(self.failures)[0]
            if '1' in failed_spool:
                other_line_outage = self.outage_schedule['L2 outage list']
            elif '2' in failed_spool:
                other_line_outage = self.outage_schedule['L1 outage list']
            else:
                return 0
                
            if self.simulation_date in other_line_outage.values:
                if self.simulation_date.month in [4,5,6,7,8,9,10,11]:
                    return -5000000
                elif self.simulation_date.month in [1,2,3,12]:
                    return -9800000
        return 0
    
    def step(self, actions):
        reward = 0

        # Advance simulation by one day
        self.simulation_date += timedelta(days=1)
        
        # Calculate NOH (Net Operating Hours) for the day
        noh = 24 * 0.8  # 80% uptime as specified in assumptions
        
        # Simulate failures
        failure_reward = self.simulate_failure(noh)
        reward += failure_reward
        
        # Apply both-lines-down penalty
        penalty = self.get_both_down_penalty()
        reward += penalty
        
        # Check if actions can be taken today
        valid_actions = self.get_valid_actions_for_day()
        
        # Apply only valid actions
        for spool, action in actions.items():
            if spool in valid_actions and action != 0:  # 0 = do nothing
                reward += self.apply_action(spool, action)
        
        # Update total reward
        self.total_reward += reward
        
        # Check if episode is done (5 years = 1825 days)
        start_date = datetime(2025, 1, 1)
        done = (self.simulation_date - start_date).days >= 1825
        
        return self.get_state(), reward, done, {'valid_actions': valid_actions}
    
    def apply_failure(self, spool):
        """Apply failure effects and return associated cost"""
        # Failure takes 1 day of repair and causes 1 day outage
        # when failed, spool needs to be replaced, change thickness to nominal
        spool_idx = self.spool_to_idx[spool]
        self.thickness_array[spool_idx, :] = 20.6  # Reset
        # self.failures.remove(spool)  # Remove from failures
        return -23000 - 1000  # Basic failure cost

    def reset(self):
        """Reset environment to initial state"""
        self.simulation_date = datetime(2025, 1, 1)
        self.failures = set()
        self.total_reward = 0
        # Reset thickness data to initial values
        self.thickness_array = self.initial_thickness.copy()
        return self.get_state()