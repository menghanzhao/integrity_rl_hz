import pandas as pd
from datetime import datetime, timedelta
from utils import *
import random


class LDPMaintenanceEnv:
    def __init__(self, config_file_path):
        self.thickness = pd.read_excel(config_file_path, sheet_name="current thickness", header=0)
        self.outage_schedule = pd.read_excel(config_file_path, sheet_name="outage schedule", header=0)
        self.erosion_rate = list(pd.read_excel(config_file_path, sheet_name="erosion rate", header=0)['erosion rate (mm/1000NOH)'])
        self.simulation_date = datetime(2025, 1, 1)
        self.failures = []
        self.total_reward = 0
        self.bottom_failure_prob_func = weibull_failure_probability(0.1, 20.6)
        self.left_right_failure_prob_func = weibull_failure_probability(0.2, 20.6)
        self.top_failure_prob_func = weibull_failure_probability(0.3, 20.6)

    def get_state(self):
        return {
            'day': self.simulation_date,
            'spool_thickness': self.thickness,
            'failures': self.failures.copy()
        }
    
    def is_outage(self, line):
        outage_list = self.outage_schedule[f'{line} outage list']
        return self.simulation_date in outage_list.values
    
    def calculate_projected_thickness(self, spool, relative_quadrant, noh):
        current_thickness = self.thickness.loc[(self.thickness['Spool'] == spool) & (self.thickness['Relative Quadrant'] == relative_quadrant), 'Current Thickness'].iloc[0]
        if relative_quadrant == 1: # top
            noise = random.uniform(-0.5, 0.5)
        elif relative_quadrant in [2,4]:
            noise = random.uniform(-0.8, 0.8)
        else:
            noise = random.uniform(-3, 1.5)
        wear = (self.erosion_rate[relative_quadrant - 1] - noise) * noh / 1000
        return current_thickness - wear
    
    def is_failure(self, spool, relative_quad, noh):
        projected_thickness = self.calculate_projected_thickness(spool, relative_quad, noh)
        if relative_quad == 1:
            return self.top_failure_prob_func(projected_thickness) > random.random()
        elif relative_quad in [2, 4]:
            return self.left_right_failure_prob_func(projected_thickness) > random.random()
        else:
            return self.bottom_failure_prob_func(projected_thickness) > random.random()
        
    def simulate_failure(self, noh):
        self.failures = []
        reward = 0
        # Check failure for each spool and quadrant
        spools = self.thickness['Spool'].unique()
        for spool in spools:
            for quadrant in [1, 2, 3, 4]:
                is_failure = self.is_failure(spool, quadrant, noh)
                if is_failure:
                    if spool not in self.failures:
                        reward += self.apply_failure(spool)
                        self.failures.append(spool)
                    break  # Only need one quadrant to fail for spool to fail
        return reward
    
    def apply_action(self, spool, action):
        if action == 0: # do nothing
            return 0
        elif action == 1: # rotate 90 degrees clockwise
            self.thickness.loc[self.thickness['Spool'] == spool, "Relative Quadrant"] = (self.thickness.loc[self.thickness['Spool'] == spool, 'Relative Quadrant'] % 4) + 1
            return -200 
        elif action == 2: # repair (grow 4mm in the bottom quadrant, rotate that quadrant to the top)
            self.thickness.loc[(self.thickness['Spool'] == spool) & (self.thickness['Relative Quadrant'] == 3), 'Current Thickness'] += 4
            self.thickness.loc[self.thickness['Spool'] == spool, 'Relative Quadrant'] = (self.thickness.loc[self.thickness['Spool'] == spool, 'Relative Quadrant'] % 4) + 1
            return -1000
        elif action == 3: # replace
            self.thickness.loc[self.thickness['Spool'] == spool, 'Relative Quadrant'] = self.thickness.loc[self.thickness['Spool'] == spool, 'Quadrant']
            self.thickness.loc[self.thickness['Spool'] == spool, 'Current Thickness'] = 20.6
            return -23000
        
    def get_both_down_panelty(self):
        if len(self.failures) > 1:
            # if both lines in failure, then apply penalty
            lines = []
            for spool in self.failures:
                if spool[1] not in lines:
                    lines.append(spool[1])
            if len(lines) > 1 and self.simulation_date.month in [4,5,6,7,8,9,10,11]:
                return -5000000
            elif len(lines) > 1 and self.simulation_date.month in [1,2,3,12]:
                return -9800000
        elif len(self.failures) == 1:
            failed_line = f"L{self.failures[0][1]}"
            if failed_line == 'L1':
                other_line_outage = self.outage_schedule['L2 outage list']
                if self.simulation_date in other_line_outage.values and self.simulation_date.month in [4,5,6,7,8,9,10,11]:
                    return -5000000
                elif self.simulation_date in other_line_outage.values and self.simulation_date.month in [1,2,3,12]:
                    return -9800000
            elif failed_line == 'L2':
                other_line_outage = self.outage_schedule['L1 outage list']
                if self.simulation_date in other_line_outage.values and self.simulation_date.month in [4,5,6,7,8,9,10,11]:
                    return -5000000
                elif self.simulation_date in other_line_outage.values and self.simulation_date.month in [1,2,3,12]:
                    return -9800000
        return 0
    
    def get_valid_actions_for_day(self):
        """Return list of spools that can have actions taken on them today"""
        valid_spools = []
        
        # Check scheduled outages
        if self.is_outage('L1'):
            # L1 spools: Spool1A, Spool1B, Spool2A, Spool2B (line 1)
            for _, row in self.thickness.iterrows():
                spool_name = row['Spool']
                if '1' in spool_name:  # Line 1 spools
                    valid_spools.append(spool_name)
        
        if self.is_outage('L2'):
            # L2 spools: all spools with '2' in name (line 2)  
            for _, row in self.thickness.iterrows():
                spool_name = row['Spool']
                if '2' in spool_name:  # Line 2 spools
                    valid_spools.append(spool_name)
        
        # Check unscheduled outages (when failures occur)
        for failed_spool in self.failures:
            # Extract line from spool name and add all spools from that line
            if '1' in failed_spool:  # Line 1 failure
                for _, row in self.thickness.iterrows():
                    spool_name = row['Spool']
                    if '1' in spool_name and spool_name not in valid_spools:
                        valid_spools.append(spool_name)
            elif '2' in failed_spool:  # Line 2 failure
                for _, row in self.thickness.iterrows():
                    spool_name = row['Spool']
                    if '2' in spool_name and spool_name not in valid_spools:
                        valid_spools.append(spool_name)
        
        return valid_spools
    
    def step(self, actions):
        reward = 0
        
        # Check if actions can be taken today
        valid_actions = self.get_valid_actions_for_day()
        
        # Apply only valid actions
        for spool, action in actions.items():
            if spool in valid_actions and action != 0:  # 0 = do nothing
                reward += self.apply_action(spool, action)
        
        # Advance simulation by one day
        self.simulation_date += timedelta(days=1)
        
        # Calculate NOH (Net Operating Hours) for the day
        noh = 24 * 0.8  # 80% uptime as specified in assumptions
        
        # Simulate failures
        failure_reward = self.simulate_failure(noh)
        reward += failure_reward
        
        # Apply both-lines-down penalty
        penalty = self.get_both_down_panelty()
        reward += penalty
        
        # Update total reward
        self.total_reward += reward
        
        # Check if episode is done (5 years = 1825 days)
        start_date = datetime(2025, 1, 1)
        done = (self.simulation_date - start_date).days >= 1825
        
        return self.get_state(), reward, done, {'valid_actions': valid_actions}
    
    def apply_failure(self, spool):
        """Apply failure effects and return associated cost"""
        # Failure takes 1 day of repair and causes 1 day outage
        return -1000  # Basic failure cost
    
    def reset(self):
        """Reset environment to initial state"""
        self.simulation_date = datetime(2025, 1, 1)
        self.failures = []
        self.total_reward = 0
        # Reset thickness data would need to reload from Excel
        return self.get_state()