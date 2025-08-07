import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import pickle
from datetime import datetime, timedelta
from LDPMaintenanceEnv import LDPMaintenanceEnv

class QLearningAgent:
    def __init__(self, n_spools=4, n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_spools = n_spools
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(self._create_action_dict)
    
    def _create_action_dict(self):
        """Create a new defaultdict for action values - picklable alternative to lambda"""
        return defaultdict(float)
        
    def get_state_key(self, state):
        """Convert state to a hashable key"""
        # Simplify state representation for Q-learning
        thickness_data = state['spool_thickness']
        failures = state['failures']
        date = state['day']
        
        # Create a simple state representation
        state_key = []
        for _, row in thickness_data.iterrows():
            # Discretize thickness values
            avg_thickness = (row.get('Quadrant 1', 0) + row.get('Quadrant 2', 0) + 
                           row.get('Quadrant 3', 0) + row.get('Quadrant 4', 0)) / 4
            thickness_bin = min(int(avg_thickness / 5), 4)  # 0-4 bins
            state_key.append(thickness_bin)
        
        # Add failure info
        state_key.append(len(failures))
        
        # Add month info for seasonal effects
        state_key.append(date.month)
        
        return tuple(state_key)
    
    def choose_action(self, state, valid_spools=None):
        """Choose action using epsilon-greedy policy, respecting valid spools"""
        state_key = self.get_state_key(state)
        
        actions = {}
        thickness_data = state['spool_thickness']
        
        for _, row in thickness_data.iterrows():
            spool = row['Spool']
            
            # If spool is not in valid spools list, default to "do nothing" (action 0)
            if valid_spools is not None and spool not in valid_spools:
                actions[spool] = 0
                continue
            
            if random.random() < self.epsilon:
                # Explore: random action
                actions[spool] = random.randint(0, self.n_actions - 1)
            else:
                # Exploit: best action
                q_values = [self.q_table[state_key][(spool, a)] for a in range(self.n_actions)]
                actions[spool] = np.argmax(q_values)
        
        return actions
    
    def update_q_table(self, state, actions, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        for spool, action in actions.items():
            # Current Q-value
            current_q = self.q_table[state_key][(spool, action)]
            
            # Next state max Q-value
            next_q_values = [self.q_table[next_state_key][(spool, a)] for a in range(self.n_actions)]
            max_next_q = max(next_q_values) if next_q_values else 0
            
            # Q-learning update
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.q_table[state_key][(spool, action)] = new_q
    
    def decay_epsilon(self, decay_rate=0.995):
        """Decay exploration rate"""
        self.epsilon = max(0.01, self.epsilon * decay_rate)

def train_agent(config_file, n_episodes=1000, save_path='trained_agent.pkl'):
    """Train the Q-learning agent"""
    env = LDPMaintenanceEnv(config_file)
    agent = QLearningAgent()
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"Starting training for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Get valid spools for today (before taking actions)
            valid_spools = env.get_valid_actions_for_day()
            
            # Choose actions
            actions = agent.choose_action(state, valid_spools)
            
            # Take step
            next_state, reward, done, info = env.step(actions)
            
            # Update Q-table
            agent.update_q_table(state, actions, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        agent.decay_epsilon()
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    # Save trained agent
    with open(save_path, 'wb') as f:
        pickle.dump(agent, f)
    
    print(f"Training completed. Agent saved to {save_path}")
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    # Moving average
    window = 100
    moving_avg = [np.mean(episode_rewards[i:i+window]) for i in range(len(episode_rewards)-window)]
    plt.plot(moving_avg)
    plt.title('Moving Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    
    return agent, episode_rewards

if __name__ == "__main__":
    config_file = "RL work scope.xlsx"
    agent, rewards = train_agent(config_file, n_episodes=1000)