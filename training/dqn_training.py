import os
import sys
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv  # Correct import
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import ChessEnv

# Wrapper to make old gym env compatible with stable-baselines3
def make_env():
    env = ChessEnv()
    return env

def train_dqn():
    # Create the environment
    env = DummyVecEnv([make_env])  # Wrap the environment

    # Define hyperparameters
    learning_rate = 1e-4
    gamma = 0.99
    batch_size = 32
    exploration_fraction = 0.1
    exploration_final_eps = 0.02
    total_timesteps = 100000

    # Create the DQN model
    model = DQN('MlpPolicy', env, learning_rate=learning_rate, gamma=gamma,
                batch_size=batch_size, exploration_fraction=exploration_fraction,
                exploration_final_eps=exploration_final_eps, verbose=1)

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save("models/dqn/chess_dqn")

if __name__ == "__main__":
    train_dqn()