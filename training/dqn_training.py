import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv
from environment.custom_env import ChessEnv  # Assuming custom_env.py is in the environment folder

def train_dqn():
    # Create the environment
    env = ChessEnv()
    env = DummyVecEnv([lambda: env])  # Wrap the environment

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