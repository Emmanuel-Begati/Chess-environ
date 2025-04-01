import os
import sys
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import ChessEnv

# Wrapper to make old gym env compatible with stable-baselines3
def make_env():
    env = ChessEnv()
    return env

def train_dqn(total_timesteps=100000, 
              learning_rate=1e-4, 
              gamma=0.99, 
              batch_size=128, 
              exploration_fraction=0.6, 
              exploration_final_eps=0.15,
              buffer_size=200000):
    """
    Train a DQN model with customizable hyperparameters
    
    Args:
        total_timesteps: Total number of timesteps to train
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor
        batch_size: Batch size for training
        exploration_fraction: Fraction of training time exploration decreases
        exploration_final_eps: Final probability of random actions
        buffer_size: Size of the replay buffer
    """
    # Create the environment
    env = DummyVecEnv([make_env])  # Wrap the environment

    # Print training parameters
    print(f"Training DQN model with the following parameters:")
    print(f"Learning rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Batch size: {batch_size}")
    print(f"Exploration fraction: {exploration_fraction}")
    print(f"Final exploration epsilon: {exploration_final_eps}")
    print(f"Buffer size: {buffer_size}")
    print(f"Total timesteps: {total_timesteps}")

    # Create the DQN model
    model = DQN('MlpPolicy', env, 
                learning_rate=learning_rate, 
                gamma=gamma,
                batch_size=batch_size, 
                exploration_fraction=exploration_fraction,
                exploration_final_eps=exploration_final_eps, 
                buffer_size=buffer_size,
                verbose=1)

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Make sure the models directory exists
    os.makedirs("models/dqn", exist_ok=True)
    
    # Save the model
    model.save("models/dqn/chess_dqn")
    print("DQN model training complete and saved to models/dqn/chess_dqn.zip")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train DQN chess model')
    
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Number of timesteps to train (default: 100000)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--exp_fraction', type=float, default=0.6,
                        help='Exploration fraction (default: 0.6)')
    parser.add_argument('--final_eps', type=float, default=0.15,
                        help='Final exploration epsilon (default: 0.15)')
    parser.add_argument('--buffer_size', type=int, default=200000,
                        help='Size of replay buffer (default: 200000)')
    
    args = parser.parse_args()
    
    train_dqn(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        exploration_fraction=args.exp_fraction,
        exploration_final_eps=args.final_eps,
        buffer_size=args.buffer_size
    )