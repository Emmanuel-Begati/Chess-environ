import gym
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import ChessEnv

def train_ppo(total_timesteps=200000, 
             learning_rate=0.0003, 
             n_steps=1024, 
             batch_size=128, 
             ent_coef=0.03, 
             gamma=0.99):
    """
    Train a PPO model with customizable hyperparameters
    
    Args:
        total_timesteps: Total timesteps to train
        learning_rate: Learning rate for the optimizer
        n_steps: Steps per environment per update
        batch_size: Batch size for training
        ent_coef: Entropy coefficient for loss calculation (higher = more exploration)
        gamma: Discount factor
    """
    # Create the custom chess environment
    env = ChessEnv()
    env = DummyVecEnv([lambda: env])  # Wrap the environment
    
    # Print training parameters
    print(f"Training PPO model with the following parameters:")
    print(f"Learning rate: {learning_rate}")
    print(f"N steps: {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"Gamma: {gamma}")
    print(f"Total timesteps: {total_timesteps}")

    # Define the PPO model
    model = PPO("MlpPolicy", env, 
               learning_rate=learning_rate, 
               n_steps=n_steps,
               batch_size=batch_size, 
               ent_coef=ent_coef, 
               gamma=gamma,
               verbose=1)

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Make sure the models directory exists
    os.makedirs("models/pg", exist_ok=True)
    
    # Save the model
    model.save("models/pg/ppo_chess")
    print("PPO model training complete and saved to models/pg/ppo_chess.zip")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train PPO chess model')
    
    parser.add_argument('--timesteps', type=int, default=200000,
                        help='Number of timesteps to train (default: 200000)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate (default: 0.0003)')
    parser.add_argument('--n_steps', type=int, default=1024,
                        help='Steps per update (default: 1024)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--ent_coef', type=float, default=0.03,
                        help='Entropy coefficient (default: 0.03)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    
    args = parser.parse_args()
    
    train_ppo(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        gamma=args.gamma
    )