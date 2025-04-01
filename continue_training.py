import os
import argparse
import time
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import ChessEnv

def make_env():
    """Create a chess environment instance"""
    return ChessEnv()

def continue_training(model_type='dqn', 
                     additional_timesteps=50000,
                     learning_rate=None,
                     make_default=True):
    """
    Continue training an existing model
    
    Args:
        model_type: 'dqn' or 'ppo'
        additional_timesteps: Number of additional timesteps to train
        learning_rate: Optional new learning rate (if None, keep the model's current rate)
        make_default: Whether to make the new version the default model
    """
    # Set model paths based on type
    if model_type == 'dqn':
        model_dir = "models/dqn"
        base_model_name = "chess_dqn"
    else:
        model_dir = "models/pg"
        base_model_name = "ppo_chess"
    
    model_path = f"{model_dir}/{base_model_name}"
    
    # Make sure model exists
    if not os.path.exists(f"{model_path}.zip"):
        print(f"Error: Model not found at {model_path}.zip")
        return False
    
    # Create environment
    env = DummyVecEnv([make_env])
    
    # Load existing model
    print(f"Loading existing {model_type.upper()} model from {model_path}.zip")
    
    # Determine model class based on type
    model_class = DQN if model_type == 'dqn' else PPO
    model = model_class.load(model_path, env=env)
    
    # Update learning rate if provided
    if learning_rate is not None:
        print(f"Updating learning rate to {learning_rate}")
        model.learning_rate = learning_rate
    
    # Create versions directory
    versions_dir = f"{model_dir}/versions"
    os.makedirs(versions_dir, exist_ok=True)
    
    # Create timestamp for versioning
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print(f"Continuing training for {additional_timesteps} additional timesteps...")
    
    # Train the model (continuing from current state)
    model.learn(total_timesteps=additional_timesteps, reset_num_timesteps=False)
    
    # Save versioned model
    version_path = f"{versions_dir}/{base_model_name}_{timestamp}"
    model.save(version_path)
    print(f"New version saved to {version_path}.zip")
    
    # Optionally make it the default model
    if make_default:
        model.save(model_path)
        print(f"Updated default model at {model_path}.zip")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continue training an existing chess model')
    
    parser.add_argument('--model_type', choices=['dqn', 'ppo'], default='dqn',
                      help='Type of model to continue training (default: dqn)')
    parser.add_argument('--timesteps', type=int, default=50000,
                      help='Number of additional timesteps to train (default: 50000)')
    parser.add_argument('--lr', type=float, default=None,
                      help='New learning rate (optional, default: keep existing rate)')
    parser.add_argument('--no-default', action='store_true',
                      help='Don\'t update the default model, just save the version')
    
    args = parser.parse_args()
    
    continue_training(
        model_type=args.model_type,
        additional_timesteps=args.timesteps,
        learning_rate=args.lr,
        make_default=not args.no_default
    )