import os
import sys
import numpy as np
import optuna
import json
import pandas as pd
from datetime import datetime
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import ChessEnv

# Wrapper to make env compatible with stable-baselines3
def make_env():
    env = ChessEnv()
    return env

def optimize_dqn(trial):
    """
    Optimization function for Optuna to find the best hyperparameters
    """
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.9)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.3)
    buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 200000, 500000])
    
    # Create the environment
    env = DummyVecEnv([make_env])
    
    # Create the model with the suggested hyperparameters
    model = DQN('MlpPolicy', env, 
                learning_rate=learning_rate, 
                gamma=gamma,
                batch_size=batch_size, 
                exploration_fraction=exploration_fraction,
                exploration_final_eps=exploration_final_eps, 
                buffer_size=buffer_size,
                verbose=0)
    
    # Train for a short period to evaluate performance
    evaluation_timesteps = 20000  # Use a small number for quick evaluations
    
    try:
        model.learn(total_timesteps=evaluation_timesteps)
        
        # Evaluate the model against a random agent and/or self-play
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        
        return mean_reward
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        return -1000  # Return a low score for failed trials

def find_best_hyperparameters(n_trials=20, study_name="dqn_optimization"):
    """
    Run hyperparameter optimization using Optuna
    
    Args:
        n_trials: Number of trials to run
        study_name: Name for the Optuna study
        
    Returns:
        Best hyperparameters found
    """
    # Create directory for storing tuning results
    tuning_dir = os.path.join("models", "tuning_results", "dqn")
    os.makedirs(tuning_dir, exist_ok=True)
    
    # Create timestamp for the tuning session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_file = os.path.join(tuning_dir, f"tuning_results_{timestamp}.json")
    tuning_csv = os.path.join(tuning_dir, f"tuning_results_{timestamp}.csv")
    
    # Create the Optuna study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    # Define a callback to save results after each trial
    def save_trial_callback(study, trial):
        # Get all completed trials
        trials_df = study.trials_dataframe()
        
        # Save as CSV for easy analysis
        trials_df.to_csv(tuning_csv, index=False)
        
        # Also save as JSON with more details
        trials_data = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                trial_data = {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "datetime": t.datetime_start.isoformat()
                }
                trials_data.append(trial_data)
        
        tuning_results = {
            "study_name": study_name,
            "direction": "maximize",
            "timestamp": timestamp,
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "trials": trials_data
        }
        
        with open(tuning_file, 'w') as f:
            json.dump(tuning_results, f, indent=4)
        
        # Print progress
        if len(study.trials) % 5 == 0:  # Print every 5 trials
            print(f"Completed {len(study.trials)}/{n_trials} trials. Current best value: {study.best_value}")
    
    # Run the optimization
    print(f"Running {n_trials} trials to find optimal hyperparameters...")
    print(f"Saving tuning results to {tuning_file}")
    study.optimize(optimize_dqn, n_trials=n_trials, callbacks=[save_trial_callback])
    
    # Get best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n=== Best Hyperparameters Found ===")
    print(f"Best reward achieved: {best_value}")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Generate summary plots
    try:
        # Import visualization dependencies
        from optuna.visualization import plot_optimization_history, plot_param_importances
        import matplotlib.pyplot as plt
        
        # Create plots directory
        plots_dir = os.path.join(tuning_dir, f"plots_{timestamp}")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save optimization history plot
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(plots_dir, "optimization_history.png"))
        
        # Save parameter importances plot
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(plots_dir, "param_importances.png"))
        
        print(f"Visualization plots saved to {plots_dir}")
    except ImportError:
        print("Skipping visualization plots - required packages not available")
    
    # Save final results to a dedicated file
    final_results = {
        "best_params": best_params,
        "best_value": best_value,
        "timestamp": timestamp,
        "n_trials": n_trials,
        "tuning_file": tuning_file,
    }
    
    with open(os.path.join(tuning_dir, f"best_params_{timestamp}.json"), 'w') as f:
        json.dump(final_results, f, indent=4)
    
    return best_params

def train_dqn(total_timesteps=100000, 
              learning_rate=1e-4, 
              gamma=0.99, 
              batch_size=128, 
              exploration_fraction=0.6, 
              exploration_final_eps=0.15,
              buffer_size=200000,
              optimize=False,
              n_trials=20):
    """
    Train a DQN model with optimized or specified hyperparameters
    
    Args:
        total_timesteps: Total number of timesteps to train
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor
        batch_size: Batch size for training
        exploration_fraction: Fraction of training time exploration decreases
        exploration_final_eps: Final probability of random actions
        buffer_size: Size of the replay buffer
        optimize: Whether to optimize hyperparameters first
        n_trials: Number of optimization trials if optimize=True
    """
    if optimize:
        print("Starting hyperparameter optimization...")
        best_params = find_best_hyperparameters(n_trials=n_trials)
        
        # Use optimized hyperparameters
        learning_rate = best_params.get("learning_rate", learning_rate)
        gamma = best_params.get("gamma", gamma)
        batch_size = best_params.get("batch_size", batch_size)
        exploration_fraction = best_params.get("exploration_fraction", exploration_fraction)
        exploration_final_eps = best_params.get("exploration_final_eps", exploration_final_eps)
        buffer_size = best_params.get("buffer_size", buffer_size)
    
    # Create the environment
    env = DummyVecEnv([make_env])

    # Print training parameters
    print(f"\nTraining DQN model with the following parameters:")
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
    
    # Save the model and hyperparameters
    model.save("models/dqn/chess_dqn")
    
    # Save hyperparameters to a file for reference
    hyperparams = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "batch_size": batch_size,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "buffer_size": buffer_size,
        "total_timesteps": total_timesteps,
        "optimized": optimize
    }
    
    with open("models/dqn/hyperparameters.json", "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    print("DQN model training complete and saved to models/dqn/chess_dqn.zip")
    print("Hyperparameters saved to models/dqn/hyperparameters.json")

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
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize hyperparameters before training')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of optimization trials (default: 20)')
    
    args = parser.parse_args()
    
    train_dqn(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        exploration_fraction=args.exp_fraction,
        exploration_final_eps=args.final_eps,
        buffer_size=args.buffer_size,
        optimize=args.optimize,
        n_trials=args.n_trials
    )