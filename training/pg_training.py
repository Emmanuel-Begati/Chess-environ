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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import ChessEnv

def make_env():
    """Create a chess environment instance"""
    return ChessEnv()

def optimize_ppo(trial):
    """
    Optimization function for Optuna to find the best hyperparameters
    """
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    
    # Create the environment
    env = DummyVecEnv([make_env])
    
    # Create the model with the suggested hyperparameters
    model = PPO("MlpPolicy", env, 
               learning_rate=learning_rate, 
               n_steps=n_steps,
               batch_size=batch_size, 
               gamma=gamma,
               gae_lambda=gae_lambda,
               ent_coef=ent_coef, 
               clip_range=clip_range,
               verbose=0)
    
    # Train for a short period to evaluate performance
    evaluation_timesteps = 30000  # Use a small number for quick evaluations
    
    try:
        model.learn(total_timesteps=evaluation_timesteps)
        
        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        
        return mean_reward
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        return -1000  # Return a low score for failed trials

def find_best_hyperparameters(n_trials=20, study_name="ppo_optimization"):
    """
    Run hyperparameter optimization using Optuna
    
    Args:
        n_trials: Number of trials to run
        study_name: Name for the Optuna study
        
    Returns:
        Best hyperparameters found
    """
    # Create directory for storing tuning results
    tuning_dir = os.path.join("models", "tuning_results", "ppo")
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
    study.optimize(optimize_ppo, n_trials=n_trials, callbacks=[save_trial_callback])
    
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

def train_ppo(total_timesteps=200000, 
             learning_rate=0.0003, 
             n_steps=1024, 
             batch_size=128, 
             ent_coef=0.03, 
             gamma=0.99,
             gae_lambda=0.95,
             clip_range=0.2,
             optimize=False,
             n_trials=20):
    """
    Train a PPO model with optimized or specified hyperparameters
    
    Args:
        total_timesteps: Total timesteps to train
        learning_rate: Learning rate for the optimizer
        n_steps: Steps per environment per update
        batch_size: Batch size for training
        ent_coef: Entropy coefficient for loss calculation
        gamma: Discount factor
        gae_lambda: Factor for GAE
        clip_range: PPO clip range
        optimize: Whether to optimize hyperparameters first
        n_trials: Number of optimization trials if optimize=True
    """
    if optimize:
        print("Starting hyperparameter optimization...")
        best_params = find_best_hyperparameters(n_trials=n_trials)
        
        # Use optimized hyperparameters
        learning_rate = best_params.get("learning_rate", learning_rate)
        n_steps = best_params.get("n_steps", n_steps)
        batch_size = best_params.get("batch_size", batch_size)
        gamma = best_params.get("gamma", gamma)
        gae_lambda = best_params.get("gae_lambda", gae_lambda)
        ent_coef = best_params.get("ent_coef", ent_coef)
        clip_range = best_params.get("clip_range", clip_range)
    
    # Create the environment
    env = DummyVecEnv([make_env])
    
    # Print training parameters
    print(f"\nTraining PPO model with the following parameters:")
    print(f"Learning rate: {learning_rate}")
    print(f"N steps: {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"Gamma: {gamma}")
    print(f"GAE Lambda: {gae_lambda}")
    print(f"Clip range: {clip_range}")
    print(f"Total timesteps: {total_timesteps}")

    # Define the PPO model
    model = PPO("MlpPolicy", env, 
               learning_rate=learning_rate, 
               n_steps=n_steps,
               batch_size=batch_size, 
               ent_coef=ent_coef, 
               gamma=gamma,
               gae_lambda=gae_lambda,
               clip_range=clip_range,
               verbose=1)

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Make sure the models directory exists
    os.makedirs("models/pg", exist_ok=True)
    
    # Save the model and hyperparameters
    model.save("models/pg/ppo_chess")
    
    # Save hyperparameters to a file for reference
    hyperparams = {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "ent_coef": ent_coef,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "total_timesteps": total_timesteps,
        "optimized": optimize
    }
    
    with open("models/pg/hyperparameters.json", "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    print("PPO model training complete and saved to models/pg/ppo_chess.zip")
    print("Hyperparameters saved to models/pg/hyperparameters.json")

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
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda (default: 0.95)')
    parser.add_argument('--clip_range', type=float, default=0.2,
                        help='PPO clip range (default: 0.2)')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize hyperparameters before training')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of optimization trials (default: 20)')
    
    args = parser.parse_args()
    
    train_ppo(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        optimize=args.optimize,
        n_trials=args.n_trials
    )