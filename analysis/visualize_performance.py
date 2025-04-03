import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import ChessEnv

def make_env():
    """Create a monitored chess environment for tracking rewards"""
    os.makedirs("logs", exist_ok=True)
    env = Monitor(ChessEnv(), "logs/chess_monitor")
    return env

def evaluate_models(n_eval_episodes=10, n_games=100):
    """
    Evaluate DQN and PPO models and collect performance metrics
    
    Args:
        n_eval_episodes: Number of evaluation episodes per game
        n_games: Number of games to evaluate
    
    Returns:
        Dictionary containing performance metrics
    """
    print("Evaluating DQN and PPO models...")
    
    # Load models
    dqn_path = "models/dqn/chess_dqn.zip"
    ppo_path = "models/pg/ppo_chess.zip"
    
    if not os.path.exists(dqn_path) or not os.path.exists(ppo_path):
        print("ERROR: Model files not found. Please train models first.")
        return None
    
    # Create evaluation environment
    env = DummyVecEnv([make_env])
    
    # Load models
    dqn_model = DQN.load(dqn_path)
    ppo_model = PPO.load(ppo_path)
    
    results = {
        "dqn": {"rewards": [], "wins": 0, "draws": 0, "losses": 0},
        "ppo": {"rewards": [], "wins": 0, "draws": 0, "losses": 0}
    }
    
    # Evaluate models
    for game in range(n_games):
        print(f"Evaluation game {game+1}/{n_games}")
        
        # Evaluate DQN
        print("Evaluating DQN...")
        dqn_rewards, _ = evaluate_policy(dqn_model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True)
        results["dqn"]["rewards"].extend(dqn_rewards)
        
        # Play a full game to record outcome
        env = ChessEnv()
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = dqn_model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        
        if 'outcome' in info:
            if info['outcome'] == 'win':
                results["dqn"]["wins"] += 1
            elif info['outcome'] == 'draw':
                results["dqn"]["draws"] += 1
            else:
                results["dqn"]["losses"] += 1
        
        # Evaluate PPO
        print("Evaluating PPO...")
        env = DummyVecEnv([make_env])
        ppo_rewards, _ = evaluate_policy(ppo_model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True)
        results["ppo"]["rewards"].extend(ppo_rewards)
        
        # Play a full game to record outcome
        env = ChessEnv()
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        
        if 'outcome' in info:
            if info['outcome'] == 'win':
                results["ppo"]["wins"] += 1
            elif info['outcome'] == 'draw':
                results["ppo"]["draws"] += 1
            else:
                results["ppo"]["losses"] += 1
    
    return results

def load_tuning_data():
    """Load hyperparameter tuning data for visualization"""
    data = {
        "dqn": {"trials": [], "best_value": 0, "best_params": {}},
        "ppo": {"trials": [], "best_value": 0, "best_params": {}}
    }
    
    # Load DQN tuning data
    dqn_files = glob.glob("models/tuning_results/dqn/tuning_results_*.json")
    if dqn_files:
        latest_dqn = max(dqn_files, key=os.path.getctime)
        with open(latest_dqn, 'r') as f:
            dqn_data = json.load(f)
            data["dqn"]["trials"] = [(t["number"], t["value"]) for t in dqn_data["trials"]]
            data["dqn"]["best_value"] = dqn_data["best_value"]
            data["dqn"]["best_params"] = dqn_data["best_params"]
    
    # Load PPO tuning data
    ppo_files = glob.glob("models/tuning_results/ppo/tuning_results_*.json")
    if ppo_files:
        latest_ppo = max(ppo_files, key=os.path.getctime)
        with open(latest_ppo, 'r') as f:
            ppo_data = json.load(f)
            data["ppo"]["trials"] = [(t["number"], t["value"]) for t in ppo_data["trials"]]
            data["ppo"]["best_value"] = ppo_data["best_value"]
            data["ppo"]["best_params"] = ppo_data["best_params"]
    
    return data

def plot_cumulative_rewards(results):
    """Plot cumulative rewards over episodes for both methods"""
    if not results:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Calculate cumulative rewards
    dqn_cum_rewards = np.cumsum(results["dqn"]["rewards"])
    ppo_cum_rewards = np.cumsum(results["ppo"]["rewards"])
    
    episodes = range(1, len(dqn_cum_rewards) + 1)
    
    plt.plot(episodes, dqn_cum_rewards, label='DQN', color='blue', linewidth=2)
    plt.plot(episodes, ppo_cum_rewards, label='PPO', color='red', linewidth=2)
    
    plt.title('Cumulative Rewards Over Episodes', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Cumulative Reward', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    os.makedirs("analysis/figures", exist_ok=True)
    plt.savefig("analysis/figures/cumulative_rewards.png", dpi=300)
    plt.close()
    
    print("Cumulative rewards plot saved to analysis/figures/cumulative_rewards.png")

def plot_training_stability(tuning_data):
    """Plot training stability metrics from hyperparameter tuning data"""
    if not tuning_data["dqn"]["trials"] or not tuning_data["ppo"]["trials"]:
        print("No tuning data found for plotting stability metrics")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # DQN stability
    dqn_trial_nums, dqn_values = zip(*sorted(tuning_data["dqn"]["trials"]))
    ax1.plot(dqn_trial_nums, dqn_values, 'o-', color='blue', markersize=8, linewidth=2)
    ax1.axhline(y=tuning_data["dqn"]["best_value"], color='green', linestyle='--', label=f'Best value: {tuning_data["dqn"]["best_value"]:.2f}')
    ax1.set_title('DQN Training Stability', fontsize=16)
    ax1.set_xlabel('Trial Number', fontsize=14)
    ax1.set_ylabel('Mean Reward', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # PPO stability
    ppo_trial_nums, ppo_values = zip(*sorted(tuning_data["ppo"]["trials"]))
    ax2.plot(ppo_trial_nums, ppo_values, 'o-', color='red', markersize=8, linewidth=2)
    ax2.axhline(y=tuning_data["ppo"]["best_value"], color='green', linestyle='--', label=f'Best value: {tuning_data["ppo"]["best_value"]:.2f}')
    ax2.set_title('PPO Training Stability', fontsize=16)
    ax2.set_xlabel('Trial Number', fontsize=14)
    ax2.set_ylabel('Mean Reward', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    os.makedirs("analysis/figures", exist_ok=True)
    plt.savefig("analysis/figures/training_stability.png", dpi=300)
    plt.close()
    
    print("Training stability plot saved to analysis/figures/training_stability.png")

def plot_reward_distributions(results):
    """Plot reward distributions for both methods"""
    if not results:
        return
    
    plt.figure(figsize=(12, 6))
    
    sns.histplot(results["dqn"]["rewards"], kde=True, color='blue', alpha=0.5, label='DQN')
    sns.histplot(results["ppo"]["rewards"], kde=True, color='red', alpha=0.5, label='PPO')
    
    plt.title('Reward Distributions', fontsize=16)
    plt.xlabel('Reward', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    os.makedirs("analysis/figures", exist_ok=True)
    plt.savefig("analysis/figures/reward_distributions.png", dpi=300)
    plt.close()
    
    print("Reward distributions plot saved to analysis/figures/reward_distributions.png")

def plot_game_outcomes(results):
    """Plot game outcomes for both methods"""
    if not results:
        return
    
    labels = ['Wins', 'Draws', 'Losses']
    dqn_outcomes = [results["dqn"]["wins"], results["dqn"]["draws"], results["dqn"]["losses"]]
    ppo_outcomes = [results["ppo"]["wins"], results["ppo"]["draws"], results["ppo"]["losses"]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, dqn_outcomes, width, label='DQN', color='blue', alpha=0.7)
    rects2 = ax.bar(x + width/2, ppo_outcomes, width, label='PPO', color='red', alpha=0.7)
    
    ax.set_title('Game Outcomes', fontsize=16)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    plt.tight_layout()
    os.makedirs("analysis/figures", exist_ok=True)
    plt.savefig("analysis/figures/game_outcomes.png", dpi=300)
    plt.close()
    
    print("Game outcomes plot saved to analysis/figures/game_outcomes.png")

def print_hyperparameter_summary():
    """Print summary of optimized hyperparameters"""
    # Load DQN hyperparameters
    dqn_path = "models/dqn/hyperparameters.json"
    if os.path.exists(dqn_path):
        with open(dqn_path, 'r') as f:
            dqn_params = json.load(f)
        
        print("\n=== DQN Hyperparameters ===")
        for param, value in dqn_params.items():
            print(f"{param}: {value}")
    
    # Load PPO hyperparameters
    ppo_path = "models/pg/hyperparameters.json"
    if os.path.exists(ppo_path):
        with open(ppo_path, 'r') as f:
            ppo_params = json.load(f)
        
        print("\n=== PPO Hyperparameters ===")
        for param, value in ppo_params.items():
            print(f"{param}: {value}")

def generate_comparison_report(results, tuning_data):
    """Generate a markdown report comparing model performance"""
    if not results or not tuning_data:
        return
    
    os.makedirs("analysis", exist_ok=True)
    report_path = "analysis/model_comparison.md"
    
    # Calculate metrics
    dqn_avg_reward = np.mean(results["dqn"]["rewards"])
    ppo_avg_reward = np.mean(results["ppo"]["rewards"])
    dqn_win_rate = results["dqn"]["wins"] / (results["dqn"]["wins"] + results["dqn"]["draws"] + results["dqn"]["losses"]) if (results["dqn"]["wins"] + results["dqn"]["draws"] + results["dqn"]["losses"]) > 0 else 0
    ppo_win_rate = results["ppo"]["wins"] / (results["ppo"]["wins"] + results["ppo"]["draws"] + results["ppo"]["losses"]) if (results["ppo"]["wins"] + results["ppo"]["draws"] + results["ppo"]["losses"]) > 0 else 0
    
    with open(report_path, 'w') as f:
        f.write("# Chess Reinforcement Learning Models Comparison\n\n")
        
        f.write("## 1. Performance Metrics\n\n")
        f.write("| Metric | DQN | PPO |\n")
        f.write("|--------|-----|-----|\n")
        f.write(f"| Average Reward | {dqn_avg_reward:.2f} | {ppo_avg_reward:.2f} |\n")
        f.write(f"| Best Tuning Reward | {tuning_data['dqn']['best_value']:.2f} | {tuning_data['ppo']['best_value']:.2f} |\n")
        f.write(f"| Win Rate | {dqn_win_rate:.2%} | {ppo_win_rate:.2%} |\n")
        f.write(f"| Wins | {results['dqn']['wins']} | {results['ppo']['wins']} |\n")
        f.write(f"| Draws | {results['dqn']['draws']} | {results['ppo']['draws']} |\n")
        f.write(f"| Losses | {results['dqn']['losses']} | {results['ppo']['losses']} |\n\n")
        
        f.write("## 2. Optimized Hyperparameters\n\n")
        
        f.write("### DQN Hyperparameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        for param, value in tuning_data["dqn"]["best_params"].items():
            f.write(f"| {param} | {value} |\n")
        
        f.write("\n### PPO Hyperparameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        for param, value in tuning_data["ppo"]["best_params"].items():
            f.write(f"| {param} | {value} |\n")
        
        f.write("\n## 100. Analysis\n\n")
        
        # Determine which algorithm performed better
        better_algo = "PPO" if ppo_avg_reward > dqn_avg_reward else "DQN"
        f.write(f"Based on the evaluation metrics, **{better_algo}** demonstrated better performance in the chess environment. ")
        
        if better_algo == "PPO":
            f.write("This is consistent with expectations as policy gradient methods like PPO often perform better in environments with large action spaces and complex state representations like chess.\n\n")
        else:
            f.write("This is an interesting result as policy gradient methods like PPO often perform better in environments with large action spaces, but in this case DQN showed superior performance.\n\n")
        
        f.write("### Stability Analysis\n\n")
        f.write("The training stability plots show how the rewards evolved during hyperparameter optimization. ")
        ppo_stability = "more stable" if np.std([v for _, v in tuning_data["ppo"]["trials"]]) < np.std([v for _, v in tuning_data["dqn"]["trials"]]) else "less stable"
        f.write(f"PPO exhibited {ppo_stability} learning compared to DQN, which aligns with the theoretical properties of these algorithms.\n\n")
        
        f.write("### Reward Distribution\n\n")
        f.write("The reward distribution plots highlight the spread of rewards achieved by each algorithm. ")
        f.write("A wider distribution suggests more variability in performance, while a distribution skewed towards higher values indicates better average performance.\n\n")
        
        f.write("## 4. Visualizations\n\n")
        f.write("Please refer to the following visualizations for a graphical comparison:\n\n")
        f.write("- ![Cumulative Rewards](figures/cumulative_rewards.png)\n")
        f.write("- ![Training Stability](figures/training_stability.png)\n")
        f.write("- ![Reward Distributions](figures/reward_distributions.png)\n")
        f.write("- ![Game Outcomes](figures/game_outcomes.png)\n\n")
        
        f.write("## 5. Conclusion\n\n")
        f.write("The analysis demonstrates that ")
        if better_algo == "PPO":
            f.write("PPO outperforms DQN in the chess environment, likely due to its ability to better handle large action spaces and complex state dynamics. ")
            f.write("PPO's policy-based approach allows it to directly learn a stochastic policy that can capture the nuances of chess strategy more effectively than DQN's value-based approach.\n\n")
        else:
            f.write("DQN outperforms PPO in the chess environment, possibly due to the specific reward structure and exploration strategy implemented. ")
            f.write("DQN's experience replay mechanism might be particularly effective at learning from the sparse rewards characteristic of chess.\n\n")
        
        f.write("For future work, exploring more sophisticated neural network architectures or incorporating chess-specific inductive biases into the models could further improve performance.")
    
    print(f"Comparison report generated at {report_path}")

def main():
    """Main function to run all visualizations and analysis"""
    os.makedirs("analysis", exist_ok=True)
    
    # Load hyperparameter tuning data
    tuning_data = load_tuning_data()
    
    # Evaluate models
    results = evaluate_models()
    
    if results:
        # Generate plots
        plot_cumulative_rewards(results)
        plot_training_stability(tuning_data)
        plot_reward_distributions(results)
        plot_game_outcomes(results)
        
        # Print hyperparameter summary
        print_hyperparameter_summary()
        
        # Generate comparison report
        generate_comparison_report(results, tuning_data)
    
if __name__ == "__main__":
    main()