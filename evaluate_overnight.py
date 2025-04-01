import os
import argparse
import numpy as np
import imageio
from stable_baselines3 import DQN, PPO
from environment.custom_env import ChessEnv

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate overnight training')
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'ppo'], default='dqn',
                        help='Algorithm that was trained (dqn or ppo)')
    parser.add_argument('--stockfish', action='store_true',
                        help='Evaluate against Stockfish')
    parser.add_argument('--stockfish_elo', type=int, default=1200,
                        help='Stockfish ELO rating (set lower for easier opponent)')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of evaluation games to play')
    return parser.parse_args()

def evaluate_overnight():
    args = parse_args()
    
    # Path to the overnight models
    overnight_path = f"models/{args.algorithm}/overnight"
    
    # Find the latest version
    versions = [v.split('v')[-1].split('.')[0] for v in os.listdir(overnight_path) 
                if v.startswith('v') and v.endswith('.zip')]
    versions = [int(v) for v in versions if v.isdigit()]
    
    if not versions:
        print("No trained versions found!")
        return
        
    latest_version = max(versions)
    print(f"Found {len(versions)} versions, latest is v{latest_version}")
    
    # Load the first and latest model to compare progress
    model_class = DQN if args.algorithm == 'dqn' else PPO
    
    try:
        first_model = model_class.load(f"{overnight_path}/v1.zip")
        latest_model = model_class.load(f"{overnight_path}/v{latest_version}.zip")
        print("Successfully loaded first and latest models")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Create environment
    env = ChessEnv(render_mode='rgb_array')
    
    # Evaluate against each other
    wins_latest = 0
    draws = 0
    wins_first = 0
    
    for game in range(args.games):
        print(f"Playing evaluation game {game+1}/{args.games}")
        
        if game % 2 == 0:
            # Latest model plays as white
            outcome = play_game(env, latest_model, first_model)
            if outcome > 0:
                wins_latest += 1
            elif outcome < 0:
                wins_first += 1
            else:
                draws += 1
        else:
            # First model plays as white
            outcome = play_game(env, first_model, latest_model)
            if outcome > 0:
                wins_first += 1
            elif outcome < 0:
                wins_latest += 1
            else:
                draws += 1
    
    print("\n=== Self-Play Evaluation Results ===")
    print(f"Latest model (v{latest_version}) wins: {wins_latest}")
    print(f"First model (v1) wins: {wins_first}")
    print(f"Draws: {draws}")
    print(f"Latest model win rate: {wins_latest/args.games:.2%}")
    
    # Evaluate against Stockfish if requested
    if args.stockfish:
        wins = 0
        draws = 0
        losses = 0
        
        for game in range(args.games):
            print(f"Playing against Stockfish (ELO {args.stockfish_elo}), game {game+1}/{args.games}")
            obs, _ = env.reset(options={'stockfish': True, 'stockfish_elo': args.stockfish_elo})
            done = False
            
            while not done:
                action, _ = latest_model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
            
            # Check outcome
            if 'outcome' in info:
                if info['outcome'] == 'win':
                    wins += 1
                elif info['outcome'] == 'draw':
                    draws += 1
                else:
                    losses += 1
        
        print("\n=== Stockfish Evaluation Results ===")
        print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
        print(f"Win rate against Stockfish (ELO {args.stockfish_elo}): {wins/args.games:.2%}")
    
    # Create a summary video of the latest model playing
    record_video(env, latest_model, f"videos/overnight_{args.algorithm}_v{latest_version}.mp4", 
                stockfish=True, stockfish_elo=args.stockfish_elo)

def play_game(env, white_model, black_model):
    """Play a game between two models and return outcome (1=white wins, -1=black wins, 0=draw)"""
    obs, _ = env.reset()
    done = False
    white_turn = True
    
    while not done:
        # Select model based on current turn
        model = white_model if white_turn else black_model
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Take action
        obs, reward, done, _, info = env.step(action)
        
        # Switch turns
        white_turn = not white_turn
    
    # Determine outcome
    if 'outcome' in info:
        if info['outcome'] == 'win':
            return 1 if white_turn else -1  # White wins if it's black's turn after game ends
        elif info['outcome'] == 'loss':
            return -1 if white_turn else 1
        elif info['outcome'] == 'draw':
            return 0
    
    return 0  # Default to draw

def record_video(env, model, video_path, num_episodes=1, stockfish=False, stockfish_elo=1200):
    """Record a video of the model playing"""
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    frames = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset(options={'stockfish': stockfish, 'stockfish_elo': stockfish_elo})
        done = False
        
        while not done:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
    
    if frames:
        print(f"Saving {len(frames)} frames to {video_path}")
        imageio.mimsave(video_path, frames, fps=3)
        print(f"Video saved to {video_path}")

if __name__ == "__main__":
    evaluate_overnight()