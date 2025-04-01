import os
import argparse
import numpy as np
import imageio
from stable_baselines3 import DQN, PPO
from environment.custom_env import ChessEnv
from training.dqn_training import train_dqn
from training.pg_training import train_ppo

def parse_args():
    parser = argparse.ArgumentParser(description='Chess RL Project')
    parser.add_argument('--train', choices=['dqn', 'ppo', 'both', 'none'], default='none',
                        help='Which algorithm to train')
    parser.add_argument('--play', action='store_true',
                        help='Play a game with a trained agent')
    parser.add_argument('--model', choices=['dqn', 'ppo', 'both'], default='dqn',
                        help='Which model to use for playing')
    parser.add_argument('--record', action='store_true',
                        help='Record a video of the gameplay')
    parser.add_argument('--stockfish', action='store_true',
                        help='Play against Stockfish engine')
    parser.add_argument('--stockfish_elo', type=int, default=1500,
                        help='Stockfish ELO rating (approximate difficulty)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to play/record')
    parser.add_argument('--compare', action='store_true',
                        help='Compare both models in a video')
    return parser.parse_args()

def record_video(env, model, video_path="videos/chess_gameplay.mp4", num_episodes=1, stockfish=False, stockfish_elo=1500):
    """
    Record a video of the agent playing chess
    
    Args:
        env: The chess environment
        model: The trained model (DQN or PPO)
        video_path: Path to save the video
        num_episodes: Number of episodes to record
        stockfish: Whether to play against Stockfish
        stockfish_elo: ELO rating for Stockfish
    """
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    frames = []
    wins = 0
    draws = 0
    losses = 0
    
    for episode in range(num_episodes):
        print(f"Recording episode {episode+1}/{num_episodes}")
        # Pass stockfish=True to env.reset() if playing against Stockfish
        obs, _ = env.reset(options={'stockfish': stockfish, 'stockfish_elo': stockfish_elo})
        done = False
        episode_reward = 0
        step_count = 0
        max_steps = 200  # Safety limit to prevent infinite games
        
        while not done and step_count < max_steps:
            # Render the current state
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)
            
            # Take action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Add debugging information
            print(f"Step {step_count}, Reward: {reward}, Done: {done}")
            if done:
                print(f"Game finished! Adding final frame.")
                # Add the final frame
                final_frame = env.render(mode='rgb_array')
                if final_frame is not None:
                    frames.append(final_frame)
        
        print(f"Episode {episode+1} finished with reward: {episode_reward}, steps: {step_count}")
        
        # Count outcomes
        if 'outcome' in info:
            if info['outcome'] == 'win':
                wins += 1
            elif info['outcome'] == 'draw':
                draws += 1
            else:
                losses += 1
        
        if step_count >= max_steps:
            print("Maximum steps reached, ending episode")
    
    print(f"Results: {wins} wins, {draws} draws, {losses} losses")
    
    if frames:
        print(f"Saving video with {len(frames)} frames to {video_path}")
        import imageio
        imageio.mimsave(video_path, frames, fps=3)
        print(f"Video saved to {video_path}")
    else:
        print("Warning: No frames were captured. Check your rendering function.")
    
    return wins, draws, losses

def compare_models_video(env, dqn_model, ppo_model, video_path="videos/model_comparison.mp4", 
                        num_episodes=1, stockfish=False, stockfish_elo=1500):
    """
    Create a side-by-side comparison video of DQN and PPO models
    """
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    dqn_frames = []
    ppo_frames = []
    
    # Record DQN games
    print("Recording DQN model gameplay...")
    for episode in range(num_episodes):
        obs, _ = env.reset(options={'stockfish': stockfish, 'stockfish_elo': stockfish_elo})
        done = False
        
        while not done:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                dqn_frames.append(frame)
            
            action, _ = dqn_model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
    
    # Record PPO games
    print("Recording PPO model gameplay...")
    for episode in range(num_episodes):
        obs, _ = env.reset(options={'stockfish': stockfish, 'stockfish_elo': stockfish_elo})
        done = False
        
        while not done:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                ppo_frames.append(frame)
            
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
    
    # Create side-by-side frames
    combined_frames = []
    max_frames = min(len(dqn_frames), len(ppo_frames))
    
    for i in range(max_frames):
        # Get frames from each model (or use last frame if one is shorter)
        dqn_frame = dqn_frames[min(i, len(dqn_frames)-1)]
        ppo_frame = ppo_frames[min(i, len(ppo_frames)-1)]
        
        # Create a side-by-side image
        h, w = dqn_frame.shape[0], dqn_frame.shape[1]
        combined = np.zeros((h, w*2, 3), dtype=np.uint8)
        combined[:, :w] = dqn_frame
        combined[:, w:] = ppo_frame
        
        # Add labels
        # (Note: This is a simple approach - for better text, use PIL or OpenCV)
        combined_frames.append(combined)
    
    print(f"Saving comparison video to {video_path}")
    imageio.mimsave(video_path, combined_frames, fps=3)
    print(f"Comparison video saved to {video_path}")

def main():
    args = parse_args()
    
    # Training
    if args.train in ['dqn', 'both']:
        print("Training DQN model...")
        train_dqn()
    
    if args.train in ['ppo', 'both']:
        print("Training PPO model...")
        train_ppo()
    
    # Playing or recording
    if args.play or args.record or args.compare:
        # Load models
        dqn_model = None
        ppo_model = None
        
        # Load DQN if needed
        if args.model in ['dqn', 'both']:
            dqn_path = "models/dqn/chess_dqn"
            if os.path.exists(dqn_path + ".zip"):
                dqn_model = DQN.load(dqn_path)
                print(f"Loaded DQN model from {dqn_path}")
            else:
                print(f"DQN model not found at {dqn_path}")
                if args.model == 'dqn':
                    return
        
        # Load PPO if needed
        if args.model in ['ppo', 'both']:
            ppo_path = "models/pg/ppo_chess"
            if os.path.exists(ppo_path + ".zip"):
                ppo_model = PPO.load(ppo_path)
                print(f"Loaded PPO model from {ppo_path}")
            else:
                print(f"PPO model not found at {ppo_path}")
                if args.model == 'ppo':
                    return
        
        # Initialize environment
        env = ChessEnv(render_mode='rgb_array' if args.record or args.compare else 'human')
        
        # Compare models
        if args.compare and dqn_model and ppo_model:
            compare_models_video(env, dqn_model, ppo_model, 
                               stockfish=args.stockfish, 
                               stockfish_elo=args.stockfish_elo,
                               num_episodes=args.episodes)
            return
        
        # Select appropriate model
        model = dqn_model if args.model == 'dqn' else ppo_model
        
        if args.record:
            # Record video
            model_name = args.model
            opponent = "stockfish" if args.stockfish else "self"
            video_path = f"videos/{model_name}_vs_{opponent}.mp4"
            record_video(env, model, video_path, args.episodes, 
                       stockfish=args.stockfish, 
                       stockfish_elo=args.stockfish_elo)
        else:
            # Play game without recording
            for episode in range(args.episodes):
                print(f"Playing episode {episode+1}/{args.episodes}")
                obs, _ = env.reset(options={'stockfish': args.stockfish, 
                                          'stockfish_elo': args.stockfish_elo})
                done = False
                total_reward = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    total_reward += reward
                    env.render()
                
                outcome = info.get('outcome', 'unknown')
                print(f"Episode {episode+1} finished with reward: {total_reward}, outcome: {outcome}")

if __name__ == "__main__":
    main()