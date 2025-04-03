import os
import argparse
import imageio
import time
from stable_baselines3 import DQN, PPO
from environment.custom_env import ChessEnv
from training.dqn_training import train_dqn
from training.pg_training import train_ppo

def parse_args():
    parser = argparse.ArgumentParser(description='Chess RL Project')
    parser.add_argument('--train', action='store_true',
                        help='Train the chess model')
    parser.add_argument('--model_type', choices=['dqn', 'ppo'], default='dqn',
                        help='Which model type to use (DQN or PPO)')
    parser.add_argument('--play', action='store_true',
                        help='Play a game with the trained model')
    parser.add_argument('--record', action='store_true',
                        help='Record a video of the gameplay')
    parser.add_argument('--stockfish', action='store_true',
                        help='Play against Stockfish engine')
    parser.add_argument('--stockfish_elo', type=int, default=1200,
                        help='Stockfish ELO rating (approximate difficulty)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to play/record')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Number of timesteps to train for')
    # Hyperparameter optimization
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize hyperparameters before training')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of trials for hyperparameter optimization')
    # DQN specific hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--buffer_size', type=int, default=200000,
                        help='Size of replay buffer (default: 200000)')
    parser.add_argument('--exp_fraction', type=float, default=0.6,
                        help='Exploration fraction (default: 0.6)')
    parser.add_argument('--final_eps', type=float, default=0.15,
                        help='Final exploration epsilon (default: 0.15)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    # PPO specific hyperparameters
    parser.add_argument('--n_steps', type=int, default=1024,
                      help='Number of steps per PPO update (default: 1024)')
    parser.add_argument('--ent_coef', type=float, default=0.03,
                      help='Entropy coefficient for PPO (default: 0.03)')
    return parser.parse_args()

def record_video(env, model, video_path="videos/chess_gameplay.mp4", num_episodes=1, stockfish=False, stockfish_elo=1200):
    """Record a video of the agent playing chess"""
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    frames = []
    wins = 0
    draws = 0
    losses = 0
    
    for episode in range(num_episodes):
        print(f"Recording episode {episode+1}/{num_episodes}")
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
    
    print(f"Results: {wins} wins, {draws} draws, {losses} losses")
    
    if frames:
        print(f"Saving video with {len(frames)} frames to {video_path}")
        imageio.mimsave(video_path, frames, fps=3)
        print(f"Video saved to {video_path}")
    else:
        print("Warning: No frames were captured. Check your rendering function.")

def main():
    args = parse_args()
    
    # Set up paths for the model
    if args.model_type == 'dqn':
        model_dir = "models/dqn"
        standard_path = "models/dqn/chess_dqn"
    else:
        model_dir = "models/pg"
        standard_path = "models/pg/ppo_chess"
    
    # Make sure model directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    
    # Training
    if args.train:
        if args.model_type == 'dqn':
            print(f"Training DQN chess model...")
            # Call the DQN training module with optimization option
            train_dqn(
                total_timesteps=args.timesteps,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                exploration_fraction=args.exp_fraction,
                exploration_final_eps=args.final_eps,
                buffer_size=args.buffer_size,
                optimize=args.optimize,
                n_trials=args.n_trials
            )
            print(f"DQN training complete. Model saved to {standard_path}")
        else:
            print(f"Training PPO chess model...")
            # Call the PPO training module with optimization option
            train_ppo(
                total_timesteps=args.timesteps,
                learning_rate=args.lr,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                ent_coef=args.ent_coef,
                optimize=args.optimize,
                n_trials=args.n_trials
            )
            print(f"PPO training complete. Model saved to {standard_path}")
    
    # Playing or recording
    if args.play or args.record:
        # Load the model from the standard path
        model_class = DQN if args.model_type == 'dqn' else PPO
        model_type_name = args.model_type.upper()
        
        if os.path.exists(standard_path + ".zip"):
            model = model_class.load(standard_path)
            print(f"Loaded {model_type_name} model from {standard_path}")
            
            # Load and print hyperparameters if available
            hyperparams_path = os.path.join(os.path.dirname(standard_path), "hyperparameters.json")
            if os.path.exists(hyperparams_path):
                import json
                with open(hyperparams_path, "r") as f:
                    hyperparams = json.load(f)
                print(f"Model trained with the following hyperparameters:")
                for param, value in hyperparams.items():
                    print(f"  {param}: {value}")
        else:
            print(f"{model_type_name} model not found at {standard_path}")
            return
        
        # Initialize environment
        env = ChessEnv(render_mode='rgb_array' if args.record else 'human')
        
        if args.record:
            # Record video
            opponent = "stockfish" if args.stockfish else "self"
            video_path = f"videos/{args.model_type}_vs_{opponent}.mp4"
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