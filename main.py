import os
import argparse
import imageio
import time
from stable_baselines3 import DQN, PPO
from environment.custom_env import ChessEnv
from training.mistake_learning import ChessSelfPlayTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Chess RL Project')
    parser.add_argument('--train', action='store_true',
                        help='Train the chess model with human-like techniques')
    parser.add_argument('--model_type', choices=['dqn', 'ppo'], default='dqn',
                        help='Which model type to use (DQN or PPO)')
    parser.add_argument('--play', action='store_true',
                        help='Play a game with the trained model')
    parser.add_argument('--record', action='store_true',
                        help='Record a video of the gameplay')
    parser.add_argument('--stockfish', action='store_true',
                        help='Play against Stockfish engine')
    parser.add_argument('--stockfish_elo', type=int, default=1500,
                        help='Stockfish ELO rating (approximate difficulty)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to play/record')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Number of timesteps to train for')
    parser.add_argument('--versions', type=int, default=5,
                        help='Number of model versions to train')
    parser.add_argument('--overnight', action='store_true',
                        help='Run extended overnight training session')
    parser.add_argument('--hours', type=int, default=8,
                        help='Number of hours to train (for overnight mode)')
    return parser.parse_args()

def record_video(env, model, video_path="videos/chess_gameplay.mp4", num_episodes=1, stockfish=False, stockfish_elo=1500):
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

def overnight_training(model_class, model_path, hours=8, versions=20, timesteps_per_version=20000):
    """Run extended overnight training with human-like techniques"""
    print(f"Starting overnight training for approximately {hours} hours")
    
    # Create environment
    env = ChessEnv()
    
    # Configure trainer
    trainer = ChessSelfPlayTrainer(
        env=env,
        model_class=model_class,
        model_path=model_path,
        version_history=5
    )
    
    start_time = time.time()
    end_time = start_time + (hours * 3600)  # Convert hours to seconds
    
    # Train for specified number of versions or until time runs out
    for version in range(versions):
        current_time = time.time()
        if current_time >= end_time:
            print(f"Training time limit of {hours} hours reached after {version} versions")
            break
            
        # Calculate remaining time
        time_elapsed = (current_time - start_time) / 3600  # in hours
        time_remaining = hours - time_elapsed
        
        print(f"Starting version {version+1}/{versions}")
        print(f"Time elapsed: {time_elapsed:.2f} hours, remaining: {time_remaining:.2f} hours")
        
        # Train this version with human-like learning
        try:
            # Reduce timesteps if we're running out of time
            if time_remaining < 1 and versions - version > 1:
                adjusted_timesteps = int(timesteps_per_version * (time_remaining / (versions - version)))
                print(f"Time running low, reducing timesteps to {adjusted_timesteps}")
                trainer.train_human_like(timesteps=adjusted_timesteps, versions=1)
            else:
                trainer.train_human_like(timesteps=timesteps_per_version, versions=1)
                
        except Exception as e:
            print(f"Error during training version {version+1}: {e}")
            continue
    
    # Calculate and print final stats
    total_time = (time.time() - start_time) / 3600
    print(f"Training completed after {total_time:.2f} hours")
    
    # Save a final copy to the standard path for compatibility
    if model_class == DQN:
        standard_path = "models/dqn/chess_dqn"
    else:
        standard_path = "models/pg/ppo_chess"
    
    final_model = model_class.load(f"{model_path}/v{trainer.current_version}.zip")
    final_model.save(standard_path)
    print(f"Saved final model to standard path: {standard_path}")

def main():
    args = parse_args()
    
    # Select the appropriate model class based on user preference
    model_class = DQN if args.model_type == 'dqn' else PPO
    model_type_name = args.model_type.upper()
    
    # Set up paths for the model - following the original file structure
    if args.model_type == 'dqn':
        model_dir = "models/dqn"
        human_like_path = "models/dqn/human_like"
        standard_path = "models/dqn/chess_dqn"
    else:
        model_dir = "models/pg"
        human_like_path = "models/pg/human_like"
        standard_path = "models/pg/ppo_chess"
    
    # Make sure model directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(human_like_path, exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    
    # Training with human-like techniques
    if args.train:
        if args.overnight:
            # Extended overnight training
            overnight_training(
                model_class=model_class,
                model_path=human_like_path,
                hours=args.hours,
                versions=args.versions,
                timesteps_per_version=args.timesteps
            )
        else:
            print(f"Training {model_type_name} model with human-like techniques...")
            
            # Create the base environment
            env = ChessEnv()
            
            # Use the human-like training method
            trainer = ChessSelfPlayTrainer(env, model_class=model_class, model_path=human_like_path)
            trainer.train_human_like(timesteps=args.timesteps, versions=args.versions)
            
            # Save a final copy to the standard path for compatibility
            final_model = model_class.load(f"{human_like_path}/v{trainer.current_version}.zip")
            final_model.save(standard_path)
            print(f"Saved final human-like {model_type_name} model to standard path: {standard_path}")
    
    # Playing or recording
    if args.play or args.record:
        # Load the model from the standard path
        if os.path.exists(standard_path + ".zip"):
            model = model_class.load(standard_path)
            print(f"Loaded {model_type_name} model from {standard_path}")
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