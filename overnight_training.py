import os
import time
import argparse
from environment.custom_env import ChessEnv
from training.mistake_learning import ChessSelfPlayTrainer
from stable_baselines3 import DQN, PPO

def parse_args():
    parser = argparse.ArgumentParser(description='Chess overnight training')
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'ppo'], default='dqn',
                        help='Algorithm to use for training (dqn or ppo)')
    parser.add_argument('--hours', type=float, default=8.0,
                        help='Number of hours to train')
    parser.add_argument('--versions', type=int, default=20,
                        help='Maximum number of model versions to create')
    parser.add_argument('--timesteps_per_version', type=int, default=20000,
                        help='Number of timesteps to train each version')
    return parser.parse_args()

def overnight_training():
    args = parse_args()
    print(f"Starting overnight training with {args.algorithm.upper()} for approximately {args.hours} hours")
    
    # Create environment
    env = ChessEnv()
    
    # Set up the trainer with appropriate model class
    model_class = DQN if args.algorithm == 'dqn' else PPO
    output_path = f"models/{args.algorithm}/overnight"
    
    # Configure with smaller buffer to avoid memory issues
    trainer = ChessSelfPlayTrainer(
        env=env,
        model_class=model_class,
        model_path=output_path,
        version_history=5  # Keep last 5 versions for self-play
    )
    
    start_time = time.time()
    end_time = start_time + (args.hours * 3600)  # Convert hours to seconds
    
    # Train for specified number of versions or until time runs out
    for version in range(args.versions):
        current_time = time.time()
        if current_time >= end_time:
            print(f"Training time limit of {args.hours} hours reached after {version} versions")
            break
            
        # Calculate remaining time
        time_elapsed = (current_time - start_time) / 3600  # in hours
        time_remaining = args.hours - time_elapsed
        versions_remaining = args.versions - version
        
        print(f"Starting version {version+1}/{args.versions}")
        print(f"Time elapsed: {time_elapsed:.2f} hours, remaining: {time_remaining:.2f} hours")
        print(f"Expected completion in: {time_remaining * versions_remaining / (version+1):.2f} hours")
        
        # Train this version
        try:
            # Reduce timesteps if we're running out of time
            if time_remaining < 1 and versions_remaining > 1:
                # Less than 1 hour left, reduce timesteps to finish
                adjusted_timesteps = int(args.timesteps_per_version * (time_remaining / versions_remaining))
                print(f"Time running low, reducing timesteps to {adjusted_timesteps}")
                trainer.train(timesteps=adjusted_timesteps, versions=1)
            else:
                trainer.train(timesteps=args.timesteps_per_version, versions=1)
                
            # Periodically save the mistake tracker
            if version % 5 == 0:
                tracker_path = f"{output_path}/mistake_tracker_v{version}.pkl"
                trainer.mistake_tracker.save(tracker_path)
                print(f"Saved mistake tracker to {tracker_path}")
                
        except Exception as e:
            print(f"Error during training version {version+1}: {e}")
            # Continue to next version if there's an error
            continue
    
    # Save final mistake tracker
    tracker_path = f"{output_path}/mistake_tracker_final.pkl"
    trainer.mistake_tracker.save(tracker_path)
    
    # Calculate and print final stats
    total_time = (time.time() - start_time) / 3600
    print(f"Training completed after {total_time:.2f} hours")
    print(f"Trained {trainer.current_version} versions")
    print(f"Final model saved at: {output_path}/v{trainer.current_version}.zip")

if __name__ == "__main__":
    overnight_training()