import os
from environment.custom_env import ChessEnv
from training.mistake_learning import ChessSelfPlayTrainer
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    # Create environment
    env = ChessEnv()
    
    # Create trainer with reduced buffer size to avoid memory issues
    trainer = ChessSelfPlayTrainer(
        env=env,
        model_class=DQN,
        model_path="models/test_mistake_learning",
        version_history=3
    )
    
    # Reduce the training parameters to fit your system's memory
    print("Starting training with smaller parameters...")
    trainer.train(timesteps=5000, versions=2)
    
    print("Training complete!")

if __name__ == "__main__":
    main()