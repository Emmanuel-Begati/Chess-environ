import gym
from environment.custom_env import ChessEnv
from training.dqn_training import train_dqn
from training.pg_training import train_pg

def main():
    # Create the chess environment
    env = ChessEnv()

    # Train DQN model
    print("Training DQN model...")
    dqn_model = train_dqn(env)
    
    # Train PPO model
    print("Training PPO model...")
    pg_model = train_pg(env)

    # Save the models
    dqn_model.save("models/dqn/dqn_model")
    pg_model.save("models/pg/pg_model")

    print("Training complete. Models saved.")

if __name__ == "__main__":
    main()