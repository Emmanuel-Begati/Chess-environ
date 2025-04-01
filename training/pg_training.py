import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import ChessEnv

def train_ppo():
    # Create the custom chess environment
    env = ChessEnv()
    env = DummyVecEnv([lambda: env])  # Wrap the environment

    # Define the PPO model
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, n_steps=2048, gamma=0.99)

    # Train the model
    model.learn(total_timesteps=100000)

    # Save the model
    model.save("models/pg/ppo_chess")

if __name__ == "__main__":
    train_ppo()