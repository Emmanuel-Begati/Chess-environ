# ♟️ Chess Reinforcement Learning Agents

## 🚀 Introduction

This project demonstrates the application of Reinforcement Learning (RL) techniques to train intelligent agents to play chess effectively. We implemented two state-of-the-art RL algorithms—**Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)**—to develop agents capable of playing against each other and the renowned chess engine, Stockfish, at various skill levels (ELO ratings).

The agents operate within a custom Gymnasium environment integrated with **Python-chess** and **Stockfish**, learning to master chess through trial, error, and strategic reward shaping that evaluates material advantage, positional strength, and king safety.

## 📂 Project Structure

```
chess_rl_project/
├── environment/
│   ├── custom_chess_env.py       # Custom Gymnasium Chess environment
│   ├── rendering.py              # Chessboard visualization with OpenGL/PyGame
├── training/
│   ├── dqn_training.py           # DQN training script
│   ├── ppo_training.py           # PPO training script
├── models/
│   ├── dqn/                      # Saved DQN models
│   ├── ppo/                      # Saved PPO models
├── tuning_results/
│   ├── dqn_optuna_results/       # DQN hyperparameter tuning
│   └── ppo_optuna_results/       # PPO hyperparameter tuning
├── videos/
│   ├── dqn_vs_self.mp4           # DQN self-play
│   ├── dqn_vs_stockfish.mp4      # DQN vs Stockfish
│   ├── ppo_vs_self.mp4           # PPO self-play
│   └── ppo_vs_stockfish.mp4      # PPO vs Stockfish
├── visualization/
│   ├── screenshots/              # Chessboard visualization screenshots
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## ✨ Features

- ✅ Custom Gymnasium environment integrating with Python-chess and Stockfish
- ✅ State Space: 8x8x12 representation of chessboard
- ✅ Action Space: 4672 possible chess moves
- ✅ Two Reinforcement Learning algorithms implemented:
  - **Deep Q-Network (DQN)**
  - **Proximal Policy Optimization (PPO)**
- ✅ Hyperparameter optimization with **Optuna**
- ✅ Visualization using **OpenGL/PyGame**
- ✅ Gameplay capability against Stockfish engine (various ELO ratings)
- ✅ Reward structure evaluating material advantage, position, and king safety

## 🎬 Video Demonstrations

| Description                  | Video                                  |
|------------------------------|----------------------------------------|
| DQN Agent vs Self            | [Watch Video](videos/dqn_vs_self.mp4) |
| DQN Agent vs Stockfish       | [Watch Video](videos/dqn_vs_stockfish.mp4) |
| PPO Agent vs Self            | [Watch Video](videos/ppo_vs_self.mp4) |
| PPO Agent vs Stockfish       | [Watch Video](videos/ppo_vs_stockfish.mp4) |

These videos illustrate gameplay, showcasing agent decisions, chessboard states, and real-time metrics.

## 🎨 Visualizations

Chessboard visualization is implemented with OpenGL/PyGame, providing clear visual feedback of game states during agent training and evaluation. Screenshots below are from recorded gameplay videos:

![Chess Visualization](visualization/screenshots/chess_visual_1.png)
![Chess Visualization](visualization/screenshots/chess_visual_2.png)

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/chess_rl_project.git
cd chess_rl_project
pip install -r requirements.txt
```

## ⚙️ Usage

### Training Agents

```bash
# Train DQN agent
python training/dqn_training.py

# Train PPO agent
python training/ppo_training.py
```

### Hyperparameter Optimization

```bash
# Optimize DQN parameters
python tuning_results/dqn_optuna.py

# Optimize PPO parameters
python tuning_results/ppo_optuna.py
```

### Playing and Recording Games

```bash
# Record DQN agent playing Stockfish
python main.py --agent dqn --opponent stockfish --record

# Record PPO agent playing Stockfish
python main.py --agent ppo --opponent stockfish --record
```

## 📊 Results and Analysis

### Hyperparameter Optimization

| Algorithm | Optimal Hyperparameters                                | Best Reward |
|-----------|--------------------------------------------------------|-------------|
| **DQN**   | lr=0.00020578, gamma=0.9517, batch_size=32, exploration_fraction=0.2818, exploration_final_eps=0.2180, buffer_size=100000 | 28.45       |
| **PPO**   | lr=0.00092405, gamma=0.9519, batch_size=128, n_steps=1024, ent_coef=0.01824, gae_lambda=0.9415, clip_range=0.1387 | **36.60**   |

### Comparative Analysis
- **PPO** outperformed **DQN**, achieving higher and more stable rewards.
- PPO demonstrated faster convergence and greater generalization against Stockfish.
- DQN, while effective, showed greater variability and slower learning.

## 🤝 Acknowledgments
- [Stockfish Chess Engine](https://stockfishchess.org)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Python-chess](https://github.com/niklasf/python-chess)
- [Optuna](https://github.com/optuna/optuna)

## 📄 License

This project is licensed under the **MIT License**.

