# Chess Reinforcement Learning Project

This project implements a reinforcement learning agent to play chess using Gymnasium and Stable Baselines3. The agent is trained using two different algorithms: Deep Q-Network (DQN) and Proximal Policy Optimization (PPO). The project includes a custom Gymnasium environment for chess, visualization of the chessboard, and training scripts for both algorithms.

## Project Structure

```
chess_rl
├── environment
│   ├── custom_env.py        # Custom Gymnasium environment for chess
│   └── rendering.py         # Visualization of the chessboard using PyOpenGL
├── training
│   ├── dqn_training.py      # Training script for DQN using Stable Baselines3
│   └── pg_training.py       # Training script for PPO using Stable Baselines3
├── models
│   ├── dqn                  # Folder to save trained DQN models
│   │   └── .gitkeep
│   └── pg                   # Folder to save trained PPO models
│       └── .gitkeep
├── main.py                  # Entry point to run experiments
├── requirements.txt         # Dependencies for the project
└── README.md                # Documentation
```

## Installation

To set up the project, you need to install the required dependencies. You can do this using pip:

```bash
pip install -r requirements.txt
```

## Training the Models

To train the models, run the `main.py` script. This script will load the chess environment, execute the training for both DQN and PPO algorithms, and save the trained models.

```bash
python main.py
```

## Visualizing Results

After training, you can visualize the results using the rendering capabilities provided in the `environment/rendering.py` file. This will allow you to see the chessboard and the moves made by the agent.

## Notes

- Ensure that you have the necessary libraries installed, including `gymnasium`, `stable-baselines3`, `python-chess`, `pygame`, `PyOpenGL`, `numpy`, and `matplotlib`.
- The project is structured to facilitate easy comparison between the two reinforcement learning algorithms.

For more detailed information on each component, please refer to the respective files in the project.# Chess-environ
