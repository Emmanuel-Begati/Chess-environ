import numpy as np
import chess
import os
import torch
from stable_baselines3 import DQN, PPO
from collections import deque
import pickle

class ChessMistakeTracker:
    """
    Tracks and analyzes chess mistakes to improve agent learning.
    
    This class identifies mistakes in gameplay, categorizes them, and
    helps the agent learn by providing additional training on problematic positions.
    """
    
    def __init__(self, capacity=1000):
        self.mistake_buffer = deque(maxlen=capacity)
        self.position_categories = {
            'tactical_blunders': [],
            'positional_errors': [],
            'endgame_mistakes': [],
            'opening_problems': [],
            'material_losses': []
        }
    
    def add_mistake(self, state, action, next_state, reward, done, info=None):
        """Add a mistake to the buffer."""
        # We consider it a mistake if:
        # 1. Agent lost material without compensation
        # 2. Made a move that led to checkmate
        # 3. Missed a winning tactic
        is_mistake = (
            reward < -1.0 or  # Significant negative reward
            (done and reward < 0)  # Game ended with loss
        )
        
        if is_mistake:
            # Store the experience
            self.mistake_buffer.append((state, action, next_state, reward, done, info))
            
            # Categorize the mistake
            category = self._categorize_mistake(state, action, next_state, reward, done, info)
            if category:
                self.position_categories[category].append((state, action))
    
    def _categorize_mistake(self, state, action, next_state, reward, done, info):
        """Categorize the type of mistake made."""
        # Basic categorization based on available information
        if done and reward < 0:
            return 'tactical_blunders'  # Lost by checkmate or resignation
        
        # Convert state back to board position for analysis
        board = self._state_to_board(state)
        if board:
            # Check game phase
            phase = self._determine_game_phase(board)
            
            # Material loss check
            material_before = self._calculate_material(board)
            board_after = self._state_to_board(next_state)
            if board_after:
                material_after = self._calculate_material(board_after)
                if material_before - material_after > 3:  # Lost a minor piece or more
                    return 'material_losses'
            
            # Phase-specific categorization
            if phase == 'opening' and board.fullmove_number <= 10:
                return 'opening_problems'
            elif phase == 'endgame':
                return 'endgame_mistakes'
            else:
                return 'positional_errors'
        
        return None
    
    def _state_to_board(self, state):
        """Convert NN input state back to chess board (approximate)."""
        # This is a simplified implementation - full implementation depends on your state representation
        try:
            # Assuming state is 8x8x12 with each layer representing a piece type
            board = chess.Board()
            board.clear()
            
            piece_map = {}
            piece_types = [
                chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING,
                chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
            ]
            
            for rank in range(8):
                for file in range(8):
                    for piece_idx in range(12):
                        if state[7-rank, file, piece_idx] > 0.5:  # Piece exists at this position
                            piece_color = chess.WHITE if piece_idx < 6 else chess.BLACK
                            piece_type = piece_types[piece_idx % 6]
                            square = chess.square(file, rank)
                            piece_map[square] = chess.Piece(piece_type, piece_color)
            
            board.set_piece_map(piece_map)
            return board
        except:
            return None
    
    def _determine_game_phase(self, board):
        """Determine the current phase of the chess game."""
        # Count total pieces to determine game phase
        num_pieces = len(board.piece_map())
        
        if num_pieces >= 26:  # All pieces minus a few pawns
            return 'opening'
        elif num_pieces >= 14:  # About half the pieces remain
            return 'middlegame'
        else:
            return 'endgame'
    
    def _calculate_material(self, board):
        """Calculate total material on the board."""
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        total = 0
        for piece in board.piece_map().values():
            total += piece_values[piece.piece_type]
        
        return total
    
    def get_training_positions(self, category=None, sample_size=32):
        """Get positions for focused training."""
        if category and category in self.position_categories:
            positions = self.position_categories[category]
        else:
            # Combine all categories with emphasis on recent mistakes
            positions = []
            for cat_positions in self.position_categories.values():
                positions.extend(cat_positions)
        
        if not positions:
            return []
        
        # Sample positions, with preference for recent ones
        indices = np.arange(len(positions))
        p = np.exp(indices / len(indices)) / np.sum(np.exp(indices / len(indices)))  # Exponential recency bias
        
        sample_indices = np.random.choice(
            indices, 
            size=min(sample_size, len(positions)), 
            replace=False,
            p=p
        )
        
        return [positions[i] for i in sample_indices]
    
    def save(self, filepath):
        """Save the mistake buffer and categories to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'buffer': self.mistake_buffer,
                'categories': self.position_categories
            }, f)
    
    def load(self, filepath):
        """Load mistake buffer and categories from disk."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.mistake_buffer = data['buffer']
                self.position_categories = data['categories']
            return True
        return False


class ChessSelfPlayTrainer:
    """
    Implements self-play training for chess reinforcement learning.
    
    The agent plays against versions of itself to learn from its own mistakes
    and continuously improve its playing strength.
    """
    
    def __init__(self, env, model_class=DQN, model_path="models", version_history=5):
        # Wrap the environment in a DummyVecEnv if it's not already vectorized
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        if not hasattr(env, 'reset_many') and not hasattr(env, 'step_wait'):
            # This is not a vectorized environment, so wrap it
            self.env = DummyVecEnv([lambda: env])
        else:
            # Already a vectorized environment
            self.env = env
            
        self.model_class = model_class
        self.model_path = model_path
        self.current_version = 0
        self.version_history = version_history
        self.mistake_tracker = ChessMistakeTracker()
        
        # Ensure model directory exists
        os.makedirs(model_path, exist_ok=True)
    
    def train(self, timesteps=10000, versions=5):
        """Train the model through multiple versions of self-play."""
        for version in range(versions):
            print(f"Training version {version + 1}/{versions}")
            
            # Load or create model
            if version == 0:
                # Initialize a new model for first version
                model = self.model_class("MlpPolicy", self.env, verbose=1, buffer_size=50000)  # Reduced buffer size
            else:
                try:
                    # Load previous version
                    model_path = f"{self.model_path}/v{version}.zip"
                    print(f"Loading model from {model_path}")
                    model = self.model_class.load(model_path)
                    model.set_env(self.env)  # Make sure the environment is set
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Creating a new model instead")
                    model = self.model_class("MlpPolicy", self.env, verbose=1, buffer_size=50000)
            
            # Train against previous versions (if available)
            try:
                self._train_against_previous_versions(model, version, timesteps)
            except Exception as e:
                print(f"Error in training against previous versions: {e}")
            
            # Save the new version
            save_path = f"{self.model_path}/v{version + 1}"
            print(f"Saving model to {save_path}")
            model.save(save_path)
            
            # Update current version
            self.current_version = version + 1
            
            # Evaluate against previous version
            if version > 0:
                try:
                    self._evaluate_progress(f"{self.model_path}/v{version + 1}.zip", 
                                         f"{self.model_path}/v{version}.zip")
                except Exception as e:
                    print(f"Error in evaluation: {e}")
    
    def _train_against_previous_versions(self, model, current_version, timesteps):
        """Train the current model against previous versions."""
        # First, regular training
        if model.env is None:
            # If the model's environment is None, set it to the trainer's environment
            model.set_env(self.env)
        
        # Now we can call learn
        model.learn(total_timesteps=int(timesteps * 0.7))  # 70% of training is standard learning
        
        # Then, train against previous versions
        remaining_timesteps = int(timesteps * 0.3)  # 30% for self-play
        versions_to_play_against = min(current_version, self.version_history)
        
        if versions_to_play_against > 0:
            timesteps_per_version = remaining_timesteps // versions_to_play_against
            
            for v in range(max(0, current_version - self.version_history), current_version):
                try:
                    opponent_path = f"{self.model_path}/v{v}.zip"
                    print(f"Loading opponent model from {opponent_path}")
                    opponent_model = self.model_class.load(opponent_path)
                    
                    # Play games against this version
                    self._self_play_training(model, opponent_model, timesteps_per_version)
                except Exception as e:
                    print(f"Error loading or playing against version {v}: {e}")
    
    def _self_play_training(self, model, opponent_model, timesteps):
        """Train the current model by playing against an opponent model."""
        # Use learning from mistakes
        steps_taken = 0
        while steps_taken < timesteps:
            obs = self.env.reset()
            done = False
            
            # Track the current game
            game_states = []
            game_actions = []
            game_rewards = []
            
            while not done:
                # Current model makes a move
                action, _ = model.predict(obs, deterministic=False)
                next_obs, rewards, dones, infos = self.env.step(action)
                
                # Extract values from the vectorized environment (always take index 0)
                reward = rewards[0]
                done = dones[0]
                info = infos[0] if infos else {}
                
                # Track this step
                game_states.append(obs[0])
                game_actions.append(action[0])
                game_rewards.append(reward)
                
                # Check for mistakes to learn from
                self.mistake_tracker.add_mistake(obs[0], action[0], next_obs[0], reward, done, info)
                
                obs = next_obs
                steps_taken += 1
                
                # If game not over, opponent model makes a move
                if not done:
                    # The environment handles the opponent move through step
                    pass
            
            # After game ends, learn from this game
            self._learn_from_game(model, game_states, game_actions, game_rewards)
    
    def _learn_from_game(self, model, states, actions, rewards):
        """Learn from a completed game, focusing on mistakes."""
        # For DQN, we'd add these experiences to the replay buffer
        # For PPO, we'd use this to update the policy
        
        # Example for DQN (if applicable):
        if isinstance(model, DQN) and hasattr(model, 'replay_buffer'):
            # Calculate returns (for simplicity, using raw rewards)
            returns = rewards
            
            # Add all experiences to replay buffer
            for i in range(len(states)):
                model.replay_buffer.add(
                    states[i],
                    actions[i],
                    returns[i],
                    states[i+1] if i+1 < len(states) else states[i],
                    done=(i+1 == len(states))
                )
    
    def _evaluate_progress(self, new_model_path, old_model_path, num_games=20):
        """Evaluate progress by comparing new model against old one."""
        new_model = self.model_class.load(new_model_path)
        old_model = self.model_class.load(old_model_path)
        
        wins = 0
        draws = 0
        losses = 0
        
        for game in range(num_games):
            # Play as white
            if game < num_games // 2:
                outcome = self._play_game(new_model, old_model)
            # Play as black
            else:
                outcome = self._play_game(old_model, new_model)
                # Invert outcome for the new model
                if outcome == 1:
                    outcome = -1
                elif outcome == -1:
                    outcome = 1
            
            if outcome == 1:
                wins += 1
            elif outcome == 0:
                draws += 1
            else:
                losses += 1
        
        print(f"Evaluation results against previous version:")
        print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
        print(f"Win rate: {wins/num_games:.2%}")
    
    def _play_game(self, white_model, black_model):
        """Play a game between two models and return the outcome."""
        obs = self.env.reset()
        done = False
        
        # Track which model is playing as white
        white_turn = True
        
        while not done:
            # Select model based on current turn
            model = white_model if white_turn else black_model
            
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take action
            obs, rewards, dones, infos = self.env.step(action)
            
            # Get results from the vectorized environment
            reward = rewards[0]
            done = dones[0]
            info = infos[0] if infos else {}
            
            # Switch turns
            white_turn = not white_turn
        
        # Determine outcome from white's perspective
        if 'outcome' in info:
            if info['outcome'] == 'win':
                return 1 if white_turn else -1  # White wins if it's black's turn after game ends
            elif info['outcome'] == 'loss':
                return -1 if white_turn else 1  # White loses if it's black's turn after game ends
            elif info['outcome'] == 'draw':
                return 0
        
        # Default to draw if outcome not clear
        return 0