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
                
                # Check for hanging pieces (pieces that could be captured next move)
                if self._has_hanging_pieces(board_after):
                    return 'tactical_blunders'
                    
                # Check for missed checkmate opportunities
                if self._missed_checkmate(board, board_after):
                    return 'tactical_blunders'
                
                # Check for repeated moves
                if self._is_repetitive_play(board):
                    return 'positional_errors'
            
            # Phase-specific categorization
            if phase == 'opening' and board.fullmove_number <= 10:
                if not self._good_opening_principles(board):
                    return 'opening_problems'
                
            elif phase == 'endgame':
                if self._poor_endgame_technique(board):
                    return 'endgame_mistakes'
            else:
                return 'positional_errors'
        
        return None

    def _has_hanging_pieces(self, board):
        """Check if any pieces are hanging (can be captured without compensation)."""
        # Simple implementation: check if any piece can be captured by a lower value piece
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                continue
                
            # Get attackers
            attackers = board.attackers(not piece.color, square)
            defenders = board.attackers(piece.color, square)
            
            # If the piece is attacked and not defended, it's hanging
            if attackers and not defenders:
                return True
                
            # If the piece is attacked by a lower value piece, it might be hanging
            if attackers:
                min_attacker_value = min(piece_values[board.piece_at(s).piece_type] for s in attackers)
                if min_attacker_value < piece_values[piece.piece_type]:
                    return True
                    
        return False

    def _missed_checkmate(self, board_before, board_after):
        """Check if there was a checkmate opportunity that was missed."""
        # Check if any legal move in the previous position would lead to checkmate
        for move in board_before.legal_moves:
            test_board = board_before.copy()
            test_board.push(move)
            if test_board.is_checkmate():
                return True
        return False

    def _is_repetitive_play(self, board):
        """Check if the same piece is being moved repeatedly."""
        if len(board.move_stack) < 6:
            return False
            
        # Get last 6 moves
        last_moves = board.move_stack[-6:]
        
        # Count how many times each piece was moved
        piece_moves = {}
        for move in last_moves:
            from_square = move.from_square
            if from_square not in piece_moves:
                piece_moves[from_square] = 1
            else:
                piece_moves[from_square] += 1
        
        # If any piece was moved more than twice in the last 6 moves, it's repetitive
        return any(count > 2 for count in piece_moves.values())

    def _good_opening_principles(self, board):
        """Check if opening principles are being followed."""
        # Common opening principles:
        # 1. Develop knights and bishops
        # 2. Control the center
        # 3. Castle early
        # 4. Connect rooks
        
        color = chess.WHITE if board.turn == chess.WHITE else chess.BLACK
        
        # Check piece development
        knights_developed = 0
        bishops_developed = 0
        
        # Knight starting positions
        knight_start = [chess.B1, chess.G1] if color == chess.WHITE else [chess.B8, chess.G8]
        for sq in knight_start:
            piece = board.piece_at(sq)
            if piece is None or piece.piece_type != chess.KNIGHT or piece.color != color:
                knights_developed += 1
        
        # Bishop starting positions
        bishop_start = [chess.C1, chess.F1] if color == chess.WHITE else [chess.C8, chess.F8]
        for sq in bishop_start:
            piece = board.piece_at(sq)
            if piece is None or piece.piece_type != chess.BISHOP or piece.color != color:
                bishops_developed += 1
        
        # Check if king is castled
        king_square = board.king(color)
        if color == chess.WHITE:
            castled = king_square in [chess.G1, chess.C1]
        else:
            castled = king_square in [chess.G8, chess.C8]
        
        # Simplified scoring
        score = knights_developed + bishops_developed + (2 if castled else 0)
        return score >= 3  # At least 3 development actions for good opening
    
    def _poor_endgame_technique(self, board):
        """Simplified check for poor endgame technique"""
        return False  # Placeholder implementation

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
    Implements self-play training for chess reinforcement learning with human-like behavior.
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
    
    def train_human_like(self, timesteps=10000, versions=5):
        """
        Train the model to play more like a human by focusing on:
        1. Opening principles
        2. Using all pieces
        3. Avoiding repetitive moves
        4. Learning from common chess mistakes
        """
        print("Starting human-like chess training...")
        
        # Initial model setup
        if not os.path.exists(f"{self.model_path}/v1.zip"):
            # Create and save initial model
            model = self.model_class("MlpPolicy", self.env, verbose=1, buffer_size=50000)
            model.save(f"{self.model_path}/v1")
            print("Created initial model")
            self.current_version = 1
        
        for version in range(1, versions + 1):
            print(f"Training version {version}/{versions}")
            
            # Load current model
            model = self.model_class.load(f"{self.model_path}/v{version}.zip")
            model.set_env(self.env)
            
            # Standard training phase (70%)
            print("Phase 1: Standard training")
            model.learn(total_timesteps=int(timesteps * 0.7))
            
            # Human-like correction phase (30%)
            print("Phase 2: Human-like correction")
            self._train_human_principles(model, int(timesteps * 0.15))
            self._learn_from_mistakes(model, int(timesteps * 0.15))
            
            # Save the improved model
            next_version = version + 1
            model.save(f"{self.model_path}/v{next_version}")
            print(f"Saved model version {next_version}")
            self.current_version = next_version
            
            # Evaluate against previous version
            if version > 1:
                self._evaluate_progress(
                    f"{self.model_path}/v{next_version}.zip", 
                    f"{self.model_path}/v{version}.zip",
                    num_games=10
                )
        
        print("Human-like training complete!")

    def _train_human_principles(self, model, timesteps):
        """Train specifically on human chess principles"""
        steps_taken = 0
        
        while steps_taken < timesteps:
            obs = self.env.reset()
            done = False
            
            # Inject structured chess knowledge
            # 1. Opening phase - focus on development
            opening_phase = True
            middlegame_phase = False
            endgame_phase = False
            move_count = 0
            
            while not done:
                # Current state analysis
                if opening_phase and move_count >= 10:
                    opening_phase = False
                    middlegame_phase = True
                    
                # Get action
                action, _ = model.predict(obs, deterministic=False)
                next_obs, rewards, dones, infos = self.env.step(action)
                
                # Process transition
                reward = rewards[0]
                done = dones[0]
                info = infos[0] if infos else {}
                
                # Add custom reward shaping for human principles
                if opening_phase:
                    # In opening, prioritize development and center control
                    obs_board = self._get_board_from_obs(obs[0])
                    if obs_board:
                        developed_pieces = self._count_developed_pieces(obs_board)
                        center_control = self._evaluate_center_control(obs_board)
                        
                        # Add this data to help learning
                        if hasattr(model, 'replay_buffer'):
                            # Modify rewards to emphasize opening principles
                            shaped_reward = reward + (developed_pieces * 0.1) + (center_control * 0.1)
                            model.replay_buffer.add(obs[0], action[0], shaped_reward, next_obs[0], done)
                
                obs = next_obs
                steps_taken += 1
                move_count += 1
                
        return steps_taken

    def _learn_from_mistakes(self, model, timesteps):
        """Learn specifically from common chess mistakes"""
        # Generate or load common chess mistakes
        mistake_positions = self._get_common_mistakes()
        
        if not mistake_positions:
            print("No mistake positions available for training")
            return 0
        
        # Train on these mistakes
        steps_taken = 0
        while steps_taken < timesteps:
            # Sample a mistake position
            mistake = np.random.choice(mistake_positions)
            state, correct_action = mistake
            
            # Convert to environment state
            self.env.reset()
            self.env.envs[0].board = state  # Access the unwrapped environment
            
            # Get current prediction
            obs = self.env.envs[0]._get_obs()  # Access the unwrapped environment
            current_action, _ = model.predict(obs.reshape(1, *obs.shape), deterministic=True)
            current_action = current_action[0]
            
            # If the model's action differs from correct action, train on this example
            if current_action != correct_action:
                # For DQN, add to replay buffer with high priority
                if hasattr(model, 'replay_buffer'):
                    # Simulate taking the wrong action and show the negative outcome
                    test_board = state.copy()
                    wrong_move = self.env.envs[0]._action_to_move(current_action)
                    test_board.push(wrong_move)
                    
                    # Now simulate taking the correct action
                    right_board = state.copy()
                    right_move = self.env.envs[0]._action_to_move(correct_action)
                    right_board.push(right_move)
                    
                    # Compare outcomes to get reward signal
                    wrong_eval = self._simple_evaluate_position(test_board)
                    right_eval = self._simple_evaluate_position(right_board)
                    
                    # Calculate special reward for learning
                    corrective_reward = right_eval - wrong_eval + 1.0  # Add bonus for correct move
                    
                    # Add to replay buffer (with higher weight)
                    model.replay_buffer.add(obs, correct_action, corrective_reward, obs, False)
                    
            steps_taken += 1
                
        return steps_taken

    def _count_developed_pieces(self, board):
        """Count how many pieces have been developed from their starting positions"""
        # Simple implementation focusing on knights and bishops
        white_dev = 0
        black_dev = 0
        
        # Check knight development
        if board.piece_at(chess.B1) is None or board.piece_at(chess.B1).piece_type != chess.KNIGHT:
            white_dev += 1
        if board.piece_at(chess.G1) is None or board.piece_at(chess.G1).piece_type != chess.KNIGHT:
            white_dev += 1
        if board.piece_at(chess.B8) is None or board.piece_at(chess.B8).piece_type != chess.KNIGHT:
            black_dev += 1
        if board.piece_at(chess.G8) is None or board.piece_at(chess.G8).piece_type != chess.KNIGHT:
            black_dev += 1
            
        # Check bishop development
        if board.piece_at(chess.C1) is None or board.piece_at(chess.C1).piece_type != chess.BISHOP:
            white_dev += 1
        if board.piece_at(chess.F1) is None or board.piece_at(chess.F1).piece_type != chess.BISHOP:
            white_dev += 1
        if board.piece_at(chess.C8) is None or board.piece_at(chess.C8).piece_type != chess.BISHOP:
            black_dev += 1
        if board.piece_at(chess.F8) is None or board.piece_at(chess.F8).piece_type != chess.BISHOP:
            black_dev += 1
        
        # Return development for the side to move
        return white_dev if board.turn == chess.WHITE else black_dev

    def _get_common_mistakes(self):
        """Get a list of common chess mistakes for training"""
        # First check if we have recorded mistakes from actual play
        if sum(len(m) for m in self.mistake_tracker.position_categories.values()) > 20:
            # Use recorded mistakes
            all_mistakes = []
            for category, mistakes in self.mistake_tracker.position_categories.items():
                all_mistakes.extend(mistakes)
            return all_mistakes
        
        # Otherwise, return some predefined common mistakes
        # Each entry is (board_state, correct_action)
        common_mistakes = []
        
        # Example: Don't hang a piece
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        board.push_san("Nc6")
        board.push_san("Bc4")  # Position after 3.Bc4
        
        try:
            # Instead of defending f7, black blunders with d6
            correct_move = board.parse_san("Nf6")
            correct_action = list(board.legal_moves).index(correct_move)
            common_mistakes.append((board.copy(), correct_action))
        except:
            pass  # Skip if move parsing fails
        
        # Example: Don't miss a fork
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        board.push_san("Nc6")
        board.push_san("d4")  # Position where white can fork with d4
        
        try:
            correct_move = board.parse_san("exd4")
            correct_action = list(board.legal_moves).index(correct_move)
            common_mistakes.append((board.copy(), correct_action))
        except:
            pass  # Skip if move parsing fails
        
        return common_mistakes

    def _get_board_from_obs(self, obs):
        """
        Convert an observation back to a chess board.
        This bridges the gap between neural network input and chess positions.
        
        Args:
            obs: The observation from the environment (neural network input format)
            
        Returns:
            chess.Board: A chess board representing the observation, or None if conversion fails
        """
        # Create a fresh board
        board = chess.Board()
        board.clear()
        
        try:
            # Reshape if necessary (assuming obs is 8x8x12 or flattened version)
            if len(obs.shape) == 1:
                # Assuming shape is (8*8*12,)
                obs_reshaped = obs.reshape(8, 8, 12)
            else:
                obs_reshaped = obs
                
            # Map piece types and positions
            piece_map = {}
            piece_types = [
                chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING,
                chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
            ]
            
            for rank in range(8):
                for file in range(8):
                    for piece_idx in range(12):
                        # Check if this piece exists at this position (value > 0.5)
                        if obs_reshaped[7-rank, file, piece_idx] > 0.5:
                            piece_color = chess.WHITE if piece_idx < 6 else chess.BLACK
                            piece_type = piece_types[piece_idx]
                            square = chess.square(file, rank)
                            piece_map[square] = chess.Piece(piece_type, piece_color)
            
            # Set the piece map on the board
            board.set_piece_map(piece_map)
            
            return board
            
        except Exception as e:
            print(f"Error converting observation to board: {e}")
            return None

    def _evaluate_center_control(self, board):
        """
        Evaluate the level of center control in a chess position.
        
        Args:
            board: A chess.Board representing the position
            
        Returns:
            float: A score between -1 and 1 representing center control
                  (positive for white advantage, negative for black)
        """
        # Define center squares
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        
        # Evaluate control of center squares
        white_control = 0
        black_control = 0
        
        # For each center square, count attackers from each side
        for square in center_squares:
            white_attackers = len(board.attackers(chess.WHITE, square))
            black_attackers = len(board.attackers(chess.BLACK, square))
            
            # Award points for occupation
            piece = board.piece_at(square)
            if piece is not None:
                if piece.color == chess.WHITE:
                    white_control += 1
                else:
                    black_control += 1
                    
            # Award fractional points for attacking
            white_control += 0.5 * white_attackers
            black_control += 0.5 * black_attackers
        
        # Normalize to a -1 to 1 range
        total_control = max(1, white_control + black_control)  # Avoid division by zero
        normalized_control = (white_control - black_control) / total_control
        
        return normalized_control

    def _simple_evaluate_position(self, board):
        """
        Simple position evaluation for comparing moves.
        
        Args:
            board: A chess.Board representing the position
            
        Returns:
            float: An evaluation score where positive favors white and negative favors black
        """
        if board.is_checkmate():
            return -100 if board.turn else 100
            
        material_balance = 0
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = piece_values[piece.piece_type]
                material_balance += value if piece.color == chess.WHITE else -value
        
        # Add a small bonus for center control
        center_bonus = 0.1 * self._evaluate_center_control(board)
        material_balance += center_bonus
        
        # Convert to perspective of side to move
        if board.turn == chess.BLACK:
            material_balance = -material_balance
            
        return material_balance