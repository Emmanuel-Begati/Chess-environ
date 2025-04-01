import gymnasium as gym
import numpy as np
import chess
import chess.engine
import os

class ChessEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = gym.spaces.Discrete(4672)  # Number of possible moves in chess
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)
        self.render_mode = render_mode
        self.stockfish = None
        self.stockfish_elo = 1500
        self.play_as_white = True  # By default, RL agent plays as white
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        
        options = options or {}
        self.play_against_stockfish = options.get('stockfish', False)
        self.stockfish_elo = options.get('stockfish_elo', 1500)
        
        # Setup Stockfish if needed
        if self.play_against_stockfish and self.stockfish is None:
            try:
                self.stockfish = chess.engine.SimpleEngine.popen_uci("stockfish")
                # Set Stockfish playing strength (approximate ELO)
                self.stockfish.configure({"Skill Level": min(20, max(0, (self.stockfish_elo - 1000) // 100))})
            except Exception as e:
                print(f"Failed to initialize Stockfish: {e}")
                self.play_against_stockfish = False
        
        # If playing against Stockfish and Stockfish goes first (agent is black)
        if self.play_against_stockfish and not self.play_as_white:
            self._make_stockfish_move()
            
        return self._get_obs(), {}
    
    def step(self, action):
        move = self._action_to_move(action)
        if move not in self.board.legal_moves:
            return self._get_obs(), -10, True, False, {"error": "Illegal move", "outcome": "loss"}
        
        # Make agent's move
        self.board.push(move)
        
        # Check if game is over after agent's move
        if self.board.is_game_over():
            return self._get_obs(), self._get_reward(), True, False, self._get_info()
        
        # If playing against Stockfish, make its move
        if self.play_against_stockfish:
            self._make_stockfish_move()
        
        reward = self._get_reward()
        done = self.board.is_game_over()
        
        return self._get_obs(), reward, done, False, self._get_info()
    
    def _make_stockfish_move(self):
        """Have Stockfish make a move on the board"""
        if self.stockfish and not self.board.is_game_over():
            result = self.stockfish.play(self.board, chess.engine.Limit(time=0.1))
            self.board.push(result.move)
    
    def _get_obs(self):
        # Convert the board to a 3D numpy array representation
        board_array = np.zeros((8, 8, 12), dtype=np.float32)
        
        # Map each piece type to a specific channel
        pieces = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank = 7 - chess.square_rank(square)  # Flip rank (0 = bottom rank)
                file = chess.square_file(square)
                channel = pieces[piece.symbol()]
                board_array[rank, file, channel] = 1
                
        return board_array
    
    def _action_to_move(self, action):
        # Convert action index to chess move
        legal_moves = list(self.board.legal_moves)
        if action < len(legal_moves):
            return legal_moves[action]
        else:
            # If the action index is out of range, return the first legal move
            return legal_moves[0] if legal_moves else None
    
    def _get_reward(self):
        """
        Enhanced reward function for chess-playing agent.
        Provides more chess-specific feedback to guide learning.
        """
        # Terminal states with high rewards/penalties
        if self.board.is_checkmate():
            return 100 if self.board.outcome().winner == self.play_as_white else -100
        
        if self.board.is_stalemate():
            # Stalemate is better than losing but worse than winning
            return -10 if self.board.turn == self.play_as_white else 10
        
        if self.board.is_insufficient_material():
            # Draw by insufficient material
            return 0
        
        # Non-terminal rewards based on chess principles
        reward = 0.0
        
        # Material advantage (most important chess metric)
        material_value = self._calculate_material_advantage()
        reward += material_value * 0.3  # Scale material advantage
        
        # Center control reward
        center_control = self._evaluate_center_control()
        reward += center_control * 0.2
        
        # Piece development and king safety
        development = self._evaluate_development()
        reward += development * 0.1
        
        # Check rewards (immediate tactical advantage)
        if self.board.is_check():
            reward += 0.5
        
        # Mobility (number of legal moves)
        mobility = len(list(self.board.legal_moves)) / 30.0  # Normalize
        reward += mobility * 0.1
        
        # Small reward for game progression to encourage finishing games
        reward += 0.01
        
        return reward

    def _calculate_material_advantage(self):
        """Calculate material advantage using standard piece values."""
        piece_values = {
            'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0,
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0
        }
        
        material_balance = 0
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values[piece.symbol()]
                # Add value if it's the agent's piece, subtract if opponent's
                material_balance += value if piece.color == self.play_as_white else -value
        
        return material_balance / 39.0  # Normalize by total material value

    def _evaluate_center_control(self):
        """Evaluate control of the center of the board."""
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        center_control = 0
        
        # Piece in center
        for square in center_squares:
            piece = self.board.piece_at(square)
            if piece:
                center_control += 0.5 if piece.color == self.play_as_white else -0.5
        
        # Attacks on center
        for square in center_squares:
            if self.board.is_attacked_by(self.play_as_white, square):
                center_control += 0.25
            if self.board.is_attacked_by(not self.play_as_white, square):
                center_control -= 0.25
        
        return center_control / 4.0  # Normalize

    def _evaluate_development(self):
        """Evaluate piece development and king safety."""
        development = 0
        
        # Check if knights and bishops are moved from starting positions
        start_minor_pieces = [chess.B1, chess.G1, chess.C1, chess.F1] if self.play_as_white else [chess.B8, chess.G8, chess.C8, chess.F8]
        
        for square in start_minor_pieces:
            piece = self.board.piece_at(square)
            piece_type = None if piece is None else piece.piece_type
            piece_color = None if piece is None else piece.color
            
            expected_color = self.play_as_white
            expected_types = [chess.KNIGHT, chess.BISHOP]
            
            # If piece is not there or it's not the original piece, development has occurred
            if piece is None or piece_color != expected_color or piece_type not in expected_types:
                development += 0.25
        
        # Castle bonus (major development goal)
        if self._has_castled(self.play_as_white):
            development += 1.0
        
        # Penalty for unsafe king
        if self._king_in_danger(self.play_as_white):
            development -= 1.0
        
        return development / 3.0  # Normalize

    def _has_castled(self, color):
        """Check if a player has castled."""
        # Check king position
        king_square = self.board.king(color)
        start_square = chess.E1 if color else chess.E8
        
        # If king has moved and it's not on starting square, check if it's on a typical castling square
        if king_square != start_square:
            if king_square in [chess.G1, chess.C1] if color else king_square in [chess.G8, chess.C8]:
                return True
        return False

    def _king_in_danger(self, color):
        """Check if the king is in danger (attacks nearby)."""
        king_square = self.board.king(color)
        if king_square is None:
            return True  # No king is definitely dangerous
        
        # Check if squares around king are attacked
        king_danger = 0
        rank, file = chess.square_rank(king_square), chess.square_file(king_square)
        
        for r in range(max(0, rank-1), min(8, rank+2)):
            for f in range(max(0, file-1), min(8, file+2)):
                square = chess.square(f, r)
                if self.board.is_attacked_by(not color, square):
                    king_danger += 1
        
        return king_danger >= 3  # If 3 or more squares around king are attacked
    
    def _get_info(self):
        """Get additional information about the current game state"""
        info = {}
        
        if self.board.is_game_over():
            outcome = self.board.outcome()
            if outcome is None:  # This shouldn't happen, but just in case
                info['outcome'] = 'draw'
            elif outcome.winner is None:
                info['outcome'] = 'draw'
            elif outcome.winner == self.play_as_white:
                info['outcome'] = 'win'
            else:
                info['outcome'] = 'loss'
                
            # Add reason for game ending
            info['termination_reason'] = str(outcome.termination) if outcome else "unknown"
        
        info['fen'] = self.board.fen()
        info['fullmoves'] = self.board.fullmove_number
        info['is_check'] = self.board.is_check()
        
        return info
    
    def render(self, mode=None):
        """
        Render the current state of the chess board.
        
        Args:
            mode: Either 'human' or 'rgb_array'
        
        Returns:
            RGB array if mode is 'rgb_array', None otherwise
        """
        mode = mode or self.render_mode
        
        if mode == 'human':
            # Print a simple text representation of the board
            print(self.board)
            return None
        elif mode == 'rgb_array':
            # Create a simple visualization using matplotlib and return as RGB array
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.patches import Rectangle
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 8)
            
            # Draw the board squares
            for i in range(8):
                for j in range(8):
                    color = '#D18B47' if (i + j) % 2 == 1 else '#FFCE9E'
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor=color))
            
            # Draw the pieces
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece:
                    file_idx = chess.square_file(square)
                    rank_idx = 7 - chess.square_rank(square)  # Flip for correct orientation
                    
                    color = 'black' if piece.color == chess.BLACK else 'white'
                    symbol = piece.symbol().upper()
                    
                    plt.text(file_idx + 0.5, rank_idx + 0.5, symbol, 
                             fontsize=40, ha='center', va='center', color=color)
            
            # Remove axis labels and ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            
            # Convert to RGB array - fixed method
            fig.canvas.draw()
            
            # Get the RGBA buffer from the figure
            w, h = fig.canvas.get_width_height()
            
            # Try different methods based on what's available
            try:
                # Modern matplotlib
                buf = fig.canvas.buffer_rgba()
                img = np.asarray(buf)
            except AttributeError:
                try:
                    # Alternative method
                    buf = fig.canvas.tostring_argb()
                    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                    # Convert ARGB to RGB
                    img = img[:, :, 1:] 
                except AttributeError:
                    # Last resort
                    from PIL import Image
                    canvas = fig.canvas
                    img = np.array(Image.frombytes('RGB', (w, h), canvas.tostring_rgb()))
                    
            plt.close(fig)
            return img
    
    def close(self):
        if self.stockfish:
            self.stockfish.quit()
            self.stockfish = None
        
        # Clean up other resources
        pass