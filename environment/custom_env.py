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
        
        # CRITICAL CHANGE: Piece variety reward - dramatically increased weight
        piece_variety = self._evaluate_piece_variety()
        reward += piece_variety * 0.5  # Increased from 0.15 to 0.5
        
        # Development and king safety - decreased slightly
        development = self._evaluate_development()
        reward += development * 0.1  # Decreased from 0.15
        
        # Check rewards (immediate tactical advantage)
        if self.board.is_check():
            reward += 0.5
        
        # Pawn structure reward
        pawn_structure = self._evaluate_pawn_structure()
        reward += pawn_structure * 0.1
        
        # CRITICAL CHANGE: Much stronger penalty for repeated moves
        repetition_penalty = self._evaluate_move_repetition()
        reward -= repetition_penalty * 1.0  # Increased from 0.2 to 1.0
        
        # Small reward for game progression to encourage finishing games
        reward += 0.01
        
        return reward

    def _evaluate_piece_variety(self):
        """
        Reward for moving a variety of pieces rather than the same piece repeatedly.
        Tracks which pieces have been moved and rewards accordingly.
        """
        # Initialize piece movement tracking on first call
        if not hasattr(self, 'piece_movement_tracker'):
            self.piece_movement_tracker = {}
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece:
                    self.piece_movement_tracker[square] = 0
        
        # Update tracker based on the last move
        if self.board.move_stack:
            last_move = self.board.move_stack[-1]
            from_square = last_move.from_square
            
            # Increment the counter for this piece's square
            if from_square in self.piece_movement_tracker:
                self.piece_movement_tracker[from_square] += 1
        
        # Calculate variety score based on the distribution of moves
        total_moves = sum(self.piece_movement_tracker.values())
        if total_moves == 0:
            return 0
        
        # Count how many different pieces have been moved
        moved_pieces = sum(1 for count in self.piece_movement_tracker.values() if count > 0)
        max_possible = len([sq for sq in self.piece_movement_tracker if self.board.piece_at(sq) is not None])
        max_possible = max(1, max_possible)  # Avoid division by zero
        
        # CRITICAL CHANGE: Higher reward for piece variety
        variety_score = (moved_pieces / max_possible) * 2.0  # Multiplied by 2.0 to emphasize
        
        # CRITICAL CHANGE: Stronger penalty for moving the same piece repeatedly
        if moved_pieces > 0:
            avg_moves = total_moves / moved_pieces
            max_moves = max(self.piece_movement_tracker.values())
            if max_moves > avg_moves * 1.5:  # Reduced threshold from 2 to 1.5
                variety_score *= 0.5  # Increased penalty from 0.7 to 0.5
            
            # CRITICAL ADDITION: Add penalty for the most-moved piece being moved too much
            if max_moves > 3 and total_moves > 5:
                variety_score -= 0.3  # Direct penalty for excessive use of one piece
        
        return variety_score

    def _evaluate_move_repetition(self):
        """
        Penalize repetitive moves (like moving the same piece back and forth).
        """
        # Need at least 4 moves to detect a repetition
        if len(self.board.move_stack) < 4:
            return 0
        
        # Check last 4 moves for a pattern like A->B->A->B
        moves = self.board.move_stack[-4:]
        if (moves[0].from_square == moves[2].from_square and 
            moves[0].to_square == moves[2].to_square and
            moves[1].from_square == moves[3].from_square and
            moves[1].to_square == moves[3].to_square):
            return 2.0  # Increased from 1.0 to 2.0 - Found a repetitive pattern
        
        # Check for a single piece being moved repeatedly
        last_piece_moves = {}
        for move in self.board.move_stack[-6:]:  # Reduced from 8 to 6 moves
            from_square = move.from_square
            if from_square not in last_piece_moves:
                last_piece_moves[from_square] = 1
            else:
                last_piece_moves[from_square] += 1
        
        # CRITICAL CHANGE: More strict rules for repetitive moves
        max_moves = max(last_piece_moves.values()) if last_piece_moves else 0
        
        # If any piece has been moved 3+ times in the last 6 moves, strongly penalize
        if max_moves >= 3:
            return 1.5  # Increased penalty
        elif max_moves == 2:
            return 0.5  # Small penalty for minor repetition
        
        return 0

    def _evaluate_pawn_structure(self):
        """
        Evaluate the pawn structure - isolated pawns, doubled pawns, etc.
        """
        # Get all pawns for the agent's color
        pawn_type = chess.PAWN
        color = chess.WHITE if self.play_as_white else chess.BLACK
        pawn_squares = [s for s in chess.SQUARES if 
                       self.board.piece_at(s) is not None and 
                       self.board.piece_at(s).piece_type == pawn_type and
                       self.board.piece_at(s).color == color]
        
        if not pawn_squares:
            return 0
        
        score = 0
        
        # Reward for pawn chain (adjacent pawns protecting each other)
        files = [chess.square_file(s) for s in pawn_squares]
        ranks = [chess.square_rank(s) for s in pawn_squares]
        
        # Count doubled pawns (negative)
        doubled_count = len(files) - len(set(files))
        score -= doubled_count * 0.1
        
        # Check for isolated pawns (no friendly pawns on adjacent files)
        isolated_count = 0
        for pawn in pawn_squares:
            f = chess.square_file(pawn)
            if (f-1 not in files) and (f+1 not in files):
                isolated_count += 1
        
        score -= isolated_count * 0.1
        
        # Center pawns are good (e4, d4, e5, d5)
        center_files = [3, 4]  # d and e files
        center_ranks = [3, 4]  # 4th and 5th ranks
        center_pawn_count = sum(1 for f, r in zip(files, ranks) 
                               if f in center_files and r in center_ranks)
        score += center_pawn_count * 0.15
        
        # Normalize
        return score / len(pawn_squares)

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