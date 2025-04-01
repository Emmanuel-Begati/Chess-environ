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
        # Reward function based on the game state
        if self.board.is_checkmate():
            # Check who won the game
            return 100 if self.board.outcome().winner == self.play_as_white else -100
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0  # Draw
        elif self.board.is_check():
            return 1  # Small reward for putting opponent in check
        else:
            # Evaluate board position for more sophisticated reward
            return 0.01  # Small positive reward for each step
    
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
            
            # Convert to RGB array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return img
    
    def close(self):
        if self.stockfish:
            self.stockfish.quit()
            self.stockfish = None
        
        # Clean up other resources
        pass