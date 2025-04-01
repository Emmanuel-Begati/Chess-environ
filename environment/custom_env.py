import gym
import numpy as np
import chess
import chess.engine

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = gym.spaces.Discrete(4672)  # Number of possible moves in chess
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)

    def reset(self):
        self.board.reset()
        return self._get_obs()

    def step(self, action):
        move = self._action_to_move(action)
        if move not in self.board.legal_moves:
            return self._get_obs(), -10, True, {}  # Invalid move penalty

        self.board.push(move)
        reward = self._get_reward()
        done = self.board.is_game_over()
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Convert the board to a 3D numpy array representation
        board_array = np.zeros((8, 8, 12), dtype=np.float32)
        for piece in chess.PIECE_NAMES:
            piece_index = chess.PIECE_NAMES.index(piece)
            for square in chess.SQUARES:
                if self.board.piece_at(square) is not None:
                    if self.board.piece_at(square).symbol() == piece:
                        board_array[chess.square_rank(square), chess.square_file(square), piece_index] = 1
        return board_array

    def _action_to_move(self, action):
        # Convert action index to chess move
        legal_moves = list(self.board.legal_moves)
        return legal_moves[action] if action < len(legal_moves) else None

    def _get_reward(self):
        # Reward function based on the game state
        if self.board.is_checkmate():
            return 100  # Win
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0  # Draw
        elif self.board.is_check():
            return 10  # Check
        return -1  # Default penalty for each step

    def render(self, mode='human'):
        # Optional: Implement rendering logic here
        pass