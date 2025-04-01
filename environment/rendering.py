from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import pygame

class ChessRenderer:
    def __init__(self, board):
        self.board = board
        self.square_size = 60
        self.width = self.square_size * 8
        self.height = self.square_size * 8
        self.init_pygame()
        self.init_opengl()

    def init_pygame(self):
        pygame.init()
        pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption('Chess Visualization')

    def init_opengl(self):
        glClearColor(1, 1, 1, 1)
        glEnable(GL_DEPTH_TEST)

    def draw_square(self, x, y, color):
        glBegin(GL_QUADS)
        glColor3f(*color)
        glVertex2f(x, y)
        glVertex2f(x + self.square_size, y)
        glVertex2f(x + self.square_size, y + self.square_size)
        glVertex2f(x, y + self.square_size)
        glEnd()

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = (0.8, 0.8, 0.8) if (row + col) % 2 == 0 else (0.2, 0.2, 0.2)
                self.draw_square(col * self.square_size, row * self.square_size, color)

    def draw_pieces(self):
        piece_map = {
            'P': (1, 1, 1), 'R': (1, 1, 1), 'N': (1, 1, 1), 'B': (1, 1, 1),
            'Q': (1, 1, 1), 'K': (1, 1, 1), 'p': (0, 0, 0), 'r': (0, 0, 0),
            'n': (0, 0, 0), 'b': (0, 0, 0), 'q': (0, 0, 0), 'k': (0, 0, 0)
        }
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != '.':
                    glColor3f(*piece_map[piece])
                    self.draw_square(col * self.square_size + 10, row * self.square_size + 10, (1, 1, 1))

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.draw_board()
        self.draw_pieces()
        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            self.render()

# Example usage:
# board = [['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
#          ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
#          ['.', '.', '.', '.', '.', '.', '.', '.'],
#          ['.', '.', '.', '.', '.', '.', '.', '.'],
#          ['.', '.', '.', '.', '.', '.', '.', '.'],
#          ['.', '.', '.', '.', '.', '.', '.', '.'],
#          ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
#          ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']]
# renderer = ChessRenderer(board)
# renderer.run()