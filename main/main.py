
import sys
import pygame

from game.board import Board
from game.board_visualizer import BoardVisualizer


board = Board(6, starting_player=None)


visulazation = BoardVisualizer(board)
visulazation.run()

"""
# Keep the visualization on screen until user exits
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()
"""
