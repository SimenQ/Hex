
from tests.board import Board
from tests.board_visulizer import BoardVisualizer

board = Board(3, starting_player=0)
visulize = BoardVisualizer()
visulize.draw_board(board)

