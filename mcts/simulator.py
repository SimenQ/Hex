
import numpy as np
import random
from game.board import Board
class Simulator: 
    def __init__(self, board, board_size, tree):
        self.board = board
        self.board_size = board_size
        self.tree = tree
        self.initial_board = board.copy()

    def initilize_root(self, state, player): 
        state_values = state.split()
        board_state = np.zeros((self.board_size, self.board_size), dtype=int)
        for i in range(self.board_size):
            row_start = i * self.board_size
            row_end = row_start + self.board_size
            board_state[i] = [int(val) for val in state_values[row_start:row_end]]
        self.board.player = player
        self.board.board = board_state 

    def rollout_game(self, sigma, epsilon, board_copy): 
        if sigma < random.random(): 
            return self.tree.critic(board_copy, board_copy.player)
        while True: 
            next_move = self.tree.rollout(board_copy, epsilon, board_copy.player)
            board_copy.make_move(next_move)
            if board_copy.check_winning_state():
                return board_copy.get_reward()
  
    def tree_search(self, board_copy):
        traversal_seq = self.tree.traverse(board_copy)
        if not traversal_seq: 
            return []
        return traversal_seq
    
    def simulate(self, sigma, epsilon, num_search_games):
        board_copy = self.board.copy()
        num_simulations = int(num_search_games / len(board_copy.get_legal_moves()))
     
        for i in range(max(num_simulations, 10)): 
            seq = self.tree_search(board_copy)
            seq.reverse()
            self.tree.expand_tree(board_copy)
            reward = self.rollout_game(sigma, epsilon, board_copy)
           
            for val in seq: 
                self.tree.update(val[0], val[1], reward)
            board_copy = self.board.copy()
        return self.tree.get_distribution(self.bord)

    def reset(self, player): 
        self.board = Board(self.board_size, player)

    
    