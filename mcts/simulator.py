
import time
import numpy as np
import random
from game.board import Board
class Simulator: 
    def __init__(self, playing_board, board_size, starting_player, tree):
        self.board = Board(board_size, starting_player)
        self.board_size = board_size
        self.tree = tree
        

    def initialize_root(self, state, player): 
        player = player
        state_values = state.split()
        board_state = np.zeros((self.board_size, self.board_size), dtype=int)
        for i in range(self.board_size):
            row_start = i * self.board_size
            row_end = row_start + self.board_size
            board_state[i] = [int(val) for val in state_values[row_start:row_end]]
    
        self.board.player = player
        self.board.board = board_state 

    def rollout_game(self, sigma, epsilon, board_copy): 
        if sigma < np.random.random(): 
            return self.tree.critic(board_copy, board_copy.player)
        while not board_copy.check_winning_state(): 
            next_move = self.tree.rollout(board_copy, epsilon, board_copy.player)
            board_copy.make_move(next_move)
        return board_copy.get_reward(1)
  
    def tree_search(self, board_copy):
        traversal_seq = self.tree.traverse(board_copy)
        if not traversal_seq: 
            return []
        return traversal_seq
    
    def simulate(self, sigma, epsilon, num_search):
        board_copy = self.board.copy()
        num_simulations_dynamic = int(num_search / (len(board_copy.get_legal_moves())))

        t_start = time.time()
        i = 0
        while ((i < num_search ) or time.time() - t_start < 2): 
        #for i in range(max(num_simulations_dynamic, )): 
            seq = self.tree_search(board_copy)
            self.tree.expand_tree(board_copy)
            reward = self.rollout_game(sigma, epsilon, board_copy)
            seq.reverse()
            for val in seq: 
                self.tree.update(val[0], val[1], reward)
            board_copy = self.board.copy()
            i +=1
        print(i ,":", time.time() - t_start)
        return self.tree.get_distribution(self.board)

    def reset(self, player): 
        self.board = Board(self.board_size, player)

    
    