
import numpy as np
import random
from game.board import Board
class Simulator: 
    def __init__(self, board, board_size, starting_player, tree):
        self.board = Board(board_size, starting_player)
        self.board_size = board_size
        self.tree = tree
        

    def initialize_root(self, state, player): 
        state_values = state.split()
        board_state = np.array([[int(i) for i in state_values[j*self.board_size:(j+1)*self.board_size]] for j in range(self.board_size)],dtype=object)
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
        num_legal_moves = len(board_copy.get_legal_moves())
        print("Number of legal moves: ", num_legal_moves)
        num_simulations = int(num_search_games / num_legal_moves)
     
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

    
    