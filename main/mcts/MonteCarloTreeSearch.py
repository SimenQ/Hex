import math
import numpy as np


class MCTS:
    def __init__(self, root, neural_net, c):
        self.root = root
        self.sates = {}
        self.state_action = {}
        self.c = c
        self.neural_net = neural_net

    def update(self, state, action, reward):
        return

    def get_N(self, state, action=None):
        return

    def get_Q(self, state, action=None):
        return

    def exploration_reward(self, state, action):
        return

    def get_distribution(self, board):
        return

    def rollout(self, board, epsilion, player):
        return

    def critic(self, board, player):
        return

    def random_action(self, board):
        return

    def expand_tree(self, board):
        return

    def select_action(self, board, player):
        return

    def get_max_value_move(self, board, move):
        return

    def get_min_value_move(self, board, move):
        return

    def traverse(self, board):
        return

    def reset(self):
        self.states = {}
        self.state_action = {}
