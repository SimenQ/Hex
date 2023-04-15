import math
import random 
import numpy as np


class MCTS:
    def __init__(self, root, neural_net, c):
        self.root = root
        self.states = {}
        self.state_action = {}
        self.c = c
        self.neural_net = neural_net


#Q(s,a) = Q(s,a) + α[r + γ(maxa'Q(s',a')) - Q(s,a)] this is a more common way to update Q-values, this code uses  "sample average" update rule.
    def update(self, state, action, reward):
        self.state_action[(state, action)]["N"] += 1
        self.state_action[(state, action)]["Q"] += (reward - self.get_Q(state, action)) / (1 + self.get_N(state, action))

        self.state[state]["N"] += 1
        self.state[state]["Q"] += (reward - self.get_Q(state)) / (1 + self.get_N(state))

        return

    # Returns the number of times a state-action pair has been visited or the number of times a state has been visited
    def get_N(self, state, action=None):
        """ "
        Parameters:
            state (int or tuple): The state for which to get the number of visits.
            action (int, optional): The action for which to get the number of visits.
        """
        if action:
            self.state_action.setdefault((state, action), {"N": 0, "Q": 0})
            return self.state_action[(state, action)]["N"]
        else:
            self.states.setdefault(state, {"N": 0, "Q": 0})
            return self.states[state]["N"]

    # Returns the estimated Q-value for the given state-action pair or state.
    def get_Q(self, state, action=None):
        if action:
            return self.state_action[(state, action)]["Q"]
        return self.state[state]["Q"]

    def exploration_bonus(self, state, action):
        if self.get_N(state) == 0:
            return math.inf
        return self.c*np.sqrt(np.log(self.get_N(state)) / (1 + self.get_N(state,action)))
        
    #Needs the Neural Net to be implemented 
    def get_distribution(self, board):
        return

    def rollout(self, board, epsilion, player):
        return

    def critic(self, board, player):
        return

    def random_action(self, board):
        return random.choice(board.get_legal_actions())

    def expand_tree(self, board):
       if board.check_winning_state():
        return
       
       state = board.get_state()
       actions = board.get_legal_actions()
       if state not in self.state:
        self.state[state] = {"N": 0, "Q": 0}
       for action in actions: 
        if(state,action) not in self.state_action:
            self.state_action[(state,action)] = {"N": 0, "Q": 0}


    def select_action(self, board, player):
        actions = board.get_legal_actions()
        for action in actions: 
            if (player == 1):
                action_value = [self.get_Q(board,action) + self.exploration_bonus(board,action)]
                action_index = action_value.index(max(action_value))
            else:  
                action_value = [self.get_Q(board,action) - self.exploration_bonus(board,action)]
                action_index = action_value.index(min(action_value))
        
        return actions[action_index]
            
    def traverse(self, board):
        sequence = []
        while not board.check_winning_state() and board.get_state() in self.states:
            legal_actions = []
            for move in board.get_legal_moves():
                if board.check_legal_move(move):
                    legal_actions.append(move)
            if not legal_actions:
                break
            action = self.select_action(board, legal_actions)
            sequence.append((board.get_state(), action))
            board.make_move(action)
        return sequence


    def reset(self):
        self.states = {}
        self.state_action = {}

