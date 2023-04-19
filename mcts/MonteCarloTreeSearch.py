import math
import random 
import numpy as np
from neuralnet.neuralnet import NeuralNet


class MCTS:
    def __init__(self, root, neural_net):
        self.root = root
        self.states = {}
        self.state_action = {}
        self.c = 1
        self.neural_net = neural_net


#Q(s,a) = Q(s,a) + α[r + γ(maxa'Q(s',a')) - Q(s,a)] this is a more common way to update Q-values, this code uses  "sample average" update rule.
    def update(self, state, action, reward):
        self.state_action[(state, action)]["N"] += 1
        self.state_action[(state, action)]["Q"] += (reward - self.get_Q(state, action)) / (1 + self.get_N(state, action))

        self.states[state]["N"] += 1
        self.states[state]["Q"] += (reward - self.get_Q(state)) / (1 + self.get_N(state))

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
        return self.states[state]["Q"]

    def exploration_bonus(self, state, action):
        if self.get_N(state) == 0:
            return math.inf
        return self.c*np.sqrt(np.log(self.get_N(state)) / (1 + self.get_N(state,action)))
        
    #Compute the distribution over moves for a given board state using the neural network.
    #Returns a tuple containing the distribution over moves as a list of (move, probability) pairs, and the estimated
    #value of the current state as a scalar.
    def get_distribution(self, board):
        num_moves = board.board_size ** 2
        all_moves = []
        for i in range(num_moves):
            move = NeuralNet.convert_to_2d_move(i, board.board_size)
            all_moves.append(move)
        
        current_state = board.get_state()
        N_values = []
        for move in all_moves: 
            value = self.get_N(current_state, move)
            N_values.append(value)
        
        normalized_N_values = NeuralNet.normalize(np.array(N_values))

        move_probs = []
        for i in range(len(all_moves)):
            move_probs.append((all_moves[i], normalized_N_values[i]))

        
        return move_probs, self.get_Q(current_state)
    
    
    
    #Aims to predict an action using neural network given the current state of the board and epsilon value.
    #Returns the best action if the random generated number is grater that epsilon otherwise returns a random action
    def rollout(self, board, epsilon, player):
        random_number = random.random()
        if random_number > epsilon:
            current_state = board.get_state()
            split_values = [player]
            for i in current_state.split():
                split_values.append(int(i))
            split_values = np.array([split_values])
            preds = self.neural_net.predict(split_values)
            return self.neural_net.best_action(preds[0])
        else: 
            return self.random_action(board)

    #Aims to predict the value of the current board state using the neural network.
    #Returns the predicted value of the current board state
    def critic(self, board, player):
        current_state = board.get_state()
        split_values = [player]
        for i in current_state.split():
            split_values.append(int(i))
        split_values = np.array([split_values])
        preds = self.neural_net.predict(split_values)
        return preds[1][0][0]


    def random_action(self, board):
        return random.choice(board.get_legal_moves())

    def expand_tree(self, board):
        if board.check_winning_state():
            return
        current_state = board.get_state()
        actions = board.get_legal_moves()
        for action in actions: 
            if(current_state,action) not in self.state_action:
                self.state_action[(current_state,action)] = {"N": 0, "Q": 0}
        if current_state not in self.states:
            self.states[current_state] = {"N": 0, "Q": 0}
      

    def select_action(self, board, player):
        current_state = board.get_state()
        actions = board.get_legal_moves()
        for action in actions: 
            if (player == 1):
                action_value = [self.get_Q(current_state,action) + self.exploration_bonus(current_state,action)]
                action_index = action_value.index(max(action_value))
            else:  
                action_value = [self.get_Q(current_state,action) - self.exploration_bonus(current_state,action)]
                action_index = action_value.index(min(action_value))
        
        return actions[action_index]
            
    def traverse(self, board):
        sequence = []
        while not board.check_winning_state() and board.get_state() in self.states:
            """
             legal_actions = []
            for move in board.get_legal_moves():
                if board.check_legal_move(move):
                    legal_actions.append(move)
            if not legal_actions:
                break
            """
            action = self.select_action(board, board.player)
            sequence.append((board.get_state(), action))
            board.make_move(action)
        return sequence


    def reset(self):
        self.states = {}
        self.state_action = {}

