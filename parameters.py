class Parameters:
    def __init__(self):
        # Board parameters
        self.board_size = 4
        self.starting_player = None #0 for mix
        # MCTS parameters
        self.number_of_games = 10
        self.number_search_episodes_for_each_move = 200
        self.epsilon = 1
        self.epsilon_decay = 0.985
        # ANET parameters
        self.lr = 0.01
        self.batch_size = 64
        self.nn_dims = (((self.board_size**2)*2/3) + self.board_size**2, self.board_size**2 - self.board_size )
        self.activation_function = ["sigmoid", "tanh"] # [0] = actor, [1] = critic
        self.optimizer = "adam"
        self.train_ANET = True
        # With sigma=1.5 and decay=0.97, first chance of critic eval is at episode 14
        self.sigma = 1
        self.sigma_decay = 0.97
        # TOPP parameters
        self.number_of_ANET = 5 # + 1 for episode 0
        self.topp = True
        self.topp_games = 25
        self.visualize_last_game = True
        


