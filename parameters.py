class Parameters:
    def __init__(self):
        # Board parameters
        self.board_size = 5
        self.starting_player = None #0 for mix
        # MCTS parameters
        self.number_of_games = 200
        self.number_of_search_episodes = 500
        self.epsilon = 1
        self.epsilon_decay = 0.98
        # ANET parameters
        self.lr = 0.01
        self.batch_size = 64
        self.nn_dims = (512, 256)
        self.activation_function = ["sigmoid", "tanh"] # [0] = actor, [1] = critic
        self.optimizer = "adam"
        # With sigma=1.5 and decay=0.97, first chance of critic eval is at episode 14
        self.sigma = 2
        self.sigma_decay = 1
        # TOPP parameters
        self.number_of_ANET = 5 # + 1 for episode 0
        self.topp = True
        self.topp_games = 25
        self.visualize_last_game = True
        


