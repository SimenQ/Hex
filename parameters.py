class Parameters:
    def __init__(self):
        # Board parameters
        self.board_size = 6
        self.starting_player = 1 # 0 for mix
        # MCTS parameters
        self.number_of_games = 25
        self.number_of_search_episodes = 100
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
        self.number_of_cached_anet = 20 # + 1 for episode 0
        self.topp = False
        self.topp_games = 4
        self.visualize_last_game = False
        