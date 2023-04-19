from parameters import Parameters
from game.board import Board
from game.board_visualizer import BoardVisualizer
from game.topp import Topp
from mcts.MonteCarloTreeSearch import MCTS
from mcts.simulator import Simulator
from neuralnet.neuralnet import NeuralNet
from neuralnet.rbuf import RBUF

#Initialize the necceasry objects and parameters from the different classes
p = Parameters()
board = Board(p.board_size, p.starting_player)
board_visualizer = BoardVisualizer()
nn = NeuralNet(p.nn_dims, p.board_size, p.lr, p.activation_function, p.optimizer)
tree = MCTS(board.get_state(), nn)
s = Simulator(board, p.board_size, p.starting_player, tree)
topp = Topp()
rbuf = RBUF()

def run_game(sig, eps, starting_player):
    # Initialize the game board
    board.initialize_board(starting_player)
    
    while not board.check_winning_state():
        # Set the root node to the current state of the board
        tree.root = board.get_state()

        # Initialize the tree search with the root node and the current player
        s.initialize_root(tree.root, board.player)
        
        # Simulate the game using MCTS
        D, Q = s.simulate(sigma=sig, epsilon=eps, num_search=p.number_search_episodes_for_each_move)
        
        # Check if there is a winning move in the distribution and choose it if there is
        D = check_for_winning_move_from_D(board, D)
        
        # Get the current player and state string for logging purposes
        current_player = str(board.player)
        state_str = current_player + " " + tree.root

        # Choose the move with the most visits, given that D is not empty
        if not D:
            return None
        best_move, most_visits = max(D, key=lambda item: item[1])
        
        rbuf.add((state_str, D, Q))
        board.make_move(best_move)
        s.reset(board.player)
    tree.reset()
    nn.fit(rbuf.get_random_batch(p.batch_size))

   

def check_for_winning_move_from_D(board, D):
    new_D = []
    threshold = 0.5
    for i in range(len(D)):
        move, visits = D[i]

        # If a given node has a probability greater than the threshold, investigate it further
        # to check if it results in a winning move. The threshold is set to prevent the agent
        # from wasting time exploring moves with low probabilities.
        if visits > threshold:
            board_copy = board.copy()
            board_copy.make_move(move)

            # Check for winning move
            if board_copy.check_winning_state():
                for move_weight in D:
                    if move_weight[0] == move:
                        new_D.append((move_weight[0], 1.0))
                    else:
                        new_D.append((move_weight[0], 0.0))
                return new_D

            # Check for opponent winning move
            board_copy = board.copy()
            board_copy.make_move(move, board_copy.player % 2 + 1)
            if board_copy.check_winning_state():
                for weighted_move in D:
                    if weighted_move[0] == move:
                        new_D.append((weighted_move[0], 1.0))
                    else:
                        new_D.append((weighted_move[0], weighted_move[1]))
                return new_D

        new_D.append((move, visits))
    return new_D


#Save interval for ANET (i.e  if p.number_of_games is 1000 and p.number_of_ANET is 10, then the model will be saved every 100 games.) 
save_interval = p.number_of_games // p.number_of_ANET

def run_and_save_model(save_interval):
    sigma = p.sigma
    epsilon = p.epsilon
    for game in range(p.number_of_games): 
        if game % save_interval == 0:
            nn.save_model(f"{p.board_size}x{p.board_size}_episode", game)
        print("Game number: ", str(game+1))
        if p.starting_player == None: 
            player = game % 2 + 1
        else: 
            player = p.starting_player
        run_game(sig=sigma, eps=epsilon, starting_player=player)
        sigma *= p.sigma_decay
        epsilon *= p.epsilon_decay
    nn.save_model(f"{p.board_size}x{p.board_size}_episode", p.number_of_games)

def run_topp(save_interval): 
    episodes = []
    for i in range(p.number_of_ANET + 1): 
        episodes.append(i*save_interval)

    models = []
    for i in episodes:
        model = NeuralNet(board_size=p.board_size, load_saved_model=True, episode_number=i)
        models.append(model)

    topp.tournament(board, episodes, models, p.topp_games, board_visualizer, display_last_game=p.visualize_last_game)


if __name__ == "__main__":
    if(p.train_ANET and not p.topp):
        run_and_save_model(save_interval)
    elif (p.topp and not p.train_ANET):
        run_topp(save_interval)
    else: 
        run_and_save_model(save_interval)
        run_topp(save_interval)
       
    