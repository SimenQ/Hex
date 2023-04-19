from rbuf import RBUF
from topp import Topp
from neuralnet.neuralnet import NeuralNet
from parameters import Parameters
from game.board import Board
from game.board_visualizer import BoardVisualizer
from mcts.simulator import Simulator
from mcts.MonteCarloTreeSearch import MCTS

#Initialize the necceasry objects and parameters from the different classes
p = Parameters()
board = Board(p.board_size, p.starting_player)
board_visualizer = BoardVisualizer()
nn = NeuralNet(p.nn_dims, p.board_size, p.lr, p.activation_function, p.optimizer)
tree = MCTS(board.get_state(), nn)
s = Simulator(board, p.board_size, p.starting_player, tree)
topp = Topp()
rbuf = RBUF()

#Save interval for ANET (i.e  if p.number_of_games is 1000 and p.number_of_ANET is 10, then the model will be saved every 100 games.) 
save_interval = p.number_of_games // p.number_of_ANET


def run_game(eps, sig, starting_player):
    board.initialize_board(starting_player)
    while not board.check_winning_state(): 
        tree.root = board.get_state()
        s.initialize_root(tree.root, board.player)
        D,Q = s.simulate(sigma=sig, epsilon=eps, num_search_games= p.number_search_episodes_for_each_move)
        D = check_for_winning_move(board, D)
        current_player = str(board.player)
        state_str = current_player + " " + tree.root

        
        #best_move = get_best_move_from_D(D)
        

def run_full_game(epsilon, sigma, starting_player):
    # Starting state
    board.initialize_board(starting_player)
    while not board.check_winning_state():
        # Initialize simulations
        tree.root = board.get_state()
        s.initialize_root(tree.root, board.player)
        # Return distribution
        D, Q = s.simulate(sigma, epsilon, p.number_of_search_episodes)
        D = check_for_winning_move(board, D)
        # Parse to state representation
        s = str(board.player) + " " + tree.root
        # Select move based on D
        next_move = get_best_move_from_D(D)
        # Add to replay buffer
        rbuf.add((s, D, Q))
        board.make_move(next_move)
        s.reset(board.player)
    tree.reset()
    # Reset memoization of visited states during rollouts
    nn.fit(rbuf.get_random_batch(p.batch_size))

def get_best_move_from_D(D):
    best_move = None
    most_visits = -1
    for d in D:
        if (d[1] > most_visits):
            best_move = d[0]
            most_visits = d[1]
    return best_move

def check_for_winning_move(board, D):
    for i, el in enumerate(D):
        # Set limit at 0.5 as this means this move will be taken based on D no matter what, increase the weight for rbuf
        if el[1] > 0.3:
            # Check for winning move
            board_copy = board.copy()
            board_copy.make_move(el[0])
            if board_copy.check_winning_state():
                D = [(el[0], 1.0 if ind == i else 0.0) for ind, el in enumerate(D)]
                return D
            # Check for opponent winning move
            board_copy = board.copy()
            board_copy.make_move(el[0], board_copy.player % 2 + 1)
            if board_copy.check_winning_state():
                D = [(el[0], 1.0 if ind == i else 0.0) for ind, el in enumerate(D)]
                return D
    return D


if __name__ == "__main__":
    if (p.topp):
        episodes = [i*save_interval for i in range(p.number_of_ANET +1)]
        actors = [NeuralNet(board_size=p.board_size, load_saved_model=True, episode_number=i) for i in episodes]
        topp.tournament(board, episodes, actors, p.topp_games, board_visualizer, display_last_game=p.visualize_last_game)
    else:
        epsilon = p.epsilon
        sigma = p.sigma
        for game in range(p.number_of_games):
            if game % save_interval == 0:
                nn.save_model(f"{p.board_size}x{p.board_size}_ep", game)
            print("Game no. " + str(game+1))
            run_full_game(epsilon, sigma, game % 2 + 1 if p.starting_player==None else p.starting_player)
            epsilon *= p.epsilon_decay
            sigma *= p.sigma_decay
        nn.save_model(f"{p.board_size}x{p.board_size}_ep", p.number_of_games)
