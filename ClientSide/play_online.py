
import numpy as np
from neuralnet.neuralnet import NeuralNet
from .ActorClient import ActorClient

class MyClient(ActorClient): 
    
    def __init__(self, auth, qualify, episode_number):
        self.episode_number = episode_number
        ActorClient.__init__(self, auth = auth, qualify = qualify)
        
    
    def set_actor(self, actor):
        self.actor = actor

    def handle_series_start(
        self, unique_id, series_id, player_map, num_games, game_params
    ):
        """Called at the start of each set of games against an opponent

        Args:
            unique_id (int): your unique id within the tournament
            series_id (int): whether you are player 1 or player 2
            player_map (list): (inique_id, series_id) touples for both players
            num_games (int): number of games that will be played
            game_params (list): game-specific parameters.

        Note:
            > For the qualifiers, your player_id should always be "-200",
              but this can change later
            > For Hex, game params will be a 1-length list containing
              the size of the game board ([board_size])
        """
        self.logger.info(
            'Series start: unique_id=%s series_id=%s player_map=%s num_games=%s'
            ', game_params=%s',
            unique_id, series_id, player_map, num_games, game_params,
        )

        board_size = game_params[0]
        self.actor = NeuralNet(board_size=board_size, load_saved_model=True, episode_number=self.episode_number)

    def handle_game_start(self, start_player):
        """Called at the beginning of of each game

        Args:
            start_player (int): the series_id of the starting player (1 or 2)
        """
        self.logger.info('Game start: start_player=%s', start_player)

    def handle_get_action(self, state):
        """Called whenever it's your turn to pick an action

        Args:
            state (list): board configuration as a list of board_size^2 + 1 ints

        Returns:
            tuple: action with board coordinates (row, col) (a list is ok too)

        Note:
            > Given the following state for a 5x5 Hex game
                state = [
                    1,              # Current player (you) is 1
                    0, 0, 0, 0, 0,  # First row
                    0, 2, 1, 0, 0,  # Second row
                    0, 0, 1, 0, 0,  # ...
                    2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                ]
            > Player 1 goes "top-down" and player 2 goes "left-right"
            > Returning (3, 2) would put a "1" at the free (0) position
              below the two vertically aligned ones.
            > The neighborhood around a cell is connected like
                  |/
                --0--
                 /|
        """
        self.logger.info('Get action: state=%s', state)
        preds = self.actor.predict(np.array([state]))[0]
        next_move = self.actor.best_action(preds, random = False)
        return int(next_move[0]), int(next_move[1])
    
    def handle_game_over(self, winner, end_state):
        """Called after each game

        Args:
            winner (int): the winning player (1 or 2)
            end_stats (tuple): final board configuration

        Note:
            > Given the following end state for a 5x5 Hex game
            state = [
                2,              # Current player is 2 (doesn't matter)
                0, 2, 0, 1, 2,  # First row
                0, 2, 1, 0, 0,  # Second row
                0, 0, 1, 0, 0,  # ...
                2, 2, 1, 0, 0,
                0, 1, 0, 0, 0
            ]
            > Player 1 has won here since there is a continuous
              path of ones from the top to the bottom following the
              neighborhood description given in `handle_get_action`
        """
        self.logger.info('Game over: winner=%s end_state=%s', winner, end_state)

    def handle_series_over(self, stats):
        """Called after each set of games against an opponent is finished

        Args:
            stats (list): a list of lists with stats for the series players

        Example stats (suppose you have ID=-200, and playing against ID=999):
            [
                [-200, 1, 7, 3],  # id=-200 is player 1 with 7 wins and 3 losses
                [ 999, 2, 3, 7],  # id=+999 is player 2 with 3 wins and 7 losses
            ]
        """
        self.logger.info('Series over: stats=%s', stats)

    def handle_tournament_over(self, score):
        """Called after all series have finished

        Args:
            score (float): Your score (your win %) for the tournament
        """
        self.logger.info('Tournament over: score=%s', score)
    
