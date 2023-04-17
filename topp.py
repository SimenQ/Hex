import time  # import the time module for delaying visualizations
import numpy as np  # import the numpy library for mathematical operations

class TOPP:
    
    # function for running a single game of TOPP
    def run_topp_game(self, board, actor1, actor2, starting_player, board_visualizer, visualize=True):
        # reset the board and set the starting player
        board.reset_board(starting_player)
        player_no = starting_player
        player = actor1 if player_no == 1 else actor2
        number_of_moves = 0
        
        # draw the initial board if visualization is enabled
        if visualize:
            board_visualizer.draw_board(board.board)
            time.sleep(0.5)
        
        # loop until the game is over
        while not board.check_winning_state():
            # concatenate the current player number with the current board state
            split_state = np.concatenate(([player_no], [int(i) for i in board.get_state().split()]))
            # use the current player's actor to predict the best move
            preds = player.predict(np.array([split_state]))[0]
            move = player.best_action(preds)
            # make the move on the board and increment the move counter
            board.make_move(move)
            number_of_moves += int(player_no == 1)
            # switch to the other player
            player_no = player_no % 2 + 1
            player = player = actor1 if player_no == 1 else actor2
            # draw the updated board if visualization is enabled
            if visualize:
                board_visualizer.draw_board(board.board)
                time.sleep(0.5)
        
        # print the winner and the number of moves taken
        winning_player = 1 if board.check_winning_state_player_one() else 2
        print(f'Player {winning_player} wins!')
        print(number_of_moves)
        # draw the final board if visualization is enabled
        if visualize:
            board_visualizer.draw_board(board.board)
            time.sleep(0.5)
        
        # return the winning player number
        return winning_player
    
    # function for running a round-robin tournament between multiple actors
    def run_topp(self, board, episodes, actors, topp_games, visualizer, visualize_last_game=True):
        # initialize a list to keep track of each actor's score
        actorscore = [0 for _ in episodes]
        
        # loop over all pairs of actors
        for i in range(len(actors)):
            for n in range(i+1, len(actors)):
                player1 = 0
                player2 = 0
                print(f"Actor[{episodes[i]} episodes] vs. actor[{episodes[n]} episodes]")
                
                # play a specified number of games between the current pair of actors
                for game in range(topp_games):
                    # determine the starting player randomly
                    starting_player = game % 2 + 1
                    winner = self.run_topp_game(board, actors[i], actors[n], starting_player, visualizer, visualize= visualize_last_game and game==topp_games - 1)
                    # increment the score of the winning player
                    player1 += 1 if winner == 1 else 0
                    player2 += 1 if winner == 2 else 0
                
                # print the results of the games played between the current pair of actors
                print(f"Actor[{episodes[i]} episodes] won {player1} times.\nActor[{episodes[n]} episodes] won {player2} times.\n")
                #
