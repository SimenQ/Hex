
import time
import numpy as np


class Topp:

    def game(self, board, player_1, player_2, starting_player, board_visulazier, display = True): 
        board.initialize_board(starting_player)
        if starting_player == 1: 
            player = player_1
        else:
            player = player_2

        num_moves = 0
        
        if display: 
            Topp.display_game(board,board_visulazier)

        while not board.check_winning_state():
            current_state = board.get_state()
            split_values = [starting_player]
            for i in current_state.split():
                split_values.append(int(i))
            split_values = np.array([split_values])
            preds = player.predict(split_values)[0]
            move = player.best_action(preds)
            board.make_move(move)
            num_moves += int(starting_player == 1)
            starting_player = starting_player % 2 + 1
            if display: 
                Topp.display_game(board, board_visulazier)
                    
        if board.check_winning_state(1): 
            player_won = 1 
        else: 
            player_won = 2

        #print("Player ", player_won, "won! with", num_moves," number moves")

        if display: 
            Topp.display_game(board, board_visulazier)

        return player_won
    
    def tournament(self, board, episodes, models, games, board_visulazier , display_last_game = True):
        #list over number of games won by player 1 (or both) player
        won_games = [0] * len(episodes)
        for i in range(len(models)): 
            for j in range(i+1, len(models)): 
                num_wins_player_1 = 0
                num_wins_player_2 = 0
                print("Model (Episode: " + str(episodes[i]) + ") vs. Model (Episode: " + str(episodes[j]) + ")")
                for g in range(games): 
                    winner = self.game(board, models[i], models[j], g % 2 + 1, board_visulazier, display = display_last_game and g == games -1 )
                    if winner == 1: 
                        num_wins_player_1 += 1
                    elif winner == 2: 
                        num_wins_player_2 += 1
                    else: 
                        pass
                print("Model trained with (Episode = " + str(episodes[i]) + ") won", num_wins_player_1, "times, as starting player")
                print("Model trained with (Episode =  " + str(episodes[j]) + ") won", num_wins_player_2, "times as second player")  
                print("\n")
                won_games[i] += num_wins_player_1

        for i in range(len(models) - 1, -1, -1): 
            for j in range(i-1, -1, -1):
                num_wins_player_1 = 0
                num_wins_player_2 = 0
                print("Model (Episode: " + str(episodes[i]) + ") vs. Model (Episode: " + str(episodes[j]) + ")")
                for g in range(games): 
                    winner = self.game(board, models[i], models[j], g % 2 + 1, board_visulazier, display = display_last_game and g == games -1 )
                    if winner == 1: 
                        num_wins_player_1 += 1
                    elif winner == 2: 
                        num_wins_player_2 += 1
                    else: 
                        pass
                print("Model trained with (Episode = " + str(episodes[i]) + " won", num_wins_player_1, "times, as starting player")
                print("Model trained with (Episode = : " + str(episodes[j]) + " won", num_wins_player_2, "times as second player")  
                print("\n")
                won_games[i] += num_wins_player_1

        print("-------------------------------------------------------------------------------------------------------------")
        print("Final scores")
        print("\n")
        for i in range(len(episodes)): 
            print("Model trained with (Episode = " + str(episodes[i]) + ") won", won_games[i], "times, as starting player")
        print("-------------------------------------------------------------------------------------------------------------")
           
        
    

    @staticmethod
    def display_game(board, board_visulazier): 
        board_visulazier.draw_board(board.board)
        time.sleep(0.5)
