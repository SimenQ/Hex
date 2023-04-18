
import time
import numpy as np


class Topp:

    
    def game(self, board, player_1, player_2, starting_player, board_visulazier, visualize = True): 
        board.initilize_board(starting_player)
        if starting_player == 1: 
            player = player_1
        player = player_2

        num_moves = 0
        
        if visualize: 
            display_game(board,board_visulazier)
        
        while True: 
            current_state = board.get_state()
            split_state = [starting_player]
            for i in current_state.split():
                split_state.append(int(i))
            split_state = np.array([split_state])
            preds = player.predict(split_state)[0]
            move = player.best_action(preds)
            board.make_move(move)
            num_moves += int(starting_player == 1)
            starting_player = starting_player % 2 + 1
            if visualize: 
                display_game(board, board_visulazier)
            if board.check_winning_state(): 
                break
        
        if board.check_winning_state(1): 
            player_won = 1 
        else: 
            player_won = 2
        print("Player ", player_won, "won!")
        print(num_moves)
        if visualize: 
            display_game(board, board_visulazier)
        return player_won
            



    def tournament(): 
        return  
    
@staticmethod
def display_game(board, board_visulazier): 
    board_visulazier.draw_board(board.board)
    time.sleep(0.5)
