
from board import Board


def play_game():
    board = Board()
    player = "X"
    game_over = False

    while not game_over:
        print(board)
        print(f"Player {player}'s turn")

        row = int(input("Enter row (0-2): "))
        col = int(input("Enter column (0-2): "))

        try:
            board.make_move(row, col, player)
        except ValueError as e:
            print(e)
            continue

        winner = board.get_winner()
        if winner is not None:
            print(board)
            print(f"Player {winner} wins!")
            game_over = True
        elif board.is_full():
            print(board)
            print("The game is a tie.")
            game_over = True
        else:
            player = "O" if player == "X" else "X"


if __name__ == "__main__":
    play_game()
