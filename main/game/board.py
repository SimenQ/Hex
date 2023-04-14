
import collections
import numpy as np
import random


class Board:

    def __init__(self, board_size, starting_player):
        self.board_size = board_size
        self.starting_player = starting_player
        self.board = []
        self.initilize_board(self.starting_player)

    def initilize_board(self, starting_player=None):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        if starting_player == None:
            self.starting_player = random.choice([1, 2])
        else:
            assert starting_player in [1, 2]
            self.starting_player = starting_player

    # This function takes the board and turn it into a one-dimenstional array
    def get_1D_representation_of_board(self):
        return self.board.flatten()

    def get_legal_moves(self):
        flat_board = self.get_1D_representation_of_board()
        legal_moves = []
        for i in range(len(flat_board)):
            if flat_board[i] == 0:
                row = i // self.board_size
                col = i % self.board_size
                legal_moves.append((row, col))
        return legal_moves

    def check_legal_move(self, move):
        try:
            self.board[move[0]][move[1]] == 0
        except IndexError:
            return False

    def make_move(self, move, player=None):
        if player:
            current_player = player
        else:
            current_player = self.player

        if not self.check_legal_move(move):
            raise Exception(
                f"Illegal move provided: {move} {self.get_1D_representation_of_board()}")

        if current_player not in [1, 2]:
            raise Exception("Player must be either 1 or 2")

        self.board[move[0]][move[1]] = current_player
        self.player = current_player % 2 + 1


def check_winning_state(self, player=None):
    """Checks if the given player or both players have won the game by connecting their pieces
    from one edge of the board to the opposite edge.

    Args:
        player (int or None): The player to check for winning state. If None, checks for both players.
            1 for Player 1 and 2 for Player 2. Defaults to None.

    Returns:
        bool: True if the specified player has won the game, False otherwise.
    """
    if player is None:
        # Check winning state for both players
        return self.check_winning_state(1) or self.check_winning_state(2)

    if player == 1:
        # Player 1 wins if they connect the left and right edges of the board.
        reachable_nodes = [(i, 0) for i in range(
            self.board_size) if self.board[i][0] == 1]
        winning_nodes = [(i, self.board_size - 1)
                         for i in range(self.board_size) if self.board[i][self.board_size - 1] == 1]
    elif player == 2:
        # Player 2 wins if they connect the top and bottom edges of the board.
        reachable_nodes = [(0, j) for j in range(
            self.board_size) if self.board[0][j] == 2]
        winning_nodes = [(self.board_size - 1, j)
                         for j in range(self.board_size) if self.board[self.board_size - 1][j] == 2]

    # Use breadth-first search to find all reachable nodes for the player.
    visited = set(reachable_nodes)
    queue = collections.deque(reachable_nodes)
    while queue:
        node = queue.popleft()
        neighbors = self.get_neighbors(node)
        for neighbor in neighbors:
            if neighbor not in visited and self.board[neighbor[0]][neighbor[1]] == player:
                visited.add(neighbor)
                queue.append(neighbor)

    # Check if the player has won by connecting their pieces to the opposite edge of the board.
    for node in visited:
        if node in winning_nodes:
            return True

    return False

# Can be modified or removed depening on how we want to implemnt the reward for the agent.
# Adding this simple reward function for testing
def get_reward(self, player):
    if (self.check_winning_state(player)):
        return 1
    else:
        return -100
