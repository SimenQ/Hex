
import pygame
import pygame.freetype


class BoardVisualizer:
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height

    def draw_board(self, board):
        pygame.init()
        height = self.height - 100
        width = self.width
        horisontal_spacing = find_horisontal_spacing(board, width)
        board = convert_diamond_board_shape(board)
        GAME_FONT = pygame.freetype.SysFont("Arial", 20)

        # Create blank screen to be drawn upon
        screen = pygame.display.set_mode((width, height))
        screen.fill((255, 255, 255))
        GAME_FONT.render_to(screen, (40, 40), "Player 1", (0,0,0))
        GAME_FONT.render_to(screen, (40, 60), "Player 2", (0,0,0))
        pygame.draw.circle(screen, (255,0,0),(25, 46), 12)
        pygame.draw.circle(screen, (0, 0, 0), (25, 66), 12)
        pygame.draw.line(screen, (0,0,0), (400, 10), (740, 350), 8)
        pygame.draw.line(screen, (0,0,0), (400, 690), (60, 350), 8)
        pygame.draw.line(screen, (255,0,0), (60, 350), (400, 10), 8)
        pygame.draw.line(screen, (255,0,0),(400, 690), (740, 350) , 8)

        

        for row in range(len(board)):
            horisontal_position = (width / 2) - (
                (len(board[row]) - 1) * horisontal_spacing
            )
            circle_size = 200/len(board)
            for col in range(len(board[row])):
                if board[row][col] == 1:
                    pygame.draw.circle(
                        screen,
                        (255, 0, 0),
                        (horisontal_position, height * ((row + 1) / (len(board) + 1))),
                        circle_size,
                    )
                elif board[row][col] == 0:
                    pygame.draw.circle(
                        screen,
                        (200, 200, 200),
                        (horisontal_position, height * ((row + 1) / (len(board) + 1))),
                        circle_size,
                    )
                elif board[row][col] == 2:
                    pygame.draw.circle(
                        screen,
                        (0, 0, 0),
                        (horisontal_position, height * ((row + 1) / (len(board) + 1))),
                        circle_size,
                    )
                else:
                    raise Exception(
                        "Expected a value of 0, 1 or 2 in cell, instead recieved: "
                        + str(board[row][col])
                    )
                horisontal_position += horisontal_spacing * 2

        pygame.display.update()


def find_horisontal_spacing(board, width):
    """Using the maximum number of nodes to be placed horisontally and width
    of the screen, find a distance that can be used to seperate each node
    equally horisontally."""

    max_rows = 1
    for row in board:
        if len(row) > max_rows:
            max_rows = len(row)
    return (width - 100) / (
        max_rows * 2
    )  # Using width -100, to make sure the circles aren't drawn outside of the screen.


def convert_diamond_board_shape(board):
    """Diamond shaped boards are coded as a NxN matrix. This function converts
    this matrix into a list of lists where nodes appear as they would be on a
    board. This is done so that both boards can be drawn using the same logic.
    eg. The following matrix:
    [[1,2],
    [3,4]]
    will return this:
    [[1],
    [3,2],
    [4]]
    """

    new_board = []
    for i in range(len(board)):
        temp_list = []
        for j in range(i + 1):
            temp_list.append(board[i - j][j])
        new_board.append(temp_list)

    counter = 1
    while counter < len(board):
        length = len(board)
        tmp = []
        for i in range(counter, len(board)):
            tmp.append(board[length - i+counter-1][i])
        new_board.append(tmp)
        counter += 1
    return new_board

