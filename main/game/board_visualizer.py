import pygame


class BoardVisualizer:
    def __init__(self, board):
        pygame.init()
        self.board = board
        self.cell_size = 80
        self.width = self.cell_size * (2 * self.board.board_size - 1)
        self.height = self.width
        self.window_size = (self.width, self.height)
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('HEX')
        self.font = pygame.font.SysFont('Arial', 50)

    def draw_board(self):
        self.screen.fill((255, 255, 255))
        for i in range(self.board.board_size):
            for j in range(self.board.board_size):
                if i % 2 == 0:
                    left = j * self.cell_size + i * self.cell_size // 2
                else:
                    left = (j + 0.5) * self.cell_size + i * self.cell_size // 2
                top = i * self.cell_size
                cell_rect = pygame.Rect(
                    left, top, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (0, 0, 0), cell_rect, 2)
                cell_value = self.board.board[i][j]
                if cell_value != 0:
                    text = self.font.render(str(cell_value), True, (0, 0, 0))
                    text_rect = text.get_rect(center=cell_rect.center)
                    self.screen.blit(text, text_rect)

    def get_cell_coords(self, mouse_pos):
        x, y = mouse_pos
        row = y // self.cell_size
        if row % 2 == 0:
            col = x // self.cell_size - row // 2
        else:
            col = (x - self.cell_size // 2) // self.cell_size - row // 2
        if row < self.board.board_size and col < self.board.board_size:
            return (row, col)
        else:
            return None

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    cell_coords = self.get_cell_coords(event.pos)
                    if cell_coords is not None:
                        row, col = cell_coords
                        if self.board.check_valid_move(row, col):
                            self.board.make_move(row, col)
            self.draw_board()
            pygame.display.update()
        pygame.quit()
