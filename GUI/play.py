"""
Legacy desktop GUI (pygame) for playing against the model.

The web app in app/ is the recommended interface; this is kept as the
original dissertation GUI.

Example:
    python GUI/play.py --model weights/HPC_20x256.pt --rollouts 100
"""

import argparse
import os
import sys
import time

import chess
import pygame
import torch

# Core modules and piece images live in the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import MCTS
import AlphaZeroNetwork

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Play Chess with AlphaZero AI.')
parser.add_argument('--model', type=str,
                    default=os.path.join(_REPO_ROOT, 'weights', 'HPC_20x256.pt'),
                    help='Path to the trained model')
parser.add_argument('--verbose', action='store_true', help='Print verbose information')
parser.add_argument('--rollouts', type=int, default=100, help='Number of MCTS rollouts')
parser.add_argument('--threads', type=int, default=1, help='Number of threads for MCTS rollouts')
parser.add_argument('--color', type=str, default='b',
                    help="Model's color: 'w' or 'b' (you play the other side)")

args = parser.parse_args()

# Prepare neural network
alphaZeroNet = AlphaZeroNetwork.AlphaZeroNet(20, 256)
weights = torch.load(args.model, map_location=torch.device('cpu'))
alphaZeroNet.load_state_dict(weights)

for param in alphaZeroNet.parameters():
    param.requires_grad = False

alphaZeroNet.eval()

# Game parameters
num_rollouts = args.rollouts
num_threads = args.threads
verbose = args.verbose

# Initialize Chess AI
board = chess.Board()
model_color = chess.WHITE if args.color.lower() == 'w' else chess.BLACK

# Initialize Pygame
pygame.init()

# Define the square size and board dimensions
SQUARE_SIZE = 80
BOARD_SIZE = 8 * SQUARE_SIZE

# Load the chess piece images
piece_images = {}
for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
    for color in [chess.WHITE, chess.BLACK]:
        piece_key = (piece_type, color)
        piece_color_code = "w" if color == chess.WHITE else "b"
        image_path = os.path.join(
            _REPO_ROOT, 'images',
            f"{piece_color_code}{chess.piece_symbol(piece_type).upper()}.png")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        piece_images[piece_key] = pygame.transform.scale(
            pygame.image.load(image_path), (SQUARE_SIZE, SQUARE_SIZE))

# Create the Pygame window
screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
pygame.display.set_caption('Hybrid RL Chess')

dragging_piece = None


# Function to convert Pygame coordinates to chess coordinates
def get_chess_coordinates(pygame_x, pygame_y):
    file = pygame_x // SQUARE_SIZE
    rank = 7 - pygame_y // SQUARE_SIZE
    return chess.square(file, rank)


# Function to convert chess coordinates to Pygame coordinates
def get_pygame_coordinates(square):
    file = chess.square_file(square)
    rank = 7 - chess.square_rank(square)
    return file * SQUARE_SIZE, rank * SQUARE_SIZE


# Function to display the chess board
def display_board():
    light_square_color = pygame.Color("#F0D9B5")  # Light square color
    dark_square_color = pygame.Color("#B58863")   # Dark square color

    screen.fill(pygame.Color("white"))

    for square in chess.SQUARES:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        square_color = light_square_color if (file + rank) % 2 == 1 else dark_square_color
        pygame.draw.rect(screen, square_color,
                         (file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square, piece in board.piece_map().items():
        piece_key = (piece.piece_type, piece.color)
        piece_image = piece_images[piece_key]
        screen.blit(piece_image, get_pygame_coordinates(square))

    pygame.display.flip()


# Main game loop
running = True
while running:
    display_board()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            square = get_chess_coordinates(mouse_x, mouse_y)

            if board.piece_at(square) is not None:
                dragging_piece = square

        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging_piece is not None:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                target_square = get_chess_coordinates(mouse_x, mouse_y)

                move = chess.Move(dragging_piece, target_square)
                if move in board.legal_moves:
                    board.push(move)
                    display_board()  # Update the board immediately after player's move

                dragging_piece = None

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                if len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                    print("Took back last move.")

    # Check if the game is over
    if board.is_game_over():
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                print("Black won!")
            else:
                print("White won!")
        elif board.is_stalemate():
            print("Stalemate! Game Over.")
        running = False

    # Only get model move if the game is not over
    if running and board.turn == model_color:
        print("Model is thinking...")
        starttime = time.perf_counter()
        with torch.no_grad():
            root = MCTS.Root(board, alphaZeroNet)
            for _ in range(num_rollouts):
                root.parallelRollouts(board.copy(), alphaZeroNet, num_threads)

        endtime = time.perf_counter()
        elapsed = endtime - starttime
        Q = root.getQ()
        N = root.getN()

        nps = N / elapsed

        same_paths = root.same_paths

        if verbose:
            print(root.getStatisticsString())
            print('total rollouts {} Q {:0.3f} duplicate paths {} elapsed {:0.2f} nps {:0.2f}'.format(
                int(N), Q, same_paths, elapsed, nps))

        edge = root.maxNSelect()
        model_move = edge.getMove()
        print('best move {}'.format(str(model_move)))
        board.push(model_move)

pygame.quit()
