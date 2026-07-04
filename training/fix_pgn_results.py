"""
Fill in the Result header of generated self-play PGNs (checkmate/draw)
so they can be used as training data.

Example:
    python training/fix_pgn_results.py --input-dir selfplay_games/ --output-dir selfplay_fixed/
"""

import argparse
import os

import chess.pgn


def determine_winner(board):
    if board.is_checkmate():
        return "1-0" if board.turn == chess.BLACK else "0-1"
    else:
        return "1/2-1/2"


def fix_pgn_results(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pgn"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, "r") as input_file:
                game = chess.pgn.read_game(input_file)

            result = determine_winner(game.end().board())
            game.headers["Result"] = result

            with open(output_path, "w") as output_file:
                output_file.write(str(game))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix Result headers in self-play PGNs")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    fix_pgn_results(args.input_dir, args.output_dir)
