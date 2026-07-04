"""
Generate self-play games for the Hybrid training loop.

Drives training/playchess_selfplay.py (epsilon-greedy self-play) and saves
each finished game as a PGN file.

Note: uses pexpect, so this script runs on Linux/macOS (it was originally
run on the university HPC).

Example:
    python training/selfplay.py --model weights/human_20x256.pt \
        --output-dir selfplay_games/ --num-games 1000
"""

import argparse
import os
import sys

import chess.pgn
import pexpect

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Self-play game generation")
    parser.add_argument("--model", required=True, help="Path to model (.pt) weights")
    parser.add_argument("--output-dir", default="selfplay_games", help="Where to write PGN files")
    parser.add_argument("--num-games", type=int, default=20000, help="Number of games to play")
    parser.add_argument("--start-game", type=int, default=1, help="First game number (to resume a run)")
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    play_script = os.path.join(_REPO_ROOT, "training", "playchess_selfplay.py")

    for game_num in range(args.start_game, args.num_games + 1):
        print(f"Playing game {game_num}/{args.num_games}")

        command = (
            f"{sys.executable} {play_script} --model {args.model} --mode s --color w "
            f"--rollouts {args.rollouts} --threads {args.threads} --verbose"
        )
        child = pexpect.spawn(command)
        child.logfile = None

        # Wait for the game to end
        child.expect_exact("Game over. Winner:", timeout=None)
        print("Game over.")

        print("Generating PGN...")

        # Extract the moves from the child's output and create a PGN
        pgn = chess.pgn.Game()
        node = pgn
        for line in child.before.decode("utf-8").splitlines():
            if line.startswith("best move"):
                move_uci = line.split()[-1]
                move = chess.Move.from_uci(move_uci)
                node = node.add_variation(move)
            elif "Game over. Winner:" in line:
                result = line.split(":")[-1].strip()
                node.comment = result

        pgn_file_path = os.path.join(args.output_dir, f"{game_num}.pgn")
        with open(pgn_file_path, "w") as pgn_file:
            pgn_file.write(str(pgn))

        print(f"PGN generated for game {game_num}")


if __name__ == "__main__":
    main()
