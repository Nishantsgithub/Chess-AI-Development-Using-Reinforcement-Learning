import argparse
import os
import subprocess
import chess.pgn
import pexpect

def main():
    # Adjust the model path
    model_path = "/data/acp22np/ScalableML/Alpha-zero/weights/human_20x256.pt"
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Self-play script")
    parser.add_argument("--num_games", type=int, default=25000, help="Number of games to play")
    args = parser.parse_args()
    
    directory = "/data/acp22np/ScalableML/Alpha-zero/selfplay15/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for game_num in range(20283, args.num_games + 1):  # Start from 1
        print(f"Playing game {game_num}/{args.num_games}")
        
        # Call the playchess.py script with the necessary arguments and interact using pexpect
        command = f"python /data/acp22np/ScalableML/Alpha-zero/playchess.py --model {model_path} --mode s --color w --rollouts 20 --threads 6 --verbose"
        child = pexpect.spawn(command)
        child.logfile = None  # Disable logging to a file
        
        # Wait for the game to end
        child.expect_exact("Game over. Winner:", timeout=None)
        print("Game over.")
        
        # Print a message indicating that the PGN file is being generated
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
        
        # Save the PGN to a file
        pgn_file_name = f"{game_num}.pgn"
        pgn_file_path = os.path.join(directory, pgn_file_name)
        with open(pgn_file_path, "w") as pgn_file:
            pgn_file.write(str(pgn))
            
        print(f"PGN generated for game {game_num}")

if __name__ == "__main__":
    main()
