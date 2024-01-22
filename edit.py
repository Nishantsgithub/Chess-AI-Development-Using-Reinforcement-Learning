import os
import chess.pgn

def determine_winner(board):
    if board.is_checkmate():
        return "1-0" if board.turn == chess.BLACK else "0-1"
    else:
        return "1/2-1/2"

def fix_pgn_results(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
    input_dir = "/data/acp22np/ScalableML/Alpha-zero/selfplay15/"  # Update with your input directory
    output_dir = "/data/acp22np/ScalableML/Alpha-zero/selfplay20/"  # Update with your output directory
    fix_pgn_results(input_dir, output_dir)
