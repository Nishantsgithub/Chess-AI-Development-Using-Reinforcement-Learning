import argparse
import chess
import MCTS
import torch
import AlphaZeroNetwork
import time
import numpy as np

# Define epsilon values
starting_epsilon = 0.2 # Starting epsilon value
min_epsilon = 0.1  # Minimum epsilon value

def tolist( move_generator ):
    """
    Change an iterable object of moves to a list of moves.
    
    Args:
        move_generator (Mainline object) iterable list of moves

    Returns:
        moves (list of chess.Move) list version of the input moves
    """
    moves = []
    for move in move_generator:
        moves.append( move )
    return moves

def main( modelFile, mode, color, num_rollouts, num_threads, fen, verbose ):
    
    #prepare neural network
    alphaZeroNet = AlphaZeroNetwork.AlphaZeroNet( 20, 256 )

    #toggle for cpu/gpu
    cuda = False
    if cuda:
        weights = torch.load( modelFile )
    else:
        weights = torch.load( modelFile, map_location=torch.device('cpu') )

    alphaZeroNet.load_state_dict( weights )

    if cuda:
        alphaZeroNet = alphaZeroNet.cuda()

    for param in alphaZeroNet.parameters():
        param.requires_grad = False

    alphaZeroNet.eval()
   
    #create chess board object
    if fen:
        board = chess.Board( fen )
    else:
        board = chess.Board()

    #play chess moves
    while True:

        if board.is_game_over():
            #If the game is over, output the winner and wait for user input to continue
            print( 'Game over. Winner: {}'.format( board.result() ) )
            board.reset_board()
            c = input( 'Enter any key to continue ' )

        #Print the current state of the board
        if board.turn:
            print( 'White\'s turn' )
        else:
            print( 'Black\'s turn' )
        print( board )

        if mode == 'h' and board.turn == color:
            #If we are in human mode and it is the humans turn, play the move specified from stdin
            move_list = tolist( board.legal_moves )

            idx = -1

            while not (0 <= idx and idx < len( move_list ) ):
            
                string = input( 'Choose a move ' )

                for i, move in enumerate( move_list ):
                    if str( move ) == string:
                        idx = i
                        break
            
            board.push( move_list[ idx ] )

        else:
            # AI selects the next move

            starttime = time.perf_counter()

            with torch.no_grad():
                root = MCTS.Root(board, alphaZeroNet)

                for i in range(num_rollouts):
                    root.parallelRollouts(board.copy(), alphaZeroNet, num_threads)

            endtime = time.perf_counter()

            elapsed = endtime - starttime

            Q = root.getQ()
            N = root.getN()
            nps = N / elapsed
            same_paths = root.same_paths

            if verbose:
                # Print statistics
                print(root.getStatisticsString())
                print('total rollouts {} Q {:0.3f} duplicate paths {} elapsed {:0.2f} nps {:0.2f}'.format(
                    int(N), Q, same_paths, elapsed, nps))

            # Calculate the epsilon value for this move
            game_progress = len(board.move_stack)  # Number of moves played so far
            if game_progress <= 5:  
                epsilon = max(starting_epsilon - game_progress * 0.1, min_epsilon)
            else:
                epsilon = 0.02  # Or consider a further decay mechanism


            # Check for 3-fold repetition while "winning" (Q > 0)
            best_edges = [edge for edge in root.edges if edge.getQ() == max([e.getQ() for e in root.edges])]
            if Q > 0.7:  
                non_repeating_edges = []
                for edge in best_edges:
                    temp_board = board.copy()
                    temp_board.push(edge.getMove())
                    if temp_board.fen() not in root.history:  # Make sure your MCTS.Root class maintains this history
                        non_repeating_edges.append(edge)
                if non_repeating_edges:
                    best_edges = non_repeating_edges
                
            # Find the move with the highest Q value (exploitation)
            bestmove = None
            best_q = float('-inf')
            for edge in root.edges:
                if edge.getQ() > best_q:
                    best_q = edge.getQ()
                    bestmove = edge.getMove()

            # Create a list of candidate moves for epsilon-greedy policy
            candidate_moves = []
            for edge in root.edges:
                q_value = edge.getQ()
                if q_value >= best_q - epsilon and q_value <= best_q + epsilon:
                    candidate_moves.append(edge.getMove())


            # Choose a move using epsilon-greedy policy
            if candidate_moves:
                selected_move = np.random.choice(candidate_moves)
                print('best move:', selected_move)
            else:
                selected_move = bestmove
                print('best move:', selected_move)

            board.push(selected_move)

        if mode == 'p':
            #In profile mode, exit after the first move
            break

def parseColor( colorString ):
    """
    Maps 'w' to True and 'b' to False.

    Args:
        colorString (string) a string representing white or black

    """

    if colorString == 'w' or colorString == 'W':
        return True
    elif colorString == 'b' or colorString == 'B':
        return False
    else:
        print( 'Unrecognized argument for color' )
        exit()

if __name__=='__main__':
    parser = argparse.ArgumentParser(usage='Play chess against the computer or watch self play games.')
    parser.add_argument( '--model', help='Path to model (.pt) file.' )
    parser.add_argument( '--mode', help='Operation mode: \'s\' self play, \'p\' profile, \'h\' human' )
    parser.add_argument( '--color', help='Your color w or b' )
    parser.add_argument( '--rollouts', type=int, help='The number of rollouts on computers turn' )
    parser.add_argument( '--threads', type=int, help='Number of threads used per rollout' )
    parser.add_argument( '--verbose', help='Print search statistics', action='store_true' )
    parser.add_argument( '--fen', help='Starting fen' )
    parser.set_defaults( verbose=False, mode='p', color='w', rollouts=10, threads=1 )
    parser = parser.parse_args()

    main( parser.model, parser.mode, parseColor( parser.color ), parser.rollouts, parser.threads, parser.fen, parser.verbose )