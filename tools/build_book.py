"""
Distill an opening book from the model's own deep searches.

Walks the opening tree: at engine-to-move positions it runs a long MCTS
search (ideally on GPU) and records the chosen move; at opponent-to-move
positions it branches over the opponent's most likely replies according to
the model's own policy prior. Covers the engine playing both colors.

The result (weights/opening_book.json) lets a slow CPU host play the
opening phase instantly at full search depth.

Usage:
    python tools/build_book.py [--budget 20] [--depth 8] [--replies 2]
"""

import argparse
import json
import os
import sys
import time

import chess
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'app'))

import encoder
from engine import Engine


def book_key(board):
    # FEN without move counters, so transpositions collapse.
    return ' '.join(board.fen().split()[:4])


def main():
    parser = argparse.ArgumentParser(description="Distill an opening book from deep searches")
    parser.add_argument('--weights', default=os.path.join(_REPO_ROOT, 'weights', 'HPC_20x256.pt'))
    parser.add_argument('--budget', type=float, default=20, help='Seconds per book position')
    parser.add_argument('--depth', type=int, default=8, help='Plies to cover from the start')
    parser.add_argument('--replies', type=int, default=2, help='Opponent replies to branch on')
    parser.add_argument('--root-replies', type=int, default=4,
                        help='Opponent replies at the very first move (humans open '
                             'with e4/d4/c4/Nf3 regardless of what the model prefers)')
    parser.add_argument('--out', default=os.path.join(_REPO_ROOT, 'weights', 'opening_book.json'))
    args = parser.parse_args()

    engine = Engine(args.weights)
    book = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            book = json.load(f)
        print(f'Resuming: {len(book)} positions already in book')

    stats = {'searched': 0, 'start': time.time()}

    def engine_move(board):
        key = book_key(board)
        if key in book:
            return chess.Move.from_uci(book[key])
        move, info = engine.think(board.copy(), args.budget)
        book[key] = move.uci()
        stats['searched'] += 1
        elapsed = time.time() - stats['start']
        print(f"[{stats['searched']:3d}] {elapsed:6.0f}s  ply {len(board.move_stack)}  "
              f"{move.uci()}  ({info['rollouts']} rollouts)  {board.fen().split()[0][:30]}")
        with open(args.out, 'w') as f:
            json.dump(book, f)
        return move

    def opponent_replies(board):
        value, probs = encoder.callNeuralNetwork(board, engine.net)
        moves = list(board.legal_moves)
        order = np.argsort(probs[:len(moves)])[::-1]
        n = args.root_replies if len(board.move_stack) == 0 else args.replies
        replies = [moves[i] for i in order[:n]]
        if len(board.move_stack) == 0:
            # Always cover the openings humans actually play, even if the
            # model's own prior underrates them.
            for uci in ('e2e4', 'd2d4', 'c2c4', 'g1f3'):
                move = chess.Move.from_uci(uci)
                if move not in replies:
                    replies.append(move)
        return replies

    def walk(board, engine_to_move, depth_left):
        if depth_left <= 0 or board.is_game_over():
            return
        if engine_to_move:
            move = engine_move(board)
            board.push(move)
            walk(board, False, depth_left - 1)
            board.pop()
        else:
            for reply in opponent_replies(board):
                board.push(reply)
                walk(board, True, depth_left - 1)
                board.pop()

    print('=== engine as White ===')
    walk(chess.Board(), True, args.depth)
    print('=== engine as Black ===')
    walk(chess.Board(), False, args.depth)

    print(f'Done: {len(book)} positions -> {args.out}')


if __name__ == '__main__':
    main()
