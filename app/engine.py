"""
Clock-aware wrapper around the dissertation's AlphaZero-style network + MCTS.

MCTS is an anytime algorithm: we keep running rollout batches until the
per-move time budget is spent, then play the most-visited move. The budget
is derived from the engine's remaining clock, so a 30-minute game gets a
deeper (stronger) search than a 5-minute game, just like a human paces
themselves.

Two strength boosters on top of the original code:
- GPU inference when a CUDA build of PyTorch is available.
- Search-tree reuse: the subtree explored for the previous move is carried
  over instead of starting every search from scratch.
"""

import os
import sys
import time
import threading

import chess
import torch

# The core modules (MCTS, encoder, AlphaZeroNetwork) live in the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import MCTS
import AlphaZeroNetwork
import encoder


class Engine:

    def __init__(self, weights_path, num_blocks=20, num_filters=256):
        self.cuda = torch.cuda.is_available()
        onnx_path = os.path.splitext(weights_path)[0] + '.onnx'
        self.onnx = not self.cuda and self._try_onnx(onnx_path)

        if self.onnx:
            print(f'Using ONNX Runtime (CPU): {onnx_path}')
        else:
            self.net = AlphaZeroNetwork.AlphaZeroNet(num_blocks, num_filters)
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            self.net.load_state_dict(state_dict)
            self.net.eval()
            for param in self.net.parameters():
                param.requires_grad = False
            if self.cuda:
                self.net = self.net.cuda()
                encoder.cuda = True  # encoder moves inputs to the GPU
                print(f'Using GPU: {torch.cuda.get_device_name(0)}')
            else:
                print('Using CPU eager torch (no .onnx found; '
                      'run tools/export_onnx.py for faster CPU inference)')

        # Best measured batch for unique rollouts/sec on each device.
        self.batch_size = 32 if self.cuda else 16
        self.weights_name = os.path.basename(weights_path)

        # Opening book distilled from the model's own deep GPU searches
        # (tools/build_book.py) — instant, full-strength opening play.
        self.book = {}
        book_path = os.path.join(os.path.dirname(os.path.abspath(weights_path)),
                                 'opening_book.json')
        if os.path.exists(book_path):
            import json
            with open(book_path) as f:
                self.book = json.load(f)
            print(f'Opening book loaded: {len(self.book)} positions')

        # Searches are serialized; concurrent games queue on this lock.
        self.lock = threading.Lock()
        # Per-game search trees for reuse: reuse_key -> (root, fen).
        self._trees = {}
        self._tree_cap = 48
        # How many think() calls are queued/running, for load-aware budgets.
        self._inflight = 0
        self._meta_lock = threading.Lock()

    def _book_move(self, board):
        if not self.book or len(board.move_stack) > 16:
            return None
        key = ' '.join(board.fen().split()[:4])
        uci = self.book.get(key)
        if uci is None:
            return None
        move = chess.Move.from_uci(uci)
        return move if move in board.legal_moves else None

    def _try_onnx(self, onnx_path):
        if not os.path.exists(onnx_path):
            return False
        try:
            from onnx_net import OnnxAlphaZero
            self.net = OnnxAlphaZero(onnx_path)
            return True
        except ImportError:
            return False

    def forget(self, reuse_key):
        """Drop the carried-over search tree for one game."""
        with self._meta_lock:
            self._trees.pop(reuse_key, None)

    def evaluate(self, board):
        """
        Single network call. Returns the win probability in [0, 1]
        from the perspective of the side to move.
        """
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            winner = encoder.parseResult(result)
            if not board.turn:
                winner *= -1
            return float(winner) / 2.0 + 0.5
        with torch.no_grad(), self.lock:
            value, _ = encoder.callNeuralNetwork(board, self.net)
        return float(value) / 2.0 + 0.5

    @staticmethod
    def budget_for_move(remaining_ms, increment_ms=0, cap=60.0):
        """
        Decide how long to think given the remaining clock.
        Roughly 1/24th of the remaining time plus most of the increment,
        clamped so the engine never stalls nor flags itself.
        """
        remaining_s = max(remaining_ms, 0) / 1000.0
        increment_s = increment_ms / 1000.0
        budget = remaining_s / 24.0 + increment_s * 0.75
        # In time trouble, move nearly instantly but keep a floor of one batch.
        ceiling = max(remaining_s * 0.5, 0.1)
        return max(0.15, min(budget, cap, ceiling))

    def think(self, board, budget_seconds, batch_size=None, recent_ucis=None,
              reuse_key=None):
        """
        Run MCTS rollouts until the budget is spent, then choose a move.

        recent_ucis: the moves played since the previous search (typically
        [our last move, opponent's reply]) — used to descend the stored
        tree so earlier work is reused.
        reuse_key: identifies the game the stored tree belongs to.

        Returns (move, info) where info holds search statistics for the UI.
        """
        book_move = self._book_move(board)
        if book_move is not None:
            return book_move, {
                'q': 0.5, 'root_q': 0.5, 'rollouts': 0, 'carried': 0,
                'elapsed': 0.0, 'queue': 0, 'nps': 0.0, 'book': True,
            }

        if batch_size is None:
            batch_size = self.batch_size
        with self._meta_lock:
            self._inflight += 1
            queue_depth = self._inflight
        try:
            # Under load, shrink each game's budget so no one waits forever.
            budget = max(1.5, budget_seconds / queue_depth)
            wait_start = time.perf_counter()
            with torch.no_grad(), self.lock:
                # The clock starts once it is our turn on the CPU.
                start = time.perf_counter()
                deadline = start + budget
                root = self._reused_root(board, recent_ucis, reuse_key)
                carried = int(root.getN()) if root is not None else 0
                if root is None:
                    root = MCTS.Root(board, self.net)
                batch_time = 0.0
                while time.perf_counter() + batch_time < deadline and not root.isTerminal():
                    t0 = time.perf_counter()
                    root.parallelRollouts(board.copy(), self.net, batch_size)
                    batch_time = time.perf_counter() - t0
                if reuse_key is not None:
                    with self._meta_lock:
                        self._trees[reuse_key] = (root, board.fen())
                        while len(self._trees) > self._tree_cap:
                            self._trees.pop(next(iter(self._trees)))
        finally:
            with self._meta_lock:
                self._inflight -= 1

        edge = self._select_edge(board, root)
        elapsed = time.perf_counter() - wait_start
        rollouts = int(root.getN())
        info = {
            'q': float(edge.getQ()),          # win prob for the engine after this move
            'root_q': float(root.getQ()),     # win prob for side to move at root
            'rollouts': rollouts,
            'carried': carried,               # rollouts inherited from the last search
            'elapsed': elapsed,
            'queue': queue_depth,
            'nps': (rollouts - carried) / elapsed if elapsed > 0 else 0.0,
        }
        return edge.getMove(), info

    def _reused_root(self, board, recent_ucis, reuse_key):
        """
        Descend the stored tree along the moves played since the last
        search. Returns a Root grafted with the reused subtree, or None.
        """
        if reuse_key is None or not recent_ucis:
            return None
        with self._meta_lock:
            prev = self._trees.get(reuse_key)
        if prev is None:
            return None
        prev_root, prev_fen = prev
        try:
            replay = chess.Board(prev_fen)
            node = prev_root
            for uci in recent_ucis:
                move = chess.Move.from_uci(uci)
                found = None
                for e in node.edges:
                    if e.getMove() == move:
                        found = e.getChild()
                        break
                if found is None:
                    return None
                replay.push(move)
                node = found
            if replay.fen() != board.fen() or node.isTerminal():
                return None
        except Exception:
            return None
        # Graft the subtree onto a fresh Root for the current position.
        root = MCTS.Root(board, self.net)
        root.edges = node.edges
        root.N = node.N
        root.sum_Q = node.sum_Q
        return root

    def _select_edge(self, board, root):
        """
        Most-visited edge, with one refinement from the dissertation's
        future-work list: when clearly winning, avoid moves that walk into
        a draw by repetition if a near-equal alternative exists.
        """
        edges = sorted(root.edges, key=lambda e: e.getN(), reverse=True)
        best = edges[0]
        if best.getQ() > 0.6:
            if self._repeats(board, best.getMove()):
                for alt in edges[1:]:
                    close_enough = best.getQ() - alt.getQ() < 0.05 and alt.getN() > 0
                    if close_enough and not self._repeats(board, alt.getMove()):
                        return alt
        return best

    @staticmethod
    def _repeats(board, move):
        child = board.copy()
        child.push(move)
        return child.is_repetition(2) or child.can_claim_threefold_repetition()
