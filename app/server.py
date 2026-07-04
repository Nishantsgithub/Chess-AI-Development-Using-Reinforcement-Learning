"""
Web app for playing timed games against the dissertation's chess model.

Local use:
    python app/server.py [--model weights/HPC_20x256.pt] [--port 8000]

Deployment (e.g. Hugging Face Spaces): set DEPLOYED=1 — the server then
binds 0.0.0.0, honours the PORT env var (default 7860), never opens a
browser, and clamps think times for shared CPU hosting.

Every visitor gets their own game (a game_id issued by /api/new and sent
back with every request). Games idle for a few hours are purged.
"""

import argparse
import datetime
import os
import random
import sys
import threading
import time
import uuid
import webbrowser
from collections import deque

import chess
import chess.pgn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_APP_DIR)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from engine import Engine

DEPLOYED = bool(os.environ.get('DEPLOYED') or os.environ.get('SPACE_ID'))
MAX_ENGINE_SECONDS = 20 if DEPLOYED else 60   # "full power" cap per move
MAX_MINUTES = 60 if DEPLOYED else 180
MAX_GAMES = 40                                 # concurrent stored games
GAME_TTL_S = 3 * 3600                          # purge games idle this long
NEW_GAMES_PER_IP = 12                          # per rate window
RATE_WINDOW_S = 600
# How many games may be in progress at once. 1 on shared hosting: whoever
# is playing gets the engine's full strength; others see a waiting screen.
MAX_ACTIVE_GAMES = int(os.environ.get('MAX_ACTIVE_GAMES', 1 if DEPLOYED else 40))
ACTIVE_WINDOW_S = 180                          # no heartbeat for this long = abandoned

app = FastAPI(title='Chess vs Hybrid RL Model')

ENGINE = None  # loaded in main()
GAMES = {}     # game_id -> Game
GAMES_LOCK = threading.Lock()
RATE = {}      # ip -> deque of new-game timestamps


class Game:

    def __init__(self, minutes, increment_s, human_color, engine_seconds=0):
        self.id = uuid.uuid4().hex
        self.board = chess.Board()
        self.clocks = {
            chess.WHITE: int(minutes * 60 * 1000),
            chess.BLACK: int(minutes * 60 * 1000),
        }
        self.increment_ms = int(increment_s * 1000)
        self.human = human_color
        # engine_seconds > 0 = "full power": fixed think time per move and
        # no clocks for either side.
        self.engine_seconds = engine_seconds
        self.untimed = engine_seconds > 0
        self.turn_start = time.monotonic()
        self.last_activity = time.monotonic()
        self.san_moves = []
        self.fen_history = [self.board.fen()]   # position before/after each ply
        self.uci_history = []
        self.over = None            # {'result': '1-0', 'reason': 'checkmate'}
        self.eval_white = 0.5       # white win probability in [0, 1]
        self.last_search = None     # stats from the engine's last move
        self.thinking_ply = None    # ply an engine search is running for
        self.lock = threading.Lock()

    def touch(self):
        self.last_activity = time.monotonic()

    # ---- clock helpers -------------------------------------------------

    def _elapsed_ms(self):
        return int((time.monotonic() - self.turn_start) * 1000)

    def clock_snapshot(self):
        """Clocks with the running side's elapsed time already deducted."""
        if self.untimed:
            return {'white': None, 'black': None, 'running': None}
        snap = dict(self.clocks)
        if self.over is None:
            snap[self.board.turn] = snap[self.board.turn] - self._elapsed_ms()
        return {
            'white': max(snap[chess.WHITE], 0),
            'black': max(snap[chess.BLACK], 0),
            'running': (None if self.over else
                        ('white' if self.board.turn else 'black')),
        }

    def check_flag(self):
        """If the side to move ran out of time, end the game."""
        if self.over is not None or self.untimed:
            return
        if self.clocks[self.board.turn] - self._elapsed_ms() <= 0:
            self.clocks[self.board.turn] = 0
            loser_is_white = self.board.turn == chess.WHITE
            self.over = {
                'result': '0-1' if loser_is_white else '1-0',
                'reason': 'timeout',
            }

    # ---- moves ---------------------------------------------------------

    def apply_move(self, move):
        """Push a legal move, charging the mover's clock."""
        mover = self.board.turn
        if not self.untimed:
            elapsed = self._elapsed_ms()
            if self.clocks[mover] - elapsed <= 0:
                self.check_flag()
                return
            self.clocks[mover] -= elapsed
            self.clocks[mover] += self.increment_ms
        self.san_moves.append(self.board.san(move))
        self.board.push(move)
        self.fen_history.append(self.board.fen())
        self.uci_history.append(move.uci())
        self.turn_start = time.monotonic()
        self._check_game_end()

    def _check_game_end(self):
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None:
            self.over = {
                'result': outcome.result(),
                'reason': outcome.termination.name.lower().replace('_', ' '),
            }

    # ---- serialization ---------------------------------------------------

    def state(self):
        board = self.board
        check_square = None
        if board.is_check():
            check_square = chess.square_name(board.king(board.turn))
        last_move = board.peek().uci() if board.move_stack else None
        return {
            'game_id': self.id,
            'fen': board.fen(),
            'turn': 'white' if board.turn else 'black',
            'human_color': 'white' if self.human else 'black',
            'legal_moves': [m.uci() for m in board.legal_moves] if not self.over else [],
            'san_moves': self.san_moves,
            'fens': self.fen_history,
            'ucis': self.uci_history,
            'last_move': last_move,
            'check_square': check_square,
            'clocks': self.clock_snapshot(),
            'increment': self.increment_ms // 1000,
            'over': self.over,
            'eval_white': self.eval_white,
            'last_search': self.last_search,
            'model': ENGINE.weights_name,
        }


class NewGameRequest(BaseModel):
    minutes: float
    increment: float = 0
    color: str = 'white'  # 'white' | 'black' | 'random'
    engine_seconds: float = 0  # 0 = clock-paced; >0 = full power (untimed)
    replace_game_id: str = ''  # the caller's previous game, freed on restart


class MoveRequest(BaseModel):
    game_id: str
    uci: str


class GameRequest(BaseModel):
    game_id: str


def _client_ip(request):
    fwd = request.headers.get('x-forwarded-for')
    if fwd:
        return fwd.split(',')[0].strip()
    return request.client.host if request.client else 'unknown'


def _rate_ok(ip):
    now = time.time()
    q = RATE.setdefault(ip, deque())
    while q and now - q[0] > RATE_WINDOW_S:
        q.popleft()
    if len(q) >= NEW_GAMES_PER_IP:
        return False
    q.append(now)
    return True


def _purge_games_locked():
    """Drop idle games; caller holds GAMES_LOCK."""
    now = time.monotonic()
    stale = [gid for gid, g in GAMES.items() if now - g.last_activity > GAME_TTL_S]
    for gid in stale:
        del GAMES[gid]
        ENGINE.forget(gid)
    while len(GAMES) >= MAX_GAMES:
        oldest = min(GAMES.values(), key=lambda g: g.last_activity)
        del GAMES[oldest.id]
        ENGINE.forget(oldest.id)


def _get_game(game_id):
    with GAMES_LOCK:
        game = GAMES.get(game_id or '')
    if game is None:
        raise HTTPException(status_code=404, detail='No such game (it may have expired)')
    game.touch()
    return game


def _close_abandoned_locked():
    """
    Close in-progress games whose player has gone quiet (no heartbeat),
    so ghosts don't hold the active-game slots. Caller holds GAMES_LOCK.
    """
    now = time.monotonic()
    for g in GAMES.values():
        if g.over is None and now - g.last_activity > ACTIVE_WINDOW_S:
            g.over = {'result': '*', 'reason': 'abandoned'}


def _active_count_locked():
    return sum(1 for g in GAMES.values() if g.over is None)


@app.get('/api/availability')
def availability():
    with GAMES_LOCK:
        _close_abandoned_locked()
        busy = _active_count_locked() >= MAX_ACTIVE_GAMES
    return {'busy': busy}


@app.post('/api/new')
def new_game(req: NewGameRequest, request: Request):
    if not _rate_ok(_client_ip(request)):
        raise HTTPException(status_code=429, detail='Too many new games; try again later')
    if not (0.1 <= req.minutes <= MAX_MINUTES):
        raise HTTPException(status_code=400, detail=f'Minutes must be between 0.1 and {MAX_MINUTES}')
    if not (0 <= req.increment <= 60):
        raise HTTPException(status_code=400, detail='Increment must be between 0 and 60')
    if not (0 <= req.engine_seconds <= 120):
        raise HTTPException(status_code=400, detail='Engine seconds out of range')
    engine_seconds = min(req.engine_seconds, MAX_ENGINE_SECONDS)
    color = req.color
    if color == 'random':
        color = random.choice(['white', 'black'])
    game = Game(req.minutes, req.increment, human_color=(color == 'white'),
                engine_seconds=engine_seconds)
    with GAMES_LOCK:
        # Starting over frees the caller's own previous game first.
        prev = GAMES.get(req.replace_game_id)
        if prev is not None and prev.over is None:
            prev.over = {'result': '*', 'reason': 'abandoned'}
        _close_abandoned_locked()
        if _active_count_locked() >= MAX_ACTIVE_GAMES:
            raise HTTPException(
                status_code=503,
                detail='The model is playing someone else right now — hold on, '
                       'your game will start as soon as the board frees up.')
        _purge_games_locked()
        GAMES[game.id] = game
    return game.state()


@app.get('/api/state')
def get_state(game_id: str = ''):
    game = _get_game(game_id)
    with game.lock:
        game.check_flag()
        return game.state()


@app.post('/api/move')
def player_move(req: MoveRequest):
    game = _get_game(req.game_id)
    with game.lock:
        game.check_flag()
        if game.over:
            return game.state()
        if game.board.turn != game.human:
            raise HTTPException(status_code=409, detail='Not your turn')
        try:
            move = chess.Move.from_uci(req.uci)
        except ValueError:
            raise HTTPException(status_code=400, detail='Bad move format')
        if move not in game.board.legal_moves:
            raise HTTPException(status_code=400, detail='Illegal move')
        game.apply_move(move)
    if not game.over:
        game.eval_white = _white_eval(game.board)
    return game.state()


@app.post('/api/engine-move')
def engine_move(req: GameRequest):
    game = _get_game(req.game_id)
    with game.lock:
        game.check_flag()
        if game.over:
            return game.state()
        if game.board.turn == game.human:
            raise HTTPException(status_code=409, detail="It is the human's turn")
        board = game.board.copy()
        if game.engine_seconds > 0:
            budget = game.engine_seconds
        else:
            remaining = game.clocks[game.board.turn] - game._elapsed_ms()
            budget = Engine.budget_for_move(remaining, game.increment_ms,
                                            cap=MAX_ENGINE_SECONDS)
        ply = len(game.board.move_stack)
        recent_ucis = game.uci_history[-2:] if len(game.uci_history) >= 2 else None
        if game.thinking_ply == ply:
            # e.g. the page was refreshed mid-search; the original request
            # is still computing. Tell the client to poll instead.
            state = game.state()
            state['already_thinking'] = True
            return state
        game.thinking_ply = ply

    # Search outside the lock: it can take many seconds and the game is
    # turn-based, so nothing else mutates this game's position meanwhile.
    try:
        move, info = ENGINE.think(board, budget, recent_ucis=recent_ucis,
                                  reuse_key=game.id)
    except BaseException:
        with game.lock:
            if game.thinking_ply == ply:
                game.thinking_ply = None
        raise

    with game.lock:
        if game.thinking_ply == ply:
            game.thinking_ply = None
        # The game may have ended (flag/resign) while we were thinking.
        if len(game.board.move_stack) != ply:
            return game.state()
        game.check_flag()
        if game.over:
            return game.state()
        game.apply_move(move)
        if not game.over:
            engine_is_white = not game.human
            q = info['q']
            game.eval_white = q if engine_is_white else 1.0 - q
        game.last_search = {
            'move': move.uci(),
            'rollouts': info['rollouts'],
            'carried': info['carried'],
            'elapsed': round(info['elapsed'], 2),
            'nps': round(info['nps'], 1),
            'budget': round(budget, 2),
        }
        return game.state()


@app.get('/api/pgn')
def download_pgn(game_id: str = ''):
    game = _get_game(game_id)
    with game.lock:
        pgn = chess.pgn.Game()
        model_name = ENGINE.weights_name.replace('.pt', '')
        pgn.headers['Event'] = 'Hybrid RL Chess'
        pgn.headers['Site'] = 'Chess-AI-Development-Using-Reinforcement-Learning'
        pgn.headers['Date'] = datetime.date.today().strftime('%Y.%m.%d')
        pgn.headers['White'] = 'Human' if game.human == chess.WHITE else model_name
        pgn.headers['Black'] = 'Human' if game.human == chess.BLACK else model_name
        pgn.headers['Result'] = game.over['result'] if game.over else '*'
        pgn.headers['TimeControl'] = ('-' if game.untimed else
                                      f'{game.clocks[chess.WHITE] // 1000}+{game.increment_ms // 1000}')
        node = pgn
        for uci in game.uci_history:
            node = node.add_variation(chess.Move.from_uci(uci))
        return PlainTextResponse(
            str(pgn),
            media_type='application/x-chess-pgn',
            headers={'Content-Disposition': 'attachment; filename="hybrid-rl-game.pgn"'},
        )


@app.post('/api/resign')
def resign(req: GameRequest):
    game = _get_game(req.game_id)
    with game.lock:
        if not game.over:
            human_is_white = game.human == chess.WHITE
            game.over = {
                'result': '0-1' if human_is_white else '1-0',
                'reason': 'resignation',
            }
        return game.state()


def _white_eval(board):
    """White's win probability according to a single network call."""
    q = ENGINE.evaluate(board)
    return q if board.turn == chess.WHITE else 1.0 - q


@app.get('/')
def index():
    return FileResponse(os.path.join(_APP_DIR, 'static', 'index.html'))


app.mount('/static', StaticFiles(directory=os.path.join(_APP_DIR, 'static')), name='static')
app.mount('/pieces', StaticFiles(directory=os.path.join(_REPO_ROOT, 'images')), name='pieces')


def main():
    global ENGINE
    parser = argparse.ArgumentParser(description='Play timed chess against the trained model.')
    parser.add_argument('--model', default=os.path.join(_REPO_ROOT, 'weights', 'HPC_20x256.pt'),
                        help='Path to a .pt weights file (default: weights/HPC_20x256.pt)')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 7860 if DEPLOYED else 8000)))
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    print(f'Loading model {args.model} ...')
    ENGINE = Engine(args.model)
    print('Model loaded.' + (' (deployment mode)' if DEPLOYED else ''))

    host = '0.0.0.0' if DEPLOYED else '127.0.0.1'
    if not DEPLOYED and not args.no_browser:
        url = f'http://127.0.0.1:{args.port}'
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run(app, host=host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
