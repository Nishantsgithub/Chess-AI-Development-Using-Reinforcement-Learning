/* Chess vs Hybrid RL Model — frontend */

const $ = (id) => document.getElementById(id);

let S = null;                 // last server state
let clockSyncAt = 0;          // performance.now() when S.clocks was received
let selected = null;          // selected square name, e.g. "e2"
let enginePending = false;
let flagPending = false;
let dragGhost = null;
let gameSeq = 0;              // bumped on every new game to invalidate stale fetches
let viewPly = null;           // null = live position; otherwise the ply being reviewed
let premove = null;           // {from, to} queued while the model thinks
let orientation = 'white';    // side shown at the bottom of the board
let arrows = [];              // [{from, to}] right-drag annotations
let marks = new Set();        // right-click square highlights
let rightDragFrom = null;
let lowTimeWarned = false;
let lastGameSettings = null;  // for rematch
let gameId = null;            // this browser's game on the (multi-user) server

const FILES = 'abcdefgh';

/* ---------------- settings (persisted) ---------------- */

const THEMES = {
  classic: { light: '#F0D9B5', dark: '#B58863' },
  green:   { light: '#EBECD0', dark: '#779556' },
  blue:    { light: '#DEE3E6', dark: '#8CA2AD' },
  slate:   { light: '#B8C0CC', dark: '#5E6B7E' },
};

const settings = { theme: 'classic', coords: true, sound: true };

function loadSettings() {
  try {
    Object.assign(settings, JSON.parse(localStorage.getItem('hybridrl-settings') || '{}'));
  } catch (e) { /* defaults */ }
}

function saveSettings() {
  localStorage.setItem('hybridrl-settings', JSON.stringify(settings));
}

function applySettingsUI() {
  const theme = THEMES[settings.theme] || THEMES.classic;
  document.documentElement.style.setProperty('--light-sq', theme.light);
  document.documentElement.style.setProperty('--dark-sq', theme.dark);
  document.querySelectorAll('.swatch').forEach((b) =>
    b.classList.toggle('selected', b.dataset.theme === settings.theme));
  $('btn-sound').textContent = settings.sound ? '🔊' : '🔇';
  $('btn-sound').classList.toggle('off', !settings.sound);
  $('btn-coords').classList.toggle('off', !settings.coords);
}

/* ---------------- helpers ---------------- */

function humanIsWhite() { return S.human_color === 'white'; }
function bottomIsHuman() { return S ? orientation === S.human_color : true; }
function myTurn() { return S && !S.over && S.turn === S.human_color; }
function totalPlies() { return S ? S.san_moves.length : 0; }
function shownPly() { return viewPly === null ? totalPlies() : viewPly; }
function isLive() { return viewPly === null || viewPly === totalPlies(); }
function shownFen() { return S.fens ? S.fens[shownPly()] : S.fen; }

function pieceImg(ch) {
  const color = ch === ch.toUpperCase() ? 'w' : 'b';
  return `/pieces/${color}${ch.toUpperCase()}.png`;
}

function fenBoard(fen) {
  const board = {};
  const rows = fen.split(' ')[0].split('/');
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r]) {
      if (/\d/.test(ch)) { file += parseInt(ch, 10); }
      else { board[FILES[file] + (8 - r)] = ch; file++; }
    }
  }
  return board;
}

function fmtClock(ms) {
  if (ms === null || ms === undefined) return '∞';
  ms = Math.max(ms, 0);
  const total = Math.ceil(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  if (ms < 20000) {
    const tenths = Math.floor((ms % 1000) / 100);
    return `${m}:${String(s).padStart(2, '0')}.${tenths}`;
  }
  return `${m}:${String(s).padStart(2, '0')}`;
}

/* ---------------- sounds (synthesized, no assets) ---------------- */

let audioCtx = null;

function tone(freq, startDelay, dur, type, gain) {
  const t0 = audioCtx.currentTime + startDelay;
  const o = audioCtx.createOscillator();
  const g = audioCtx.createGain();
  o.type = type; o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(gain, t0 + 0.008);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(audioCtx.destination);
  o.start(t0); o.stop(t0 + dur + 0.02);
}

function playSound(kind) {
  if (!settings.sound) return;
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    switch (kind) {
      case 'move':    tone(340, 0, 0.07, 'triangle', 0.22); break;
      case 'capture': tone(220, 0, 0.09, 'triangle', 0.3); tone(110, 0.015, 0.09, 'sine', 0.2); break;
      case 'castle':  tone(300, 0, 0.06, 'triangle', 0.22); tone(300, 0.1, 0.06, 'triangle', 0.22); break;
      case 'check':   tone(587, 0, 0.09, 'square', 0.1); tone(784, 0.09, 0.13, 'square', 0.1); break;
      case 'promote': tone(523, 0, 0.09, 'sine', 0.2); tone(659, 0.08, 0.09, 'sine', 0.2); tone(784, 0.16, 0.14, 'sine', 0.2); break;
      case 'start':   tone(392, 0, 0.1, 'sine', 0.18); tone(523, 0.1, 0.16, 'sine', 0.18); break;
      case 'end':     tone(523, 0, 0.12, 'sine', 0.18); tone(392, 0.12, 0.12, 'sine', 0.16); tone(262, 0.24, 0.22, 'sine', 0.16); break;
      case 'lowtime': tone(880, 0, 0.06, 'square', 0.12); tone(880, 0.12, 0.06, 'square', 0.12); break;
    }
  } catch (e) { /* audio optional */ }
}

function playMoveSound(san) {
  if (!san) { playSound('move'); return; }
  if (san.includes('+') || san.includes('#')) playSound('check');
  else if (san.includes('=')) playSound('promote');
  else if (san.startsWith('O-O')) playSound('castle');
  else if (san.includes('x')) playSound('capture');
  else playSound('move');
}

/* ---------------- rendering ---------------- */

function render() {
  if (!S) return;
  renderBoard();
  renderClocks();
  renderMoves();
  renderEval();
  renderStats();
  renderCaptured();

  $('thinking').hidden = !(enginePending && !S.over);
  $('model-name').textContent = `${S.model} · MCTS`;

  const bottomHuman = bottomIsHuman();
  const bottomWhite = orientation === 'white';
  $('name-bottom').textContent = bottomHuman ? 'You' : 'Model';
  $('name-top').textContent = bottomHuman ? 'Model' : 'You';
  $('dot-bottom').className = 'dot' + (bottomWhite ? '' : ' black');
  $('dot-top').className = 'dot' + (bottomWhite ? ' black' : '');

  if (S.over) showGameOver();
}

function renderBoard() {
  const boardEl = $('board');
  boardEl.innerHTML = '';
  const ply = shownPly();
  const live = isLive();
  const pieces = fenBoard(shownFen());
  const flip = orientation === 'black';
  const shownMove = ply > 0 && S.ucis ? S.ucis[ply - 1] : null;
  const lastFrom = shownMove ? shownMove.slice(0, 2) : null;
  const lastTo = shownMove ? shownMove.slice(2, 4) : null;
  const targets = live && myTurn() ? legalTargetsFrom(selected) : new Set();

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const file = flip ? 7 - col : col;
      const rank = flip ? row + 1 : 8 - row;
      const sqName = FILES[file] + rank;
      const sq = document.createElement('div');
      // a1 is a dark square.
      const isLight = (file + rank) % 2 === 0;
      sq.className = 'sq ' + (isLight ? 'light' : 'dark');
      sq.dataset.sq = sqName;

      if (sqName === lastFrom || sqName === lastTo) sq.classList.add('last');
      if (live && sqName === selected) sq.classList.add('sel');
      if (live && sqName === S.check_square) sq.classList.add('check');
      if (live && premove && (sqName === premove.from || sqName === premove.to)) sq.classList.add('premove');
      if (marks.has(sqName)) sq.classList.add('mark');

      if (settings.coords) {
        if (col === 0) {
          const c = document.createElement('span');
          c.className = 'coord rank'; c.textContent = rank;
          sq.appendChild(c);
        }
        if (row === 7) {
          const c = document.createElement('span');
          c.className = 'coord file'; c.textContent = FILES[file];
          sq.appendChild(c);
        }
      }

      const piece = pieces[sqName];
      if (piece) {
        const img = document.createElement('img');
        img.src = pieceImg(piece);
        img.draggable = false;
        sq.appendChild(img);
      }

      if (targets.has(sqName)) {
        const hint = document.createElement('div');
        hint.className = 'hint ' + (piece ? 'capture' : 'move');
        sq.appendChild(hint);
      }

      sq.addEventListener('pointerdown', onSquareDown);
      sq.addEventListener('pointerup', onSquareUp);
      boardEl.appendChild(sq);
    }
  }
  renderArrows();
}

function sqCenter(sqName) {
  // percent coordinates of a square's center, orientation-aware
  const flip = orientation === 'black';
  const file = FILES.indexOf(sqName[0]);
  const rank = parseInt(sqName[1], 10);
  const col = flip ? 7 - file : file;
  const row = flip ? rank - 1 : 8 - rank;
  return { x: (col + 0.5) * 12.5, y: (row + 0.5) * 12.5 };
}

function renderArrows() {
  const boardEl = $('board');
  const old = boardEl.querySelector('.arrow-layer');
  if (old) old.remove();
  if (!arrows.length) return;
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('class', 'arrow-layer');
  svg.setAttribute('viewBox', '0 0 100 100');
  svg.setAttribute('preserveAspectRatio', 'none');
  for (const a of arrows) {
    const p1 = sqCenter(a.from);
    const p2 = sqCenter(a.to);
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const len = Math.hypot(dx, dy);
    if (len < 1) continue;
    const ux = dx / len, uy = dy / len;
    // shorten the shaft so the head lands on the target square center
    const hx = p2.x - ux * 3.4, hy = p2.y - uy * 3.4;
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', p1.x + ux * 2.5); line.setAttribute('y1', p1.y + uy * 2.5);
    line.setAttribute('x2', hx); line.setAttribute('y2', hy);
    line.setAttribute('class', 'arrow-shaft');
    svg.appendChild(line);
    const head = document.createElementNS(ns, 'polygon');
    const px = -uy, py = ux; // perpendicular
    head.setAttribute('points',
      `${p2.x},${p2.y} ${hx + px * 2.2},${hy + py * 2.2} ${hx - px * 2.2},${hy - py * 2.2}`);
    head.setAttribute('class', 'arrow-head');
    svg.appendChild(head);
  }
  boardEl.appendChild(svg);
}

function legalTargetsFrom(from) {
  const set = new Set();
  if (!from || !S) return set;
  for (const uci of S.legal_moves) {
    if (uci.slice(0, 2) === from) set.add(uci.slice(2, 4));
  }
  return set;
}

function renderClocks() {
  tickClocks(true);
}

function tickClocks(force) {
  if (!S) return;
  const elapsed = performance.now() - clockSyncAt;
  const run = S.clocks.running;
  const tick = (v, side) => (v === null ? null : v - (run === side ? elapsed : 0));
  const w = tick(S.clocks.white, 'white');
  const b = tick(S.clocks.black, 'black');
  const bottomWhite = orientation === 'white';
  const bottomMs = bottomWhite ? w : b;
  const topMs = bottomWhite ? b : w;

  setClock('time-bottom', bottomMs);
  setClock('time-top', topMs);

  const engineThinks = enginePending && !S.over;
  const bottomActive = run ? (run === 'white') === bottomWhite : false;
  const bottomUntimed = bottomMs === null && !bottomIsHuman() && engineThinks;
  const topUntimed = topMs === null && bottomIsHuman() && engineThinks;
  $('clock-bottom').classList.toggle('active', bottomActive || bottomUntimed);
  $('clock-top').classList.toggle('active', (!!run && !bottomActive) || topUntimed);

  // Low-time warning for the human, once per game.
  const humanMs = humanIsWhite() ? w : b;
  const humanRunning = run === S.human_color;
  if (humanRunning && humanMs !== null && humanMs < 10000 && !lowTimeWarned && !S.over) {
    lowTimeWarned = true;
    playSound('lowtime');
  }

  // Client noticed a flag -> ask the server to confirm.
  const lowest = Math.min(w === null ? Infinity : w, b === null ? Infinity : b);
  if (run && !flagPending && lowest <= 0) {
    flagPending = true;
    fetch(`/api/state?game_id=${gameId}`).then(r => r.json()).then(applyState)
      .finally(() => { flagPending = false; });
  }
}

function setClock(id, ms) {
  const el = $(id);
  el.textContent = fmtClock(ms);
  el.classList.toggle('low', ms !== null && ms < 30000);
}

/* ---------------- captured pieces / material ---------------- */

const START_COUNTS = { p: 8, n: 2, b: 2, r: 2, q: 1 };
const PIECE_VALUE = { p: 1, n: 3, b: 3, r: 5, q: 9 };
const CAP_ORDER = ['p', 'n', 'b', 'r', 'q'];

function renderCaptured() {
  const pieces = fenBoard(shownFen());
  const counts = { w: { p: 0, n: 0, b: 0, r: 0, q: 0 }, b: { p: 0, n: 0, b: 0, r: 0, q: 0 } };
  for (const ch of Object.values(pieces)) {
    const lower = ch.toLowerCase();
    if (lower === 'k') continue;
    counts[ch === ch.toUpperCase() ? 'w' : 'b'][lower]++;
  }
  const capturedBy = (side) => { // pieces this side has taken = opponent's missing
    const opp = side === 'w' ? 'b' : 'w';
    const out = [];
    let value = 0;
    for (const t of CAP_ORDER) {
      const missing = Math.max(START_COUNTS[t] - counts[opp][t], 0);
      for (let i = 0; i < missing; i++) out.push(opp === 'w' ? t.toUpperCase() : t);
      value += missing * PIECE_VALUE[t];
    }
    return { out, value };
  };
  const white = capturedBy('w');
  const black = capturedBy('b');
  const bottomWhite = orientation === 'white';
  fillCaptured('cap-bottom', bottomWhite ? white : black, bottomWhite ? black : white);
  fillCaptured('cap-top', bottomWhite ? black : white, bottomWhite ? white : black);
}

function fillCaptured(id, own, other) {
  const el = $(id);
  el.innerHTML = '';
  for (const ch of own.out) {
    const img = document.createElement('img');
    img.src = pieceImg(ch);
    el.appendChild(img);
  }
  const diff = own.value - other.value;
  if (diff > 0) {
    const span = document.createElement('span');
    span.textContent = `+${diff}`;
    el.appendChild(span);
  }
}

/* ---------------- move list / eval / stats ---------------- */

function renderMoves() {
  const ol = $('movelist');
  ol.innerHTML = '';
  const moves = S.san_moves;
  const viewed = shownPly();
  for (let i = 0; i < moves.length; i += 2) {
    const li = document.createElement('li');
    li.innerHTML = `<span class="num">${i / 2 + 1}.</span>`;
    for (const k of [i, i + 1]) {
      const span = document.createElement('span');
      span.className = 'm';
      if (moves[k] !== undefined) {
        span.textContent = moves[k];
        const ply = k + 1;
        if (ply === moves.length && isLive()) span.classList.add('latest');
        if (ply === viewed && !isLive()) span.classList.add('viewing');
        span.addEventListener('click', () => gotoPly(ply));
      }
      li.appendChild(span);
    }
    ol.appendChild(li);
  }
  const target = ol.querySelector('.viewing') || ol.querySelector('.latest');
  if (target) target.scrollIntoView({ block: 'nearest' });
}

function gotoPly(ply) {
  ply = Math.max(0, Math.min(ply, totalPlies()));
  viewPly = ply === totalPlies() ? null : ply;
  selected = null;
  render();
}

function renderEval() {
  const p = S.eval_white;
  $('eval-white').style.height = `${(p * 100).toFixed(1)}%`;
  const shown = humanIsWhite() ? p : 1 - p;
  $('eval-label').textContent = `${Math.round(shown * 100)}%`;
}

function renderStats() {
  const st = S.last_search;
  $('search-stats').hidden = !st;
  if (!st) return;
  $('stat-rollouts').textContent = st.rollouts.toLocaleString();
  $('stat-time').textContent = st.elapsed;
  $('stat-nps').textContent = st.nps;
}

/* ---------------- game over ---------------- */

let overShown = false;

function showGameOver() {
  if (overShown) return;
  overShown = true;
  playSound('end');
  const o = S.over;
  const humanWhite = humanIsWhite();
  let title, icon;
  if (o.reason === 'abandoned') { title = 'Game closed'; icon = '🕐'; }
  else if (o.result === '1/2-1/2') { title = 'Draw'; icon = '½'; }
  else if ((o.result === '1-0') === humanWhite) { title = 'You win! 🎉'; icon = '🏆'; }
  else { title = 'Model wins'; icon = '🤖'; }
  $('over-icon').textContent = icon;
  $('over-title').textContent = title;
  $('over-detail').textContent = o.reason === 'abandoned'
    ? 'This game was closed after a period of inactivity.'
    : `${o.result} · by ${o.reason}`;
  const n = Math.ceil(S.san_moves.length / 2);
  $('over-stats').textContent = `${n} move${n === 1 ? '' : 's'} played`;
  $('over-overlay').hidden = false;
}

/* ---------------- move animation ---------------- */

function animateAppliedMove(st) {
  // Slide the moved piece from its old square to the new one.
  const uci = st.last_move;
  if (!uci || !isLive()) { applyState(st); return; }
  const boardEl = $('board');
  const slides = [[uci.slice(0, 2), uci.slice(2, 4)]];
  const san = st.san_moves[st.san_moves.length - 1] || '';
  if (san.startsWith('O-O')) {
    const rank = uci[1]; // castling rank
    slides.push(san.startsWith('O-O-O') ? ['a' + rank, 'd' + rank] : ['h' + rank, 'f' + rank]);
  }
  const beforeRects = slides.map(([from, to]) => {
    const fromEl = boardEl.querySelector(`[data-sq="${from}"]`);
    const toEl = boardEl.querySelector(`[data-sq="${to}"]`);
    return fromEl && toEl ? [fromEl.getBoundingClientRect(), toEl.getBoundingClientRect()] : null;
  });

  applyState(st);

  slides.forEach(([from, to], i) => {
    if (!beforeRects[i]) return;
    const [r1, r2] = beforeRects[i];
    const destImg = $('board').querySelector(`[data-sq="${to}"] img`);
    if (!destImg) return;
    destImg.style.visibility = 'hidden';
    const ghost = destImg.cloneNode();
    ghost.style.cssText =
      `position:fixed;left:${r1.left}px;top:${r1.top}px;width:${r2.width}px;height:${r2.height}px;` +
      'visibility:visible;pointer-events:none;z-index:90;transition:transform 0.14s ease;';
    document.body.appendChild(ghost);
    requestAnimationFrame(() => {
      ghost.style.transform = `translate(${r2.left - r1.left}px, ${r2.top - r1.top}px)`;
    });
    setTimeout(() => { ghost.remove(); destImg.style.visibility = ''; }, 170);
  });
}

/* ---------------- interaction ---------------- */

function clearAnnotations() {
  if (arrows.length || marks.size) { arrows = []; marks.clear(); return true; }
  return false;
}

function onSquareDown(e) {
  const sqName = e.currentTarget.dataset.sq;
  if (e.button === 2) { rightDragFrom = sqName; return; }
  if (e.button !== 0) return;
  const hadAnnotations = clearAnnotations();
  if (!isLive()) { gotoPly(totalPlies()); return; }
  if (!S || S.over) { if (hadAnnotations) render(); return; }

  const pieces = fenBoard(S.fen);
  const piece = pieces[sqName];
  const mine = piece && ((piece === piece.toUpperCase()) === humanIsWhite());

  if (myTurn()) {
    if (selected && legalTargetsFrom(selected).has(sqName)) {
      attemptMove(selected, sqName);
      return;
    }
  } else {
    // model's turn: allow queueing a premove
    if (premove && !mine) { premove = null; selected = null; render(); return; }
    if (selected && selected !== sqName && !mine) {
      setPremove(selected, sqName);
      return;
    }
  }

  if (mine) {
    selected = selected === sqName ? null : sqName;
    render();
    if (selected) startDrag(e, sqName, piece);
  } else {
    selected = null;
    render();
  }
}

function onSquareUp(e) {
  if (e.button !== 2 || !rightDragFrom) return;
  const from = rightDragFrom;
  rightDragFrom = null;
  const el = document.elementFromPoint(e.clientX, e.clientY);
  const sq = el && el.closest ? el.closest('.sq') : null;
  const to = sq ? sq.dataset.sq : null;
  if (!to || to === from) {
    if (marks.has(from)) marks.delete(from); else marks.add(from);
  } else {
    const idx = arrows.findIndex((a) => a.from === from && a.to === to);
    if (idx >= 0) arrows.splice(idx, 1); else arrows.push({ from, to });
  }
  render();
}

function setPremove(from, to) {
  premove = { from, to };
  selected = null;
  render();
}

function startDrag(e, from, piece) {
  const img = document.createElement('img');
  img.src = pieceImg(piece);
  const sqSize = e.currentTarget.getBoundingClientRect().width;
  img.style.cssText =
    `position:fixed;width:${sqSize}px;height:${sqSize}px;pointer-events:none;z-index:100;opacity:0.9;`;
  document.body.appendChild(img);
  dragGhost = img;
  const half = sqSize / 2;
  const place = (ev) => {
    img.style.left = `${ev.clientX - half}px`;
    img.style.top = `${ev.clientY - half}px`;
  };
  place(e);

  const move = (ev) => place(ev);
  const up = (ev) => {
    document.removeEventListener('pointermove', move);
    document.removeEventListener('pointerup', up);
    if (dragGhost) { dragGhost.remove(); dragGhost = null; }
    const el = document.elementFromPoint(ev.clientX, ev.clientY);
    const sq = el && el.closest ? el.closest('.sq') : null;
    if (!sq || sq.dataset.sq === from) return;
    const to = sq.dataset.sq;
    if (myTurn()) {
      if (legalTargetsFrom(from).has(to)) attemptMove(from, to);
    } else if (!S.over && isLive()) {
      const pieces = fenBoard(S.fen);
      const target = pieces[to];
      const targetMine = target && ((target === target.toUpperCase()) === humanIsWhite());
      if (!targetMine) setPremove(from, to);
    }
  };
  document.addEventListener('pointermove', move);
  document.addEventListener('pointerup', up);
}

function attemptMove(from, to) {
  const plain = from + to;
  const isPromotion = S.legal_moves.includes(plain + 'q');
  if (isPromotion && !S.legal_moves.includes(plain)) {
    showPromotionPicker(from, to);
    return;
  }
  sendMove(plain);
}

function showPromotionPicker(from, to) {
  const grid = $('promo-grid');
  grid.innerHTML = '';
  const color = humanIsWhite() ? 'w' : 'b';
  for (const [suffix, letter] of [['q', 'Q'], ['r', 'R'], ['n', 'N'], ['b', 'B']]) {
    const btn = document.createElement('button');
    btn.innerHTML = `<img src="/pieces/${color}${letter}.png" alt="${letter}">`;
    btn.onclick = () => {
      $('promo-overlay').hidden = true;
      sendMove(from + to + suffix);
    };
    grid.appendChild(btn);
  }
  $('promo-overlay').hidden = false;
}

async function sendMove(uci) {
  selected = null;
  try {
    const res = await fetch('/api/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_id: gameId, uci }),
    });
    if (!res.ok) { premove = null; render(); return; }
    const st = await res.json();
    applyState(st);
    playMoveSound(st.san_moves[st.san_moves.length - 1]);
    maybeRequestEngineMove();
  } catch (err) {
    console.error(err);
  }
}

async function maybeRequestEngineMove() {
  if (!S || S.over || myTurn() || enginePending) return;
  const seq = gameSeq;
  enginePending = true;
  render();
  try {
    const res = await fetch('/api/engine-move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_id: gameId }),
    });
    if (seq !== gameSeq) return; // a new game started while we waited
    if (res.ok) {
      const st = await res.json();
      if (st.already_thinking) {
        await pollForEngineMove(seq);
        return;
      }
      enginePending = false;
      finishEngineMove(st);
    } else {
      enginePending = false;
      render();
    }
  } catch (err) {
    if (seq !== gameSeq) return;
    enginePending = false;
    console.error(err);
    render();
  }
}

async function pollForEngineMove(seq) {
  const startPly = S.san_moves.length;
  while (seq === gameSeq) {
    await new Promise((r) => setTimeout(r, 1500));
    if (seq !== gameSeq) return;
    try {
      const res = await fetch(`/api/state?game_id=${gameId}`);
      if (!res.ok) continue;
      const st = await res.json();
      if (seq !== gameSeq) return;
      if (st.san_moves.length > startPly || st.over) {
        enginePending = false;
        finishEngineMove(st);
        return;
      }
    } catch (e) { /* keep polling */ }
  }
}

function finishEngineMove(st) {
  animateAppliedMove(st);
  playMoveSound(st.san_moves[st.san_moves.length - 1]);
  firePremove();
}

function firePremove() {
  if (!premove || !S || S.over || !myTurn() || !isLive()) { premove = null; return; }
  const plain = premove.from + premove.to;
  premove = null;
  const candidate = S.legal_moves.includes(plain) ? plain
    : (S.legal_moves.includes(plain + 'q') ? plain + 'q' : null); // auto-queen premoves
  if (candidate) {
    setTimeout(() => sendMove(candidate), 60);
  } else {
    render();
  }
}

function applyState(st) {
  const prevPlies = S ? S.san_moves.length : -1;
  S = st;
  clockSyncAt = performance.now();
  if (st.san_moves.length !== prevPlies) { arrows = []; marks.clear(); }
  render();
}

/* ---------------- tools: FEN / PGN / PNG ---------------- */

function flashTool(id, text) {
  const btn = $(id);
  const old = btn.textContent;
  btn.textContent = text;
  setTimeout(() => { btn.textContent = old; }, 1200);
}

$('btn-fen').addEventListener('click', async () => {
  if (!S) return;
  try {
    await navigator.clipboard.writeText(shownFen());
    flashTool('btn-fen', '✓');
  } catch (e) {
    flashTool('btn-fen', '✗');
  }
});

async function downloadPgn() {
  if (!S) return;
  const res = await fetch(`/api/pgn?game_id=${gameId}`);
  if (!res.ok) return;
  const blob = await res.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'hybrid-rl-game.pgn';
  a.click();
  URL.revokeObjectURL(a.href);
}

$('btn-pgn').addEventListener('click', downloadPgn);

const PIECE_CACHE = {};
function cachePieceImages() {
  for (const c of 'PNBRQK') {
    for (const side of ['w', 'b']) {
      const img = new Image();
      img.src = `/pieces/${side}${c}.png`;
      PIECE_CACHE[side + c] = img;
    }
  }
}

$('btn-png').addEventListener('click', () => {
  if (!S) return;
  const size = 88, canvas = document.createElement('canvas');
  canvas.width = canvas.height = size * 8;
  const ctx = canvas.getContext('2d');
  const theme = THEMES[settings.theme] || THEMES.classic;
  const pieces = fenBoard(shownFen());
  const flip = orientation === 'black';
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const file = flip ? 7 - col : col;
      const rank = flip ? row + 1 : 8 - row;
      ctx.fillStyle = (file + rank) % 2 === 0 ? theme.light : theme.dark;
      ctx.fillRect(col * size, row * size, size, size);
      const piece = pieces[FILES[file] + rank];
      if (piece) {
        const key = (piece === piece.toUpperCase() ? 'w' : 'b') + piece.toUpperCase();
        const img = PIECE_CACHE[key];
        if (img && img.complete) ctx.drawImage(img, col * size, row * size, size, size);
      }
    }
  }
  canvas.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'position.png';
    a.click();
    URL.revokeObjectURL(a.href);
  });
  flashTool('btn-png', '✓');
});

/* ---------------- settings buttons ---------------- */

document.querySelectorAll('.swatch').forEach((b) => {
  b.addEventListener('click', () => {
    settings.theme = b.dataset.theme;
    saveSettings();
    applySettingsUI();
    if (S) render();
  });
});

$('btn-sound').addEventListener('click', () => {
  settings.sound = !settings.sound;
  saveSettings();
  applySettingsUI();
});

$('btn-coords').addEventListener('click', () => {
  settings.coords = !settings.coords;
  saveSettings();
  applySettingsUI();
  if (S) render();
});

$('btn-flip').addEventListener('click', () => {
  orientation = orientation === 'white' ? 'black' : 'white';
  selected = null;
  if (S) render();
});

/* ---------------- setup dialog ---------------- */

let chosenColor = 'white';
let chosenStrength = 0;

const timeCards = document.querySelectorAll('.tc:not(.strength)');

for (const btn of timeCards) {
  btn.addEventListener('click', () => {
    timeCards.forEach((b) => b.classList.remove('selected'));
    btn.classList.add('selected');
    const v = btn.dataset.min;
    // Presets fill the fields so the shown minutes are always what you get.
    if (v !== 'custom') { $('custom-min').value = v; $('custom-inc').value = 0; }
  });
}

// Typing your own time selects the custom card automatically.
for (const id of ['custom-min', 'custom-inc']) {
  $(id).addEventListener('input', () => {
    timeCards.forEach((b) => b.classList.remove('selected'));
    document.querySelector('.tc[data-min="custom"]').classList.add('selected');
  });
}

const STRENGTH_HINTS = {
  0: 'Clock-paced: the model shares your time control and can lose on time.',
  60: 'Full power: no clocks for anyone. The model thinks up to 60s on every move — its true strength.',
};

for (const btn of document.querySelectorAll('.strength')) {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.strength').forEach((b) => b.classList.remove('selected'));
    btn.classList.add('selected');
    chosenStrength = parseFloat(btn.dataset.sec);
    $('strength-hint').textContent = STRENGTH_HINTS[chosenStrength];
    $('tc-block').classList.toggle('dimmed', chosenStrength > 0);
  });
}

for (const btn of document.querySelectorAll('.colorbtn')) {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.colorbtn').forEach((b) => b.classList.remove('selected'));
    btn.classList.add('selected');
    chosenColor = btn.dataset.color;
  });
}

$('btn-start').addEventListener('click', () => {
  const req = {
    minutes: parseFloat($('custom-min').value) || 10,
    increment: parseFloat($('custom-inc').value) || 0,
    color: chosenColor,
    engine_seconds: chosenStrength,
  };
  startNewGame(req);
});

$('btn-rematch').addEventListener('click', () => {
  $('over-overlay').hidden = true;
  if (lastGameSettings) {
    const req = { ...lastGameSettings };
    // Rematch swaps colors, chess.com style.
    if (req.color === 'white') req.color = 'black';
    else if (req.color === 'black') req.color = 'white';
    startNewGame(req);
  } else {
    $('setup-overlay').style.display = 'flex';
  }
});

$('btn-over-new').addEventListener('click', () => {
  $('over-overlay').hidden = true;
  $('setup-overlay').style.display = 'flex';
});

$('btn-over-pgn').addEventListener('click', downloadPgn);

$('btn-over-close').addEventListener('click', () => {
  $('over-overlay').hidden = true;
});

$('btn-new').addEventListener('click', () => {
  $('over-overlay').hidden = true;
  $('setup-overlay').style.display = 'flex';
});

$('btn-resign').addEventListener('click', async () => {
  if (!S || S.over) return;
  const res = await fetch('/api/resign', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ game_id: gameId }),
  });
  if (res.ok) applyState(await res.json());
});

let busyWaitToken = 0;

async function waitForFreeBoard(req) {
  // The model plays one game at a time; poll until the board frees up,
  // then start automatically.
  const token = ++busyWaitToken;
  $('setup-overlay').style.display = 'none';
  $('busy-overlay').hidden = false;
  while (token === busyWaitToken) {
    await new Promise((r) => setTimeout(r, 6000));
    if (token !== busyWaitToken) return;
    try {
      const res = await fetch('/api/availability');
      if (res.ok && !(await res.json()).busy) {
        $('busy-overlay').hidden = true;
        startNewGame(req);
        return;
      }
    } catch (e) { /* keep waiting */ }
  }
}

$('btn-busy-cancel').addEventListener('click', () => {
  busyWaitToken++;
  $('busy-overlay').hidden = true;
  $('setup-overlay').style.display = 'flex';
});

async function startNewGame(req) {
  req.replace_game_id = gameId || '';
  const res = await fetch('/api/new', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (res.status === 503) {
    waitForFreeBoard(req);
    return;
  }
  if (!res.ok) {
    let msg = 'Could not start a game — please try again.';
    try { msg = (await res.json()).detail || msg; } catch (e) { /* keep default */ }
    $('setup-overlay').style.display = 'flex';
    $('strength-hint').textContent = `⚠ ${msg}`;
    return;
  }
  lastGameSettings = req;
  localStorage.setItem('hybridrl-last-game', JSON.stringify(req));
  gameSeq += 1;
  enginePending = false;
  overShown = false;
  selected = null;
  viewPly = null;
  premove = null;
  lowTimeWarned = false;
  arrows = [];
  marks.clear();
  $('setup-overlay').style.display = 'none';
  $('over-overlay').hidden = true;
  const st = await res.json();
  gameId = st.game_id;
  localStorage.setItem('hybridrl-game-id', gameId);
  orientation = st.human_color;
  applyState(st);
  playSound('start');
  maybeRequestEngineMove(); // in case the model plays white
}

/* ---------------- keyboard: step through the game ---------------- */

document.addEventListener('keydown', (e) => {
  if (!S) return;
  if (e.target.tagName === 'INPUT') return;
  if ($('setup-overlay').style.display !== 'none') return;
  if (e.key === 'ArrowLeft') { gotoPly(shownPly() - 1); e.preventDefault(); }
  else if (e.key === 'ArrowRight') { gotoPly(shownPly() + 1); e.preventDefault(); }
  else if (e.key === 'ArrowUp') { gotoPly(0); e.preventDefault(); }
  else if (e.key === 'ArrowDown') { gotoPly(totalPlies()); e.preventDefault(); }
});

$('board').addEventListener('contextmenu', (e) => e.preventDefault());

/* ---------------- boot ---------------- */

setInterval(() => tickClocks(), 100);

// Heartbeat: keeps our game marked active on the server (one game at a
// time online — silent players would otherwise be treated as gone).
setInterval(async () => {
  if (!gameId || !S || S.over) return;
  try {
    const res = await fetch(`/api/state?game_id=${gameId}`);
    if (res.ok) {
      const st = await res.json();
      // Only re-render if something actually changed server-side.
      if (st.san_moves.length !== S.san_moves.length || !!st.over !== !!S.over) {
        applyState(st);
      }
    }
  } catch (e) { /* transient */ }
}, 60000);

loadSettings();
applySettingsUI();
cachePieceImages();

try {
  lastGameSettings = JSON.parse(localStorage.getItem('hybridrl-last-game')) || null;
} catch (e) { /* none saved */ }

(async function boot() {
  const stored = localStorage.getItem('hybridrl-game-id');
  if (!stored) return; // no previous game -> setup dialog stays
  try {
    const res = await fetch(`/api/state?game_id=${stored}`);
    if (res.ok) {
      gameId = stored;
      $('setup-overlay').style.display = 'none';
      const st = await res.json();
      orientation = st.human_color;
      applyState(st);
      maybeRequestEngineMove();
    }
    // 404 = expired on the server -> setup dialog stays
  } catch (e) { /* server unreachable -> setup dialog stays */ }
})();
