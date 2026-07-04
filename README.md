# Chess AI Development Using Reinforcement Learning

A hybrid chess engine that combines **supervised learning on expert games** with **AlphaZero-style MCTS self-play refinement** — built for my MSc Data Analytics dissertation at the University of Sheffield ([full dissertation PDF](Dissertation.pdf)), and now playable in your browser through a modern web app.

![Playing against the model in the web app](docs/app-screenshot.png)

## Highlights

- **AlphaZero-style network**: 20 residual blocks × 256 filters, value + policy heads, driven by a parallel Monte Carlo Tree Search with virtual loss.
- **Hybrid training**: pre-training on the CCRL expert dataset (~2.5M engine games) followed by epsilon-greedy MCTS self-play fine-tuning — reaching strong play at a fraction of full AlphaZero's compute cost.
- **Results**: the hybrid model beat Komodo 23/24 (Elo 2700/2900) and Stockfish levels 6–7 (2500/2700), and drew against Komodo 25 (3200) and Stockfish level 8 (3000).
- **Web app** (`app/`): play the model with real chess clocks — it paces its thinking to the time control like a human. Premove, move animation, themes, arrows, eval bar, PGN/FEN export, and one-command launch.

## Play against the model

```bash
git clone https://github.com/Nishantsgithub/Chess-AI-Development-Using-Reinforcement-Learning.git
cd Chess-AI-Development-Using-Reinforcement-Learning
pip install -r requirements.txt
python app/server.py
```

Your browser opens automatically. Pick a time control (5/10/30 min or custom) — the model budgets its thinking from its own clock, so longer games mean deeper, stronger search. Or pick **full power** mode: no clocks, maximum think time per move.

The engine automatically uses your NVIDIA GPU if a CUDA build of PyTorch is installed (roughly 3× the search speed); otherwise it runs on CPU.

**App features**: per-move MCTS statistics (rollouts / nodes per second / search-tree reuse), premoves while the model thinks, click or drag moves with legal-move hints, move-list review (← → keys), right-drag analysis arrows, captured-material display, four board themes, sounds, evaluation bar from the network's value head, PGN download, FEN copy, and board-snapshot PNG export.

### Deployment

The app is multi-session and ships with a [Dockerfile](Dockerfile) (listens on port 7860, Hugging Face Spaces convention). Set `DEPLOYED=1` to enable hosting mode: CPU-tuned think times, rate limiting, and — by design — **one game at a time** (`MAX_ACTIVE_GAMES`), so whoever is playing gets the engine's full strength while other visitors see a waiting room that starts their game automatically.

## Dissertation results

The research compared reinforcement-learning approaches for chess in four stages:

**1. Endgame (4×4 board, K+Q vs K) — DQN vs DDQN.** DDQN converged to optimal checkmating play in ~30,000 steps (γ = 0.95); vanilla DQN was slower and less stable, matching theory. A myopic discount factor (γ = 0.05) plateaued — long-horizon rewards are essential in chess.

**2. Full game — DDQN vs MCTS.** Over 100 games against a random opponent: MCTS scored 28 wins / 9 losses vs DDQN's 2 / 2 — MCTS handles chess's state space far better at practical compute budgets.

**3. AlphaZero pipeline.** Iterative self-play training worked (iteration 2 beat iteration 1, winning 8 of 12 decisive games) but was computationally prohibitive at dissertation scale — motivating the hybrid.

**4. Hybrid approach** — supervised pre-training + MCTS self-play refinement:

| Opponent (Elo) | As White | As Black |
|---|---|---|
| Komodo 25 (3200) | Draw | Draw |
| Komodo 24 (2900) | **Win** | Draw |
| Komodo 23 (2700) | **Win** | **Win** |
| Stockfish level 8 (3000) | Draw | Draw |
| Stockfish level 7 (2700) | **Win** | Draw |
| Stockfish level 6 (2500) | **Win** | **Win** |

A hybrid model bootstrapped from *human* games (~2500 Elo data) and refined by self-play went 11–9 against the CCRL-trained baseline (~2900–3000 Elo data) — evidence that the self-play mechanism itself adds real strength.

## Model weights

All weights are the same 20×256 architecture and are interchangeable in the app via `--model`:

| File | Description |
|---|---|
| `weights/HPC_20x256.pt` | **The hybrid model trained on Sheffield's HPC — the strongest; app default** |
| `weights/AlphaZeroNet_20x256.pt` | CCRL supervised baseline |
| `weights/selfplay_iteration1_20x256_epoch29.pt` | Hybrid self-play iteration 1 |
| `weights/AlphaZero_Iteration2_20x256.pt` | Self-play iteration 2 ("Model 2") |
| `weights/human_20x256.pt` | Supervised on ~2500-Elo human games |
| `weights/FineTuned_human_20x256_epoch2.pt` | Human-data model + self-play refinement |

## Repository structure

```
AlphaZeroNetwork.py    # the network: 20 residual blocks, value + policy heads
MCTS.py                # parallel MCTS with UCT selection and virtual loss
encoder.py             # board -> 16-plane tensor; move <-> policy-index mapping
playchess.py           # original CLI: play or self-play against the model
app/                   # web app: FastAPI server, clock-aware engine wrapper, UI
training/              # hybrid training loop: self-play generation + fine-tuning
experiments/dqn/       # 4x4 endgame DQN/DDQN experiments (Tianshou)
GUI/                   # legacy pygame desktop GUI
images/                # piece images (used by both UIs)
weights/               # trained models (~97 MB each)
Dissertation.pdf       # the full write-up
```

## The training pipeline (Hybrid approach)

1. **Supervised pre-training** on CCRL expert games gives the network a strong prior (see `training/CCRLDataset.py`).
2. **Self-play generation** — `training/selfplay.py` plays the model against itself with epsilon-greedy exploration (`training/playchess_selfplay.py`), decaying exploration as the game progresses; `training/fix_pgn_results.py` fills in results.
3. **Fine-tuning** — `training/train_self.py` trains on the self-play games (Adam, MSE value loss + cross-entropy policy loss, LR decay).
4. Evaluate the new model against the previous best; promote it if it wins ≥55% of decisive games, then repeat.

## Acknowledgements

- The core network/MCTS/encoder implementation builds on [jackdawkins11/pytorch-alpha-zero](https://github.com/jackdawkins11/pytorch-alpha-zero), substantially extended for this research (hybrid training, epsilon-greedy self-play, repetition avoidance, the web app, GPU support, and search-tree reuse).
- [python-chess](https://github.com/niklasf/python-chess) for rules and PGN handling; [Tianshou](https://github.com/thu-ml/tianshou) for the DQN experiments.
- Supervised by Prof. Eleni Vasilaki, Department of Computer Science, University of Sheffield.

## License

[MIT](LICENSE)
