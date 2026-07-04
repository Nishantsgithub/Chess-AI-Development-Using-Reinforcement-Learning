# DQN / DDQN endgame experiments

These scripts belong to the first phase of the dissertation: comparing DQN
and DDQN on a 4×4 chessboard endgame (White king + queen vs Black king),
implemented with the [Tianshou](https://github.com/thu-ml/tianshou) RL library.

Key finding: DDQN converged to the optimal checkmating strategy in ~30,000
training steps (γ = 0.95, β = 10), outperforming vanilla DQN, and a myopic
discount factor (γ = 0.05) stalled at a mean reward of 0.8 — future rewards
matter in chess. Full details in chapter 3–4 of the [dissertation](../../Dissertation.pdf).

> **Note:** `dqn_tianshou_DQN.py` additionally imports `degree_freedom_queen.py`,
> `generate_game.py`, and `Chess_env_gym.py` from the University of Sheffield
> course environment, which are not redistributed in this repository. The two
> `degree_freedom_king*.py` files here compute the legal-move degrees of freedom
> for the kings in that environment.
