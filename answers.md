# Questionnaire Answers

## 1) Explain the rationale behind your shaped reward function. What specific agent behaviors were you trying to encourage or discourage with each term?

In this project, the shaped reward in `environment.py` is:

- Baseline upright term: `cos(theta1) + cos(theta2)`
- Center penalty: `-0.1 * abs(cart_x)`
- Angular velocity penalty: `-0.01 * (abs(omega1) + abs(omega2))`
- Action penalty: `-0.001 * (action^2)`

Rationale by term:

- `cos(theta1) + cos(theta2)`: Encourages both poles to remain upright; this is the main objective signal.
- `-0.1 * abs(cart_x)`: Discourages cart drift and helps prevent edge-failure behavior.
- `-0.01 * (abs(omega1) + abs(omega2))`: Discourages violent oscillations and encourages smoother stabilization.
- `-0.001 * (action^2)`: Discourages excessive force and promotes energy-efficient, less jittery control.

Overall target behavior: upright poles with stable, centered, smooth control instead of aggressive shaking strategies.

## 2) Describe your process for tuning the PPO agent's hyperparameters. Which parameters had the most significant impact on training stability and final performance?

Methodology used: manual iterative tuning focused on environment/reward design first, then short-to-long training runs for validation.

What was tuned in practice:

- `reward_type` (`baseline` vs `shaped`) and the shaped reward coefficients.
- Training duration (`--timesteps`) for smoke tests and longer confirmation runs.
- Logging behavior (added CSV callback and optional TensorBoard path for reliability in container execution).

What was not heavily tuned in this submission:

- I mostly kept PPO algorithm defaults from Stable-Baselines3 for parameters like `learning_rate`, `n_steps`, `batch_size`, etc., because the primary performance differences came from reward shaping and training duration under this setup.

Most significant impact observed:

- Reward shaping terms/weights had the biggest impact on stability and behavior quality.
- Timesteps affected apparent policy maturity and artifact quality (GIF/plot quality).

## 3) What were the biggest challenges in designing the `pymunk` physics environment? Discuss any issues with stability, simulation speed, or realism you encountered.

Main challenges encountered:

- Joint and constraint configuration:
  - Correctly constraining cart motion to a horizontal track while keeping both pole joints physically valid.
  - Solved with a `GrooveJoint` for the cart track and `PivotJoint`s for cart-pole and pole-pole links.

- Stable simulation stepping:
  - Avoiding unstable behavior required fixed-step integration.
  - Solved by using a constant timestep (`dt = 1/60`).

- State extraction consistency:
  - Converting `pymunk` body states into a stable 6D observation for RL.

- API compatibility between Gymnasium and SB3 usage patterns:
  - Implemented compatibility handling (`legacy_api`) so both old-style and new-style step/reset signatures can work depending on script context.

- Runtime portability:
  - Headless/container rendering produced runtime warnings (e.g., `XDG_RUNTIME_DIR`) but evaluation and GIF generation were still successful.

## 4) Beyond reward shaping, what other techniques could improve the agent's learning?

Practical improvements beyond reward shaping include:

- Alternative algorithms for continuous control:
  - SAC or TD3 could improve sample efficiency and robustness on this nonlinear system.

- Curriculum learning:
  - Start with easier dynamics (smaller initial angle noise / less strict termination), then progressively increase difficulty.

- Observation improvements:
  - Normalize observations more aggressively and add engineered features (e.g., relative pole angle, energy-like terms).

- Domain randomization:
  - Randomize masses, friction, and minor disturbances during training to improve generalization.

- Policy/network changes:
  - Tune policy network size/activation and PPO hyperparameters once environment dynamics are stable.

## 5) How robust is your final trained agent? How would you quantitatively measure its resilience to disturbances (e.g., random forces applied during evaluation)?

Current robustness level:

- The trained policy is functional for the submission tasks and artifacts, but I would classify robustness as moderate rather than fully disturbance-hardened.

Quantitative robustness experiment design:

1. Disturbance protocol:
- During evaluation, apply random horizontal impulses to the cart at random intervals.
- Test multiple disturbance levels (low/medium/high magnitudes).
- Run many episodes per level (for example, 50-100 episodes with different seeds).

2. Metrics to record:
- Mean episode length before failure.
- Survival rate at fixed horizons (e.g., 200/500/1000 steps).
- Mean and worst-case episode reward.
- Recovery time after disturbance (steps to return near upright thresholds).
- Control smoothness under disturbance (action variance).

3. Reporting:
- Plot disturbance magnitude vs survival rate and reward.
- Report confidence intervals across seeds.
- Compare baseline-trained vs shaped-trained agents under identical disturbance settings.

This setup gives an objective resilience score rather than only visual inspection.
