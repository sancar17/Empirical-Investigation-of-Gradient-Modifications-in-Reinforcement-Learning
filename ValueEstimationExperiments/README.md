# ValueEstimationExperiments

This folder contains the experiments from **Section 4: Value Estimation in Discrete Environments** and **Section 5: Control Experiments with Trajectory-Based Training** of the thesis _Empirical Investigation of Gradient Modifications in Reinforcement Learning_.

---

## Summary

While differentiable physics environments enable gradient-based optimization, many reinforcement learning methods still rely on classical value estimation updates such as **Temporal Difference (TD) learning**. This part of the thesis systematically compares TD with **Gradient Descent (GD)** and **GDS-TDM (Gradient Descent Sign - Temporal Difference Magnitude)** updates in both standard online settings and **trajectory-based offline training**.

Experiments are performed in the **Lunar Lander** environment using Q-learning, SARSA, and DQN. We also include **action repetition** setups and **hybrid training** schemes (starting with TD and switching to GDS-TDM) to explore how update strategies affect learning stability and control performance.

---

## Folder Structure

- `cfg/` — contains `.json` config files used to control learning rate, update strategy, feature extractor type, number of episodes, WandB logging, and more.
- `main.py` — trains Q-learning or DQN agents using TD, GD, or GDS-TDM.
- `main_sarsa.py` — trains SARSA agents with TD, GD, or GDS-TDM.
- `main_hybrid_q_learning.py`, `main_hybrid_dqn.py`, `main_hybrid_sarsa.py` — implement hybrid training (TD to GDS-TDM switch after `switch_episode`).
- `main_repeat.py`, `main_repeat_hybrid.py`, `main_repeat_hybrid_dqn.py`, `main_sarsa_repeat.py` — repeat action experiments with and without hybrid switch.
- `main_from_trajectory_q_learning.py`, `main_from_trajectory_dqn.py` — control experiments using fixed trajectories.
  
---

## Running Standard Value Estimation Experiments

For Q-learning or DQN:
```bash
python main.py --config configs/Q_Learning/TD/config.json
```

For SARSA:
```bash
python main_sarsa.py --config configs/SARSA/GD/config.json
```

Use `"method"` in the config to select the update rule:
- `"td"`: Temporal Difference
- `"gd"`: Gradient Descent on TD error
- `"gds_tdm"`: GDS-TDM

### Hybrid Training

Hybrid training starts with TD and switches to GDS-TDM after `switch_episode`. Example:

```bash
python main_hybrid_q_learning.py --config configs/Q_Learning/hybrid/config.json
```

---

## Action Repetition Experiments

Set `num_repeat` in the config to apply action repetition. For example:

```bash
python main_repeat.py --config configs/Q_Learning/TD_repeat3/config.json
python main_sarsa_repeat.py --config configs/SARSA/GD_repeat3/config.json
```

For hybrid setups that switch both update method and action repetition (e.g., `n=3` to `n=1`):

```bash
python main_repeat_hybrid.py --config configs/Q_Learning/TD_hybrid_repeat/config.json
```

In hybrid repeat configs:
- `cfg.num_repeat` applies for the first `cfg.switch_episode` episodes
- Afterward, the repetition count switches to `n=1`

---

## Trajectory-Based Training

These experiments isolate optimization behavior by using fixed offline trajectories instead of fresh environment rollouts.

Run with:
```bash
python main_from_trajectory_q_learning.py --config configs/Q_Learning/TrajectoryBased/TD/config.json
python main_from_trajectory_dqn.py --config configs/DQN/TrajectoryBased/GDS-TDM/config.json
```

Key options in the config:
- `pretrained_model`: path to the TD-trained base agent used for trajectory collection
- `num_trajectories`: number of fixed trajectories used (e.g., 10)
- `trajectory_shift_threshold`: if > 0, switch trajectory once TD error drops below this value
- `max_episodes_per_trajectory`: number of training episodes per trajectory (e.g., 50000)

---

## Configuration Notes

All experiments are controlled via `.json` config files in the `configs/` directory.

Some key fields include:

- `method`: `"td"`, `"gd"`, or `"gds_tdm"`
- `feature_extractor`: `"polynomial"` (default), `"rbf"` also available
- `learning_rate`, `discount_factor`, `epsilon`, and decay parameters
- `wandb_project` / `wandb_name`: for logging to Weights & Biases (disable via `"use_wandb": false`)
- `switch_episode`: for hybrid experiments, when to switch from TD to GDS-TDM
- `repeat_num`: for action repetition experiments
- `save_model_freq`, `validate_freq`: checkpoints and validation
- `early_stopping_patience` and `grad_clip`: for training stability

All scripts support reproducibility via the `"seed"` field.

---

## Logging & Visualization

All training runs support logging to [Weights & Biases](https://wandb.ai) if `"use_wandb": true` is set in the config. Run names and project names can be set using:

```json
"wandb_name": "Experiment_Name",
"wandb_project": "Project_Name"
```

Set `"use_wandb": false` to disable logging.

---

For details on the update rules, experimental results, and theoretical motivation, please refer to **Sections 4 and 5 of the thesis**.
