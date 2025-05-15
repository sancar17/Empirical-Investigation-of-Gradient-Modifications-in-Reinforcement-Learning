# GradientStoppingExperiments

This folder contains the experimental setup for **Section 3: Gradient Stopping in Differentiable Physics Environments** from the thesis _Empirical Investigation of Gradient Modifications in Reinforcement Learning_.

Differentiable physics simulators enable gradient-based policy optimization by exposing gradients of transition and reward functions. While methods like Backpropagation Through Time (BPTT) and Short-Horizon Actor-Critic (SHAC) leverage this differentiability for long-horizon learning, they often suffer from instability due to exploding gradients, high memory usage, and chaotic dynamics. This section explores **gradient stopping techniques**, which modify the backpropagation path by halting gradients between the simulator and policy, aiming to improve stability.

Experiments are run across five differentiable control environments (`CartPole`, `Ant`, `Hopper`, `Cheetah`, `Humanoid`) using the DiffRL framework, though code from that library is excluded due to licensing restrictions. Scripts will be shared upon request if needed for academic reproduction.

---

## Running the Experiments

All training commands should be run from within the `./examples/` directory.

### 1. Training with Original Gradients

```bash
python train_script.py --algo=bptt --env=Ant --save-dir=./logs
```

Options for `--algo`:  
- `bptt` (Backpropagation Through Time)  
- `shac` (Short-Horizon Actor-Critic)

Options for `--env`:  
- `Ant`, `CartPole`, `Hopper`, `Cheetah`, `Humanoid`, `SNUHumanoid`

### 2. Training with Gradient Stopping (Modified)

```bash
python train_script_modified.py --algo=bptt --env=Ant --save-dir=./logs
```

### 3. Training with Combined Gradient Flow

```bash
python train_script_combined.py --algo=bptt --env=Ant --save-dir=./logs
```

### 4. Training with Directional Combined Gradients

```bash
python train_script_new.py --algo=bptt --env=Ant --save-dir=./logs
```

All training scripts log metrics to TensorBoard and follow predefined YAML configuration files located in:

- `examples/cfg/bptt/`
- `examples/cfg/shac/`

---

## Testing and Visualization

To evaluate a trained policy and generate visualizations (e.g., USD files for simulation playback):

```bash
python train_script_modified.py --env=Ant --cfg=ant_bptt_cfg.yaml --checkpoint=best_policy.pt
```

Replace `train_script_modified.py` with `train_script.py`, `train_script_combined.py`, or `train_script_new.py` depending on the variant.