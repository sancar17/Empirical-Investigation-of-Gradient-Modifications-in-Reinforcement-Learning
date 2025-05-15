# Empirical Investigation of Gradient Modifications in Reinforcement Learning

This repository contains the experimental codebase for my Master's thesis:  
**_Empirical Investigation of Gradient Modifications in Reinforcement Learning_**  
(Technical University of Munich, 2025)

The project explores the effects of gradient-based modifications in reinforcement learning settings, focusing on differentiable physics environments, discrete control tasks, and efficient algorithmic implementations.

---

## Repository Structure

### `GradientStoppingExperiments/`

This folder includes training and testing scripts used in **Section 3: Gradient Stopping in Differentiable Physics Environments**. It contains implementations of the BPTT and SHAC algorithms with:

- **Original gradient flow**
- **Modified gradient flow**
- **Combined gradient flow**
- **Directional combined gradient flow**

### `EfficientImplementationExperiments/`

Contains scripts for the **efficient forward and backward pass implementations** for gradient stopping modifications. These scripts accompany **Section 3.4: Efficient Implementation of Gradient Stopping Techniques**.

### `ValueEstimationExperiments/`

Includes training and testing scripts, configuration files, evaluation metrics, and resulting video recordings for:

- **Section 4: Value Estimation in Discrete Environments**
- **Section 5: Control Experiments with Trajectory-Based Training**

Agents are trained using TD, GD, and GDS-TDM update methods across Q-learning, SARSA, and DQN setups.

---

## Setup Instructions

### Conda Environment

To replicate the experimental setup, create the Conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate gradient-rl
```

### Running Experiments

Each folder contains a dedicated `README.md` with instructions for running the experiments.
