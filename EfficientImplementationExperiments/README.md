# EfficientImplementationExperiments

This folder contains the experiments for **Section 3.4: Efficient Implementation of Gradient Stopping Techniques** and **Appendix A.2: Efficient Implementation Details** from the thesis _Empirical Investigation of Gradient Modifications in Reinforcement Learning_.

The experiments demonstrate how gradient stopping techniques can be implemented efficiently using **PyTorch** and **JAX**, comparing their performance against each framework’s native automatic differentiation tools. Both versions reconstruct gradient flows using custom autograd operations instead of relying solely on built-in backward functions.

---

## Summary

In this benchmark:

- A small neural network is used to predict control signals.
- A differentiable physical simulator models the environment's transitions.
- A fixed initial and target state define the control objective.
- The system is optimized over a time horizon `t`, ranging from 1 to 150.
- Total runtime is measured per time horizon to evaluate performance.

These experiments quantify the computational cost of reconstructing custom backward passes for gradient stopping, compared to the frameworks’ built-in `.backward()` (PyTorch) and `.grad()` (JAX) methods. Results show that the native autograd systems remain more efficient for most cases.

---

## Implementations

### PyTorch (`main_pytorch.py`)
Implements and evaluates efficient gradient stopping as follows:

- `torch.detach()` is used to stop gradients at the network input.
- `torch.autograd.functional.jacobian()` is used to extract intermediate derivatives without storing the full backward graph.

### JAX (`main_jax.py`)
Provides a JAX-based implementation using:

- `jax.lax.stop_gradient()` to block gradients during forward pass.
- `jax.vjp()` (vector-Jacobian product) for memory-efficient gradient computation.

---

## Running the Experiments

Run the scripts to benchmark performance:

### PyTorch

```bash
python main_pytorch.py
```

### JAX

```bash
python main_jax.py
```