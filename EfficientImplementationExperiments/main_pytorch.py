import torch
import torch.nn as nn
import time
from typing import Callable
import torch.autograd as autograd

class OriginalImplementation:
    def __init__(self, theta: torch.Tensor, x_initial: torch.Tensor, x_target: torch.Tensor, 
                 time_steps: int, f: Callable, g: Callable):
        self.theta = theta
        self.x_initial = x_initial
        self.x_target = x_target
        self.time_steps = time_steps
        self.f = f
        self.g = g
        
    def forward(self, detach: bool = False) -> tuple:
        x = self.x_initial.clone()
        x_values = [x.clone()]
        
        for t in range(self.time_steps):
            x_input = x.detach() if detach else x
            c = self.f(x_input, self.theta)
            x = self.g(x, c)
            x_values.append(x.clone())
            
        loss = 0.5 * torch.sum((x - self.x_target)**2)
        return x, loss, x_values

    def compute_gradients(self) -> tuple:
        torch.cuda.empty_cache()
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated()
        
        # Run forward pass with no detach
        x, loss, _ = self.forward(detach=False)
        loss.backward(retain_graph=True)  
        original_grad = self.theta.grad.clone() if self.theta.grad is not None else torch.zeros_like(self.theta)
        self.theta.grad = None  
        
        # Run forward pass with detach
        x, loss, _ = self.forward(detach=True)
        loss.backward()
        detached_grad = self.theta.grad.clone() if self.theta.grad is not None else torch.zeros_like(self.theta)
        self.theta.grad = None
        
        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated()
        
        return original_grad, detached_grad, end_time - start_time, initial_memory, peak_memory

class EfficientImplementation:
    def __init__(self, theta: torch.Tensor, x_initial: torch.Tensor, x_target: torch.Tensor, 
                 time_steps: int, f: Callable, g: Callable):
        self.theta = theta
        self.x_initial = x_initial
        self.x_target = x_target
        self.time_steps = time_steps
        self.f = f
        self.g = g
        
    def compute_ABCD(self, x: torch.Tensor, x_next: torch.Tensor, c: torch.Tensor) -> tuple:
        """
        Compute Jacobian matrices with proper dimensions:
        A: dx_{n}/dc_{n-1} [state_dim, control_dim]
        B: dc_{n}/dtheta [control_dim, param_dim]
        C: dx_{n}/dx_{n-1} [state_dim, state_dim]
        D: dc_{n}/dx_{n} [control_dim, state_dim]
        """
        # Compute A = dxn/dcn-1 using functional.jacobian
        def g_wrapper(c_input):
            return self.g(x, c_input)
        A = torch.autograd.functional.jacobian(g_wrapper, c)
        
        # Compute B = dcn/dtheta using functional.jacobian
        def f_wrapper_theta(theta_input):
            return self.f(x, theta_input)
        B = torch.autograd.functional.jacobian(f_wrapper_theta, self.theta)
        
        # Compute C = dxn/dxn-1 using functional.jacobian
        def g_wrapper_x(x_input):
            return self.g(x_input, c)
        C = torch.autograd.functional.jacobian(g_wrapper_x, x)
        
        # Compute D = dcn/dxn using functional.jacobian
        def f_wrapper_x(x_input):
            return self.f(x_input, self.theta)
        D = torch.autograd.functional.jacobian(f_wrapper_x, x)
        
        return A, B, C, D

    def compute_gradients(self) -> tuple:
        torch.cuda.empty_cache()
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        x = self.x_initial.clone()
        derivatives = []
        xs = [x]
        cs = []
        
        for t in range(self.time_steps):
            c = self.f(x, self.theta)
            x_next = self.g(x, c)
            A, B, C, D = self.compute_ABCD(x, x_next, c)
            derivatives.append((A, B, C, D))
            cs.append(c)
            x = x_next
            xs.append(x)
            
        # Final loss gradient
        dL_dx = xs[-1] - self.x_target  # [state_dim]
        
        # Initialize gradients
        d_original = torch.zeros_like(self.theta)
        d_detached = torch.zeros_like(self.theta)
        
        # Combined backward pass for both gradients
        accumulated_grad = dL_dx  # For original gradient
        accumulated_detached = dL_dx  # For detached gradient
        
        for t in range(self.time_steps-1, -1, -1):
            A, B, C, D = derivatives[t]
            
            # Compute original gradient contribution
            control_grad_orig = A.T @ accumulated_grad  # [control_dim]
            param_grad_orig = B.T @ control_grad_orig  # [param_dim]
            d_original += param_grad_orig
            
            # Compute detached gradient contribution
            control_grad_detached = A.T @ accumulated_detached  # [control_dim]
            param_grad_detached = B.T @ control_grad_detached  # [param_dim]
            d_detached += param_grad_detached
            
            if t > 0:
                # Propagate original gradient
                state_grad = C.T @ accumulated_grad  # [state_dim]
                control_to_state = D.T @ control_grad_orig  # [state_dim]
                accumulated_grad = state_grad + control_to_state
                
                # For detached gradient, only propagate through C
                accumulated_detached = C.T @ accumulated_detached  # [state_dim]
        
        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated()
        
        return d_original, d_detached, end_time - start_time, initial_memory, peak_memory
class NeuralController(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, control_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, control_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class PhysicsSimulator(nn.Module):
    def __init__(self, state_dim: int, control_dim: int, hidden_dim: int):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.control_encoder = nn.Linear(control_dim, hidden_dim)
        
        self.dynamics = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Physical constraints
        self.max_velocity = 10.0
        self.dt = 0.01
        
    def forward(self, state, control):
        # Encode state and control
        state_encoded = self.state_encoder(state)
        control_encoded = self.control_encoder(control)
        
        # Combine and compute dynamics
        combined = torch.cat([state_encoded, control_encoded], dim=-1)
        acceleration = self.dynamics(combined)
        
        # Apply physical constraints
        acceleration = torch.clamp(acceleration, -self.max_velocity/self.dt, self.max_velocity/self.dt)
        new_state = state + acceleration * self.dt
        
        return new_state

def create_test_environment(device):
    # Environment dimensions
    state_dim = 12  # position and velocity in 3D
    control_dim = 6  # force in 3D
    hidden_dim = 128  # middle-sized network
    
    # Create neural controller (f)
    controller = NeuralController(state_dim, hidden_dim, control_dim).to(device)
    
    # Create physics simulator (g)
    simulator = PhysicsSimulator(state_dim, control_dim, hidden_dim).to(device)
    
    # Wrapper functions to match the original interface
    def f(x: torch.Tensor, theta: nn.Parameter) -> torch.Tensor:
        param_shapes = [p.shape for p in controller.parameters()]
        param_lengths = [p.numel() for p in controller.parameters()]
        
        param_chunks = torch.split(theta, param_lengths)
        reshaped_params = [chunk.reshape(shape) for chunk, shape in zip(param_chunks, param_shapes)]
        
        for param, new_value in zip(controller.parameters(), reshaped_params):
            param.data = new_value
            
        return controller(x)
    
    def g(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return simulator(x, c)
    
    # Initial conditions
    x_initial = torch.randn(state_dim, device=device)
    x_target = torch.randn(state_dim, device=device)
    
    # Pack controller parameters into a single tensor
    theta = nn.Parameter(torch.cat([p.flatten() for p in controller.parameters()]))
    
    return f, g, theta, x_initial, x_target, controller

def compare_extended():
    print("\nRunning extended comparison...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f, g, theta, x_initial, x_target, controller = create_test_environment(device)
    time_steps = 50
    
    # Create both implementations
    original_impl = OriginalImplementation(theta, x_initial, x_target, time_steps, f, g)
    efficient_impl = EfficientImplementation(theta, x_initial, x_target, time_steps, f, g)
    
    compare_results(original_impl, efficient_impl)

def compare_results(original_impl, efficient_impl):
    # Run and compare
    orig_grad, orig_detached, orig_time, orig_init_mem, orig_peak_mem = original_impl.compute_gradients()
    eff_grad, eff_detached, eff_time, eff_init_mem, eff_peak_mem = efficient_impl.compute_gradients()

    # Print dimensions
    state_dim = original_impl.x_initial.shape[0]
    control_dim = original_impl.f(original_impl.x_initial, original_impl.theta).shape[0]
    param_dim = original_impl.theta.shape[0]
    
    print("\nProblem Dimensions:")
    print(f"Observation Space Dimension: {state_dim}")
    print(f"Action Space Dimension: {control_dim}")
    print(f"Number of Model Parameters: {param_dim}")
    
    # Gradient differences analysis
    normal_diff = (orig_grad - eff_grad).abs()
    detached_diff = (orig_detached - eff_detached).abs()
    
    print("\nGradient Differences:")
    print(f"  Normal - Max Diff: {normal_diff.max():.6f}, Mean Diff: {normal_diff.mean():.6f}")
    print(f"  Detached - Max Diff: {detached_diff.max():.6f}, Mean Diff: {detached_diff.mean():.6f}")
    
    print("\nGradients Match (within 1e-5):")
    print(f"  Normal: {torch.allclose(orig_grad, eff_grad, rtol=1e-5, atol=1e-5)}")
    print(f"  Detached: {torch.allclose(orig_detached, eff_detached, rtol=1e-5, atol=1e-5)}")
    
    print("\nPerformance Comparison:")
    print(f"Original Implementation:")
    print(f"  Time: {orig_time:.4f} seconds")
    print(f"  Memory Usage: {(orig_peak_mem - orig_init_mem) / 1024**2:.2f} MB")
    print(f"Efficient Implementation:")
    print(f"  Time: {eff_time:.4f} seconds")
    print(f"  Memory Usage: {(eff_peak_mem - eff_init_mem) / 1024**2:.2f} MB")
    
    print(f"\nEfficiency Gains:")
    print(f"  Time: {(orig_time - eff_time) / orig_time * 100:.1f}% faster")
    print(f"  Memory: {(orig_peak_mem - eff_peak_mem) / orig_peak_mem * 100:.1f}% less memory")

def main():
    
    print("\nTest: Extended Implementation with Neural Networks")
    compare_extended()

if __name__ == "__main__":
    main()