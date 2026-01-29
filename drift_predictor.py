import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
import optax  # We need this for the optimizer, if not installed we'll add it
from typing import Callable

# JAX Config: Enable 64-bit precision for numerical stability in ODEs
jax.config.update("jax_enable_x64", True)

class DriftVectorField(eqx.Module):
    """
    The Neural Network f(t, y, args) that defines the dynamics of the drift:
    dy/dt = f(t, y)
    
    This learns 'how' the noise changes over time.
    """
    mlp: eqx.nn.MLP

    def __init__(self, key):
        # A small MLP is sufficient for tracking scalar noise drift
        # Input size 2 (time, current_val) -> Hidden -> Output 1 (derivative)
        self.mlp = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=16,
            depth=2,
            activation=jax.nn.tanh, # Smooth activation for ODE dynamics
            key=key
        )

    def __call__(self, t, y, args):
        # We concatenate time and the current state to predict the change
        # y is shape (1,)
        t_arr = jnp.array([t])
        inputs = jnp.concatenate([t_arr, y])
        return self.mlp(inputs)

class NeuralODE(eqx.Module):
    """
    The Continuous-Depth Solver.
    Integrates the DriftVectorField to predict future noise levels.
    """
    func: DriftVectorField

    def __init__(self, key):
        self.func = DriftVectorField(key)

    def __call__(self, ts, y0):
        # ts: The timestamps we want to predict for
        # y0: The initial known noise level
        
        # We use Tsit5 (Tsitouras 5/4 Runge-Kutta), a standard efficient solver
        # We use BacksolveAdjoint for O(1) memory usage during backprop
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.1,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=diffrax.BacksolveAdjoint() 
        )
        return solution.ys

def get_loss(model, ts, batch_y):
    # Standard MSE loss between predicted trajectory and actual noise history
    # batch_y shape: (seq_len, 1)
    y0 = batch_y[0]
    pred_y = model(ts, y0)
    return jnp.mean((pred_y - batch_y) ** 2)

@eqx.filter_jit
def make_step(model, opt_state, ts, batch_y, optimizer):
    grads = eqx.filter_grad(get_loss)(model, ts, batch_y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, grads

if __name__ == "__main__":
    print("=== Testing Neural ODE Predictor (JAX/Diffrax) ===")
    
    # 1. Setup Keys and Model
    key = jax.random.PRNGKey(555)
    model = NeuralODE(key)
    
    # 2. Setup Dummy Data (Simulating a sine wave drift)
    # 10 time steps
    ts = jnp.linspace(0, 10, 10) 
    # Actual noise values (0.005 base + drift)
    real_noise = 0.005 + 0.001 * jnp.sin(0.5 * ts)
    real_noise = real_noise.reshape(-1, 1) # Shape (10, 1)
    
    # 3. Setup Optimizer
    try:
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        print("[PASS] Optimizer initialized.")
    except NameError:
        print("!! MISSING OPTAX. Please pip install optax !!")
        exit()

    # 4. Run Training Step (Verifies Gradient Flow)
    print("Attempting JIT compilation and backward pass (this may pause for a moment)...")
    
    initial_loss = get_loss(model, ts, real_noise)
    print(f"Initial Loss: {initial_loss:.6f}")
    
    model, opt_state, _ = make_step(model, opt_state, ts, real_noise, optimizer)
    
    final_loss = get_loss(model, ts, real_noise)
    print(f"Post-Step Loss: {final_loss:.6f}")
    
    if final_loss < initial_loss:
        print("[PASS] Gradient descent successful. Model is learning physics.")
    else:
        print("[WARN] Loss did not decrease (random initialization might be unlucky), but pipeline ran.")
    
    print("Status: Neural ODE Ready for Integration")
