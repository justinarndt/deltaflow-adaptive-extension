import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from generate_challenge import DriftGenerator
from drift_predictor import NeuralODE, make_step
import sys

# JAX Config
jax.config.update("jax_enable_x64", True)

def generate_plot():
    print("=== Generating Figure 1: Engineering Proof (Optimized) ===")
    
    # 1. Setup
    STEPS = 300
    WARMUP = 50
    WINDOW_SIZE = 50 # Fixed buffer size to stop JAX recompilation
    LR = 0.005
    
    gen = DriftGenerator(distance=3, rounds=3, base_error=0.01)
    model = NeuralODE(jax.random.PRNGKey(42))
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    true_hist = []
    pred_hist = []
    
    print(f"Simulating {STEPS} steps (with sliding window)...")
    
    for t in range(STEPS):
        # Progress Bar
        sys.stdout.write(f"\rStep {t}/{STEPS}")
        sys.stdout.flush()
        
        _, _, actual = gen.generate_batch(100, t)
        true_hist.append(actual)
        
        val = np.nan 
        
        if t >= WARMUP:
            # --- OPTIMIZATION: SLIDING WINDOW ---
            # Only train on the last 50 data points.
            # This keeps the array shape constant, so JAX compiles ONCE, not 300 times.
            start_idx = max(0, len(true_hist) - WINDOW_SIZE)
            
            # Create windowed arrays
            window_data = true_hist[start_idx:]
            # We must map timestamps relative to the window for consistency or absolute. 
            # Absolute is fine for ODEs as long as we predict forward.
            window_ts = range(start_idx, len(true_hist))
            
            ts_jax = jnp.array(window_ts, dtype=jnp.float64)
            ys_jax = jnp.array(window_data, dtype=jnp.float64).reshape(-1, 1)
            
            # Train (Fast)
            for _ in range(3):
                 model, opt_state, _ = make_step(model, opt_state, ts_jax, ys_jax, optimizer)
            
            # Predict
            future_ts = jnp.array([t, t+1], dtype=jnp.float64)
            # Start prediction from the very last known point
            traj = model(future_ts, ys_jax[-1])
            val = float(traj[-1, 0])
        
        pred_hist.append(val)

    print("\nRendering 'engineering_proof.png'...")
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Plot Ground Truth
    plt.plot(true_hist, label='Environment (Drifting Noise)', color='black', alpha=0.3, linewidth=2)
    
    # Plot Prediction
    plt.plot(pred_hist, label='Deltaflow-Adaptive (Neural ODE)', color='#007acc', linewidth=2)
    
    # Add Markers
    plt.axvline(x=WARMUP, color='red', linestyle='--', label='Online Learning Start')
    
    plt.title('Engineering Proof: Dual-Timescale Drift Tracking')
    plt.xlabel('Logical Cycles (Time)')
    plt.ylabel('Physical Error Rate (p)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('engineering_proof.png')
    print("[PASS] Plot saved to project directory.")

if __name__ == "__main__":
    generate_plot()
