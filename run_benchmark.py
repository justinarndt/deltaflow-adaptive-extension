import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from generate_challenge import DriftGenerator
from drift_predictor import NeuralODE, make_step
import time

# JAX Config
jax.config.update("jax_enable_x64", True)

def run_benchmark():
    print("=== Deltaflow-Adaptive: Definitive Engineering Proof (500 Steps) ===")
    
    # 1. Configuration (Production Settings)
    STEPS = 500
    WARMUP = 50
    LR = 0.005  # The tuned "Stabilized" value
    
    # 2. Initialize
    print(f"[INIT] Environment: Surface Code (d=3, Rounds=3)")
    print(f"[INIT] AI Engine: Neural ODE (LR={LR})")
    
    gen = DriftGenerator(distance=3, rounds=3, base_error=0.01)
    key = jax.random.PRNGKey(42) # Fixed seed for reproducibility
    model = NeuralODE(key)
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Metrics
    history_true = []
    history_pred = []
    tracking_errors = []
    
    start_time = time.time()
    
    print("\n[RUNNING] Simulation in progress...")
    print(f"{'STEP':<5} | {'ACTUAL':<10} | {'PREDICT':<10} | {'ERROR':<10} | {'STATUS'}")
    print("-" * 65)
    
    for t in range(STEPS):
        # A. Environment
        _, _, actual_noise = gen.generate_batch(batch_size=100, time_step=t)
        
        # B. Slow Path (AI)
        history_true.append(actual_noise)
        ts_jax = jnp.array(range(len(history_true)), dtype=jnp.float64)
        ys_jax = jnp.array(history_true, dtype=jnp.float64).reshape(-1, 1)
        
        pred_noise = 0.0
        status = "WARMUP"
        
        if t >= WARMUP:
            # Train
            for _ in range(3): # Lower iterations for speed in long run
                model, opt_state, _ = make_step(model, opt_state, ts_jax, ys_jax, optimizer)
            
            # Predict t+1
            future_ts = jnp.array([t, t+1], dtype=jnp.float64)
            traj = model(future_ts, ys_jax[-1])
            pred_noise = float(traj[-1, 0])
            
            status = "TRACKING"
            
            # Record Metric
            error = abs(pred_noise - actual_noise)
            tracking_errors.append(error)
            
            # Log every 25 steps to avoid terminal spam
            if t % 25 == 0:
                 print(f"{t:<5} | {actual_noise:.5f}    | {pred_noise:.5f}    | {error:.5f}    | {status}")
        else:
             if t % 25 == 0:
                 print(f"{t:<5} | {actual_noise:.5f}    | {'---':<10}    | {'---':<10}    | {status}")

    total_time = time.time() - start_time
    avg_error = np.mean(tracking_errors)
    
    print("-" * 65)
    print("=== RESULTS SUMMARY ===")
    print(f"Total Simulation Time: {total_time:.2f}s")
    print(f"Total Steps: {STEPS}")
    print(f"Average Tracking Error: {avg_error:.6f}")
    
    if avg_error < 0.001:
        print("[PASS] SUCCESS: AI maintained lock on quantum noise drift.")
    else:
        print("[FAIL] WARNING: Tracking error exceeded tolerance.")

if __name__ == "__main__":
    run_benchmark()
