import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from generate_challenge import DriftGenerator
from drift_predictor import NeuralODE, make_step, get_loss

# JAX Config
jax.config.update("jax_enable_x64", True)

def weight_function(p):
    """
    Standard MWPM weight calculation: w = log((1-p)/p)
    As error rate (p) increases, the weight of the parity check decreases.
    """
    # Avoid div by zero
    p = jnp.clip(p, 0.0001, 0.499)
    return jnp.log((1 - p) / p)

def run_simulation():
    print("=== Deltaflow-Adaptive: System Integration Test ===")
    
    # 1. Configuration
    STEPS = 30
    WARMUP_STEPS = 10  # How long we watch before we start predicting
    LR = 0.005
    
    # 2. Initialize Components
    print("[INIT] Spinnning up Environment (Stim)...")
    # We use a small code (d=3) for fast iteration in the loop
    gen = DriftGenerator(distance=3, rounds=3, base_error=0.01)
    
    print("[INIT] Initializing Neural ODE (Slow Path)...")
    key = jax.random.PRNGKey(0)
    model = NeuralODE(key)
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Storage for history
    history_t = []
    history_p = [] # The 'observed' error rate (in reality, derived from syndrome density)
    
    print(f"\n[START] Starting Real-Time Simulation ({STEPS} steps)...")
    print(f"{'TIME':<5} | {'ACTUAL p':<10} | {'PREDICT p':<10} | {'STATUS':<15} | {'WEIGHT ADJ'}")
    print("-" * 75)
    
    # 3. The Control Loop
    for t in range(STEPS):
        # --- A. The Environment Step ---
        # Generate a batch of data from the quantum processor
        # We assume 1 step = 1 unit of time
        dets, obs, actual_noise = gen.generate_batch(batch_size=100, time_step=t)
        
        # In a real system, we don't know 'actual_noise'. 
        # We estimate it from Syndrome Density (fraction of checks that failed).
        # For Surface Code, p ~ density / constant. 
        # For this proof, we pass the actual_noise to the trainer to prove the ODE learns the *trend*.
        
        history_t.append(t)
        history_p.append(actual_noise)
        
        # --- B. The Slow Path (AI Update) ---
        # We convert history to JAX arrays for training
        ts_jax = jnp.array(history_t, dtype=jnp.float64)
        ys_jax = jnp.array(history_p, dtype=jnp.float64).reshape(-1, 1)
        
        pred_noise = 0.0
        status = "Gathering"
        weight_delta = 0.0
        
        if t >= WARMUP_STEPS:
            # 1. Train the model on *past* data (Online Learning)
            # We take a few gradient steps to adapt to the new data point
            for _ in range(5):
                model, opt_state, grads = make_step(model, opt_state, ts_jax, ys_jax, optimizer)
            
            # 2. Predict *Future* Drift (Next Step)
            # We ask the ODE: "Given where we are, where will noise be at t+1?"
            # We predict 1 step ahead
            future_ts = jnp.array([t, t+1], dtype=jnp.float64)
            current_y = ys_jax[-1] # Start from last known point
            
            # Integrate ODE forward
            trajectory = model(future_ts, current_y)
            
            # FIX: Index [0] to get the scalar value from the (1,) vector
            pred_noise = float(trajectory[-1, 0]) # The value at t+1
            
            status = "ADAPTIVE"
            
            # --- C. The Fast Path (Weight Adjustment) ---
            # Calculate how much we would change the Decoder Weights
            # Static Weight (based on initial 0.01)
            w_static = weight_function(0.01)
            # Adaptive Weight (based on prediction)
            w_adapt = weight_function(pred_noise)
            
            weight_delta = float(jnp.abs(w_static - w_adapt))
        
        # Output Log
        # If predicting, compare Prediction vs Actual (for the *next* step, but we show current sync for readability)
        print(f"{t:<5} | {actual_noise:.5f}    | {pred_noise:.5f}    | {status:<15} | {weight_delta:.4f}")

    print("-" * 75)
    print("[SUCCESS] Simulation Complete.")
    print("Interpretation:")
    print("1. 'ACTUAL p' drifts over time (Sine wave).")
    print("2. 'PREDICT p' should track this drift after the Warmup phase.")
    print("3. 'WEIGHT ADJ' shows the magnitude of correction applied to the FPGA solver.")

if __name__ == "__main__":
    run_simulation()
