import jax
import stim
import diffrax
import numpy as np

def verify_environment():
    print("=== Deltaflow-Adaptive WSL2 Environment Check ===")
    
    # 1. Verify Stim (The Environment)
    print(f"[PASS] Stim version: {stim.__version__}")
    
    # 2. Verify JAX GPU Access (Critical for RTX 4060)
    devices = jax.devices()
    print(f"[CHECK] JAX Devices: {devices}")
    if 'gpu' not in str(devices[0]).lower():
        print("!! WARNING !! JAX is running on CPU. Check your CUDA install.")
    else:
        print(f"[PASS] JAX is utilizing the GPU.")

    # 3. Verify Diffrax (The Adjoint Solver)
    try:
        solver = diffrax.Tsit5()
        adjoint = diffrax.BacksolveAdjoint()
        print(f"[PASS] Diffrax Adjoint method available.")
    except Exception as e:
        print(f"[FAIL] Diffrax setup issue: {e}")

    print(f"Status: Ready for Phase 1")

if __name__ == "__main__":
    verify_environment()
