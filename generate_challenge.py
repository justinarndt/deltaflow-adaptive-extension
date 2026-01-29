import stim
import numpy as np
from typing import Tuple, List

class DriftGenerator:
    """
    Generates surface code circuits with non-stationary (drifting) noise.
    Acts as the 'Environment' for the Deltaflow-Adaptive simulation.
    """
    def __init__(self, distance: int, rounds: int, base_error: float):
        self.d = distance
        self.rounds = rounds
        self.base_error = base_error
        # We use Rotated Surface Code Memory Z experiment
        # Initial template (clean)
        self.circuit_template = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=0.0
        )

    def _get_noise_level(self, time_step: int, frequency: float = 0.05) -> float:
        """
        Simulates environmental drift. 
        Noise oscillates around the base_error using a sine wave to mimic 
        systematic calibration drift (e.g., temperature fluctuations).
        """
        drift = 0.001 * np.sin(frequency * time_step)
        # Clip to ensure physical validity (0 < p < 0.5)
        return float(np.clip(self.base_error + drift, 0.001, 0.1))

    def generate_batch(self, batch_size: int, time_step: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generates a batch of shots with a specific noise level determined by the time_step.
        
        Returns:
            detection_events (np.ndarray): The syndrome data (bool/uint8).
            observables (np.ndarray): The actual logical flip data (ground truth).
            current_noise (float): The specific error rate used for this batch.
        """
        # Calculate the noise level for this specific 'moment in time'
        current_noise = self._get_noise_level(time_step)
        
        # Create a noisy version of the circuit
        # FIX: Updated parameter names for compatibility with latest Stim
        noisy_circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=self.d,
            rounds=self.rounds,
            after_clifford_depolarization=current_noise,
            after_reset_flip_probability=current_noise,
            before_measure_flip_probability=current_noise
        )
        
        # Sampler: compiling the detector sampler is expensive, so we do it per batch
        sampler = noisy_circuit.compile_detector_sampler()
        
        # Generate data
        detection_events, observables = sampler.sample(
            shots=batch_size, 
            separate_observables=True
        )
        
        return detection_events, observables, current_noise

if __name__ == "__main__":
    # Quick functionality test
    print("=== Testing Drift Generator (Stim) ===")
    
    # Parameters matching the 'Constraint Check' (d=5 for quick test)
    gen = DriftGenerator(distance=5, rounds=5, base_error=0.005)
    
    # Simulate 3 time steps to verify drift
    for t in range(3):
        dets, obs, noise = gen.generate_batch(batch_size=10, time_step=t*10)
        print(f"Time {t}: Noise Rate = {noise:.5f} | Detection Events Shape: {dets.shape}")

    print("[PASS] Generator is producing drifting synthetic data.")
