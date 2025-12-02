import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from resonance_dissonance import dissonance, resonance_nudge
from symbolic_invariants import check_energy_conservation

def run_baseline(steps=500):
    pos = torch.tensor([1.0, 0.0])
    vel = torch.tensor([0.0, 1.0])
    dt = 0.01
    history = [pos.norm().item()]

    for _ in range(steps):
        r = torch.norm(pos)
        acc = -pos / r**3
        vel += acc * dt
        pos += vel * dt
        history.append(pos.norm().item())
    return history

def run_resonant(steps=500):
    pos = torch.tensor([1.0, 0.0], requires_grad=False)
    vel = torch.tensor([0.0, 1.0])
    dt = 0.01
    history = [pos.norm().item()]
    total_diss = 0

    for _ in range(steps):
        r = torch.norm(pos)
        acc = -pos / (r**3 + 1e-8)
        vel += acc * dt
        pos += vel * dt

        # Dissonance from energy conservation
        T_val = check_energy_conservation(pos, vel, target_E=-0.5)
        diss = dissonance(T_val).item()
        total_diss += diss

        # Apply resonance correction
        pos = resonance_nudge(pos, diss)

        history.append(pos.norm().item())

    avg_diss = total_diss / steps
    return history, avg_diss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    print(f"Running {args.steps} steps...")
    baseline = run_baseline(args.steps)
    resonant, avg_diss = run_resonant(args.steps)

    plt.plot(baseline, label=f"Baseline (final={baseline[-1]:.3f})", alpha=0.7)
    plt.plot(resonant, label=f"Resonant (final={resonant[-1]:.3f}, diss={avg_diss:.3f})", linewidth=2)
    plt.axhline(1.0, color='black', linestyle='--', label="True orbit")
    plt.legend()
    plt.title("Resonance Pruning vs Vanilla Integration")
    plt.xlabel("Steps")
    plt.ylabel("Orbital radius")
    plt.savefig("orbit_comparison.png", dpi=200)
    plt.show()

    print(f"Resonant drift: {abs(resonant[-1] - 1.0)*100:.2f}%")
