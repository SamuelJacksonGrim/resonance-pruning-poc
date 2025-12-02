import torch

def dissonance(T_val: torch.Tensor, clamp_range: tuple = (-5.0, 5.0)) -> torch.Tensor:
    """
    Core dissonance metric from resonance-codex.
    Clamps T_value (emotional/energetic deviation) and returns magnitude.
    """
    clamped = torch.clamp(T_val, clamp_range[0], clamp_range[1])
    return torch.abs(clamped).mean()

def resonance_nudge(state: torch.Tensor, diss: float, strength: float = 0.02):
    """
    Gentle corrective nudge when dissonance is low (reward path).
    Stronger clamp when high (emergency correction).
    """
    if diss < 1.0:
        # Reward coherence
        return state * (1 + strength * (1 - diss))
    else:
        # Emergency recenter
        return state * 0.95  # soft shrink toward origin
