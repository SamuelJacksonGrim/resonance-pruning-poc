import sympy as sp

# Physics invariants (expandable)
x, y, vx, vy, m = sp.symbols('x y vx vy m')
energy_eq = sp.Eq(0.5 * m * (vx**2 + vy**2) - m / sp.sqrt(x**2 + y**2), sp.symbols('E'))
momentum_eq = sp.Eq(m * vx, sp.symbols('px'))
angular_momentum_eq = sp.Eq(m * (x * vy - y * vx), sp.symbols('L'))

def check_energy_conservation(pos, vel, target_E=-0.5):
    """Numeric approximation of total energy"""
    r = torch.norm(pos)
    kinetic = 0.5 * torch.sum(vel ** 2)
    potential = -1.0 / (r + 1e-6)
    return kinetic + potential - target_E  # deviation from truth
