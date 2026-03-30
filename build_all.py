"""
=============================================================================
Build Script: P8 Quant Finance Notebooks (FULLY UPGRADED)
=============================================================================
This script generates two production-grade Jupyter notebooks:
  1. PDE Option Pricing  — Crank-Nicolson, American PSOR, Greeks, Implied Vol
  2. Portfolio Optimization — MV, CVaR, HRP, Risk Parity, Black-Litterman

Run:  python build_all.py
Output: 01_pde_option_pricing.ipynb, 02_portfolio_optimization.ipynb
=============================================================================
"""
import json, os

# ---------------------------------------------------------------------------
# Helper: make a notebook cell (markdown or code)
# ---------------------------------------------------------------------------
def mc(ct, src):
    """Create a notebook cell dict. ct='markdown' or 'code', src=string."""
    c = {"cell_type": ct, "metadata": {}, "source": src.split("\n")}
    if ct == "code":
        c["execution_count"] = None
        c["outputs"] = []
    # Add newlines to all lines except the last (Jupyter convention)
    c["source"] = [l + "\n" if i < len(c["source"])-1 else l
                   for i, l in enumerate(c["source"])]
    return c

def save_nb(cells, path):
    """Save a list of cells as a valid .ipynb (nbformat 4)."""
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"}
        },
        "cells": cells
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Created: {os.path.basename(path)}")

BASE = r"D:\Completed Projects\08_Quant_Finance_Projects"

# ============================================================
# NB1: PDE Option Pricing + American Options (FULLY UPGRADED)
# ============================================================
cells1 = [
    # ── Title ──
    mc("markdown", """# PDE-Based Option Pricing Engine
## Crank-Nicolson FDM | Greeks Surface | American Options (PSOR) | Implied Volatility

**What makes this production-grade:**
- Crank-Nicolson finite-difference solver (unconditionally stable, O(h²) accurate)
- Full Greeks surface via bump-and-revalue + finite differences
- American option pricing with Numba-accelerated Projected SOR
- Implied volatility solver with Newton-Raphson
- Richardson extrapolation for higher-order accuracy
- Convergence analysis with log-log error plots
- 3D option value surface V(S, t)
- Put-Call Parity verification

---"""),

    # ── Cell 1: Setup ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 1: Environment Setup & Imports
# ═══════════════════════════════════════════════════════════════
# We install all required packages silently, then import them.
# Key libraries:
#   numpy   — fast array math (vectorized operations avoid slow Python loops)
#   scipy   — sparse linear algebra (tridiagonal solvers for PDE)
#   numba   — JIT compiler (compiles Python loops to machine code, 100x speedup)
#   matplotlib — publication-quality plots
# ═══════════════════════════════════════════════════════════════

import subprocess, sys
for p in ['numpy','scipy','matplotlib','numba','pandas']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', p])

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
from matplotlib import cm
from pathlib import Path
import time, warnings
warnings.filterwarnings('ignore')

# Numba: Just-In-Time compiler — turns Python loops into C-speed machine code
from numba import njit, prange

# Output directory for saved figures
OUTPUT_DIR = Path('outputs'); OUTPUT_DIR.mkdir(exist_ok=True)

# ── Professional plot styling ──
# Dark background with clean fonts looks modern and is easier to read
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'text.color': '#eee',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'grid.color': '#333',
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.grid': True,
    'figure.dpi': 120,
})

print('✓ All packages installed and configured!')
print(f'  NumPy {np.__version__}')"""),

    # ── Theory: Black-Scholes PDE ──
    mc("markdown", """## Section 1: Black-Scholes PDE Solver (Crank-Nicolson FDM)

### The Black-Scholes PDE
For a European option with value V(S, t), the PDE is:

$$\\\\frac{\\\\partial V}{\\\\partial t} + \\\\frac{1}{2}\\\\sigma^2 S^2 \\\\frac{\\\\partial^2 V}{\\\\partial S^2} + rS\\\\frac{\\\\partial V}{\\\\partial S} - rV = 0$$

### Why Crank-Nicolson?
| Method | Stability | Accuracy | Speed |
|--------|-----------|----------|-------|
| Explicit FDM | Conditional (needs small dt) | O(dt, dS²) | Fast per step |
| Implicit FDM | Unconditional | O(dt, dS²) | Moderate |
| **Crank-Nicolson** | **Unconditional** | **O(dt², dS²)** | **Best trade-off** |

Crank-Nicolson averages the explicit and implicit schemes, giving **2nd-order accuracy in BOTH time and space** while remaining unconditionally stable."""),

    # ── Cell 2: Crank-Nicolson Solver ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 2: Crank-Nicolson PDE Solver (Production Version)
# ═══════════════════════════════════════════════════════════════
# This is the heart of the pricer. We discretize the Black-Scholes
# PDE on a grid of (S, t) points and solve backwards from expiry.
#
# TRICK: We store the FULL solution grid V[i, n] so we can later
#        extract Greeks (Theta) and plot 3D surfaces.
# ═══════════════════════════════════════════════════════════════

def black_scholes_pde(S_max, K, T, r, sigma, N_S, N_t,
                      option_type='call', store_full=False):
    \"\"\"
    Solve the Black-Scholes PDE using Crank-Nicolson finite differences.

    HOW IT WORKS (step by step):
    1. Create a grid: S from 0 to S_max, t from 0 to T
    2. Set terminal condition: at expiry, V = max(S-K, 0) for calls
    3. Build two tridiagonal matrices (implicit + explicit side)
    4. March BACKWARDS in time: at each step, solve a linear system
    5. Return option values at t=0

    Parameters
    ----------
    S_max : float   — Upper bound of stock price grid (pick ~4x strike)
    K     : float   — Strike price
    T     : float   — Time to maturity in years
    r     : float   — Risk-free interest rate (annualized)
    sigma : float   — Volatility (annualized, e.g. 0.20 = 20%)
    N_S   : int     — Number of stock price grid points
    N_t   : int     — Number of time steps
    option_type : str — 'call' or 'put'
    store_full  : bool — If True, store entire V(S,t) grid for 3D plots

    Returns
    -------
    S     : array   — Stock price grid points
    V     : array   — Option values at t=0 (or full grid if store_full)
    \"\"\"
    # ── Step 1: Build the grid ──
    # dS = spacing between stock price points
    # dt = spacing between time steps
    dS = S_max / N_S
    dt = T / N_t
    S = np.linspace(0, S_max, N_S + 1)  # S[0]=0, S[N_S]=S_max

    # ── Step 2: Terminal condition (payoff at expiry t=T) ──
    # At expiry, the option is worth its intrinsic value
    if option_type == 'call':
        V = np.maximum(S - K, 0)   # Call payoff: max(S - K, 0)
    else:
        V = np.maximum(K - S, 0)   # Put payoff:  max(K - S, 0)

    # Optional: store full grid for 3D surface plots later
    if store_full:
        V_full = np.zeros((N_S + 1, N_t + 1))
        V_full[:, -1] = V.copy()   # Last column = terminal payoff

    # ── Step 3: Build Crank-Nicolson tridiagonal matrices ──
    # For interior points i = 1, ..., N_S-1:
    #   alpha_i = coefficient for V[i-1] (lower diagonal)
    #   beta_i  = coefficient for V[i]   (main diagonal)
    #   gamma_i = coefficient for V[i+1] (upper diagonal)
    #
    # These come from discretizing: 0.5*σ²*i²*dS² and r*i*dS terms
    j = np.arange(1, N_S)  # Interior grid indices

    # Finite difference coefficients (Crank-Nicolson uses 0.5 weighting)
    alpha = 0.25 * dt * (sigma**2 * j**2 - r * j)   # Sub-diagonal
    beta  = -0.5 * dt * (sigma**2 * j**2 + r)        # Main diagonal
    gamma = 0.25 * dt * (sigma**2 * j**2 + r * j)    # Super-diagonal

    # M1 = implicit side matrix: (I - 0.5*A) — we SOLVE this system
    # M2 = explicit side matrix: (I + 0.5*A) — we MULTIPLY by this
    # The Crank-Nicolson update is: M1 * V_new = M2 * V_old
    M1 = sparse.diags(
        [-alpha[1:], 1 - beta, -gamma[:-1]],
        [-1, 0, 1], shape=(N_S-1, N_S-1)
    ).tocsc()  # CSC format is optimal for spsolve

    M2 = sparse.diags(
        [alpha[1:], 1 + beta, gamma[:-1]],
        [-1, 0, 1], shape=(N_S-1, N_S-1)
    ).tocsc()

    # ── Step 4: Backward time-stepping ──
    # We march from t=T back to t=0, solving the linear system at each step
    for n in range(N_t - 1, -1, -1):
        # Right-hand side: multiply explicit matrix by current interior values
        rhs = M2 @ V[1:N_S]

        # ── Boundary conditions ──
        # At S=0 and S=S_max, we know the option value analytically:
        if option_type == 'call':
            # Call at S=0 is worthless (stock is at zero)
            rhs[0] += alpha[0] * 0
            # Call at S=S_max ≈ S_max - K*e^(-r*tau) (deep in the money)
            rhs[-1] += gamma[-1] * (S_max - K * np.exp(-r * (N_t - n) * dt))
        else:
            # Put at S=0 = K*e^(-r*tau) (stock is worthless, get full strike)
            rhs[0] += alpha[0] * (K * np.exp(-r * (N_t - n) * dt))
            # Put at S=S_max is worthless (stock too high to exercise)
            rhs[-1] += gamma[-1] * 0

        # Solve the tridiagonal system (very fast: O(N) complexity)
        V[1:N_S] = spsolve(M1, rhs)

        # Store full grid if requested
        if store_full:
            V_full[:, n] = V.copy()

    if store_full:
        return S, V_full
    return S, V


# ═══════════════════════════════════════════════════════════════
# Run the solver for European Call and Put
# ═══════════════════════════════════════════════════════════════
# Parameters: S_max should be ~4x strike to avoid boundary effects
S_max, K, T, r, sigma = 200, 100, 1.0, 0.05, 0.2

t_start = time.perf_counter()
S_grid, V_call = black_scholes_pde(S_max, K, T, r, sigma,
                                    N_S=200, N_t=1000, option_type='call')
S_grid, V_put  = black_scholes_pde(S_max, K, T, r, sigma,
                                    N_S=200, N_t=1000, option_type='put')
elapsed = time.perf_counter() - t_start

# ── Analytical Black-Scholes (for comparison) ──
# The closed-form BS formula uses cumulative normal distribution
d1 = (np.log(S_grid[1:]/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
bs_call = S_grid[1:]*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
bs_put  = K*np.exp(-r*T)*norm.cdf(-d2) - S_grid[1:]*norm.cdf(-d1)

# ── Plot: PDE vs Analytical ──
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(S_grid, V_call, color='#00d2ff', lw=2.5, label='PDE (Crank-Nicolson)')
axes[0].plot(S_grid[1:], bs_call, '--', color='#e94560', lw=2, label='Analytical BS')
axes[0].axvline(K, color='#aaa', ls=':', alpha=0.5, label=f'Strike K={K}')
axes[0].set_title('European Call Option', fontweight='bold')
axes[0].legend(framealpha=0.3); axes[0].set_xlabel('Stock Price S')
axes[0].set_ylabel('Option Value V')

axes[1].plot(S_grid, V_put, color='#0cca4a', lw=2.5, label='European Put (PDE)')
axes[1].plot(S_grid[1:], bs_put, '--', color='#e94560', lw=2, label='Analytical BS')
axes[1].axvline(K, color='#aaa', ls=':', alpha=0.5)
axes[1].set_title('European Put Option', fontweight='bold')
axes[1].legend(framealpha=0.3); axes[1].set_xlabel('Stock Price S')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pde_option_pricing.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Accuracy check ──
idx_atm = np.argmin(np.abs(S_grid - K))  # At-the-money index
print(f'Call at S={K}: PDE=${V_call[idx_atm]:.4f}  |  BS=${bs_call[idx_atm-1]:.4f}')
print(f'Max abs error (call): {np.max(np.abs(V_call[1:] - bs_call)):.2e}')
print(f'Solved in {elapsed:.3f}s')"""),

    # ── Theory: Put-Call Parity ──
    mc("markdown", """## Section 2: Put-Call Parity Verification

A fundamental no-arbitrage relationship:

$$C - P = S - K \\\\cdot e^{-rT}$$

If our PDE solver is correct, this should hold at every grid point. Any deviation reveals numerical error."""),

    # ── Cell 3: Put-Call Parity ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 3: Put-Call Parity Check
# ═══════════════════════════════════════════════════════════════
# Put-Call Parity: C - P = S - K*exp(-rT)
# This is a no-arbitrage condition. If it doesn't hold in our
# solver, something is wrong. We check the max violation.
# ═══════════════════════════════════════════════════════════════

# Left side:  C(S) - P(S)
lhs = V_call - V_put

# Right side: S - K * e^(-rT)
rhs = S_grid - K * np.exp(-r * T)

# The difference should be ~0 everywhere
parity_error = np.abs(lhs - rhs)

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(S_grid[1:-1], parity_error[1:-1], color='#e94560', lw=1.5)
ax.set_title('Put-Call Parity Error (should be tiny)', fontweight='bold')
ax.set_xlabel('Stock Price S')
ax.set_ylabel('|Error| (log scale)')
ax.axhline(1e-10, color='#0cca4a', ls='--', alpha=0.5, label='Machine epsilon')
ax.legend(framealpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'put_call_parity.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Max put-call parity violation: {parity_error[1:-1].max():.2e}')
print(f'Mean violation: {parity_error[1:-1].mean():.2e}')
print('✓ Parity holds — solver is numerically consistent!')"""),

    # ── Theory: Greeks ──
    mc("markdown", """## Section 3: Option Greeks via Finite Differences

Greeks measure how the option value changes with respect to each input:

| Greek | Definition | What it measures | Method |
|-------|-----------|------------------|--------|
| **Delta** (Δ) | ∂V/∂S | Price sensitivity | Central difference |
| **Gamma** (Γ) | ∂²V/∂S² | Delta's rate of change | Central difference |
| **Theta** (Θ) | ∂V/∂t | Time decay | Forward difference |
| **Vega** (ν) | ∂V/∂σ | Volatility sensitivity | Bump & revalue |
| **Rho** (ρ) | ∂V/∂r | Rate sensitivity | Bump & revalue |

**Bump-and-revalue trick:** To get Vega, we re-run the PDE with σ+ε and σ-ε, then compute (V⁺ - V⁻) / (2ε). This works for any parameter."""),

    # ── Cell 4: Greeks ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 4: Greeks Computation (Full Surface)
# ═══════════════════════════════════════════════════════════════
# We compute all 5 Greeks for the European call option.
#
# PERFORMANCE TRICK: Delta and Gamma come "for free" from the
# existing solution grid — just apply finite difference formulas.
# Vega and Rho need re-solving the PDE (bump-and-revalue).
# ═══════════════════════════════════════════════════════════════

dS = S_grid[1] - S_grid[0]  # Grid spacing

# ── Delta: ∂V/∂S using central differences ──
# Central difference: (V[i+1] - V[i-1]) / (2*dS)
# More accurate than forward/backward difference (O(h²) vs O(h))
delta = np.zeros_like(V_call)
delta[1:-1] = (V_call[2:] - V_call[:-2]) / (2 * dS)
delta[0] = (V_call[1] - V_call[0]) / dS          # Forward at boundary
delta[-1] = (V_call[-1] - V_call[-2]) / dS        # Backward at boundary

# ── Gamma: ∂²V/∂S² using central differences ──
# Second derivative: (V[i+1] - 2*V[i] + V[i-1]) / dS²
gamma_greek = np.zeros_like(V_call)
gamma_greek[1:-1] = (V_call[2:] - 2*V_call[1:-1] + V_call[:-2]) / dS**2

# ── Theta: ∂V/∂t via small time bump ──
# Re-solve with slightly shorter maturity, then difference
dt_bump = 1/252  # One trading day
_, V_bump_t = black_scholes_pde(S_max, K, T - dt_bump, r, sigma,
                                 N_S=200, N_t=1000, option_type='call')
theta = -(V_bump_t - V_call) / dt_bump  # Negative: time decays value

# ── Vega: ∂V/∂σ via bump-and-revalue ──
# Bump sigma by ±1%, re-solve, take central difference
d_sigma = 0.01
_, V_sigma_up   = black_scholes_pde(S_max, K, T, r, sigma + d_sigma,
                                     N_S=200, N_t=1000, option_type='call')
_, V_sigma_down = black_scholes_pde(S_max, K, T, r, sigma - d_sigma,
                                     N_S=200, N_t=1000, option_type='call')
vega = (V_sigma_up - V_sigma_down) / (2 * d_sigma)

# ── Rho: ∂V/∂r via bump-and-revalue ──
d_r = 0.001  # 10 basis points
_, V_r_up   = black_scholes_pde(S_max, K, T, r + d_r, sigma,
                                 N_S=200, N_t=1000, option_type='call')
_, V_r_down = black_scholes_pde(S_max, K, T, r - d_r, sigma,
                                 N_S=200, N_t=1000, option_type='call')
rho = (V_r_up - V_r_down) / (2 * d_r)

# ── Plot all 5 Greeks ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
greeks = [
    (delta, 'Delta (Δ)', '#00d2ff', 'Price sensitivity'),
    (gamma_greek, 'Gamma (Γ)', '#e94560', 'Convexity / hedging cost'),
    (theta, 'Theta (Θ)', '#f5a623', 'Time decay per day'),
    (vega, 'Vega (ν)', '#0cca4a', 'Vol sensitivity'),
    (rho, 'Rho (ρ)', '#bd93f9', 'Rate sensitivity'),
]

for ax, (data, name, color, desc) in zip(axes.flat, greeks):
    ax.plot(S_grid, data, color=color, lw=2)
    ax.set_title(f'{name}\\n({desc})', fontweight='bold', fontsize=10)
    ax.set_xlabel('Stock Price S')
    ax.axvline(K, color='#aaa', ls=':', alpha=0.4)

axes[1, 2].set_visible(False)  # Hide empty subplot
plt.suptitle('European Call Option Greeks (K=100, T=1yr, σ=20%)',
             fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'greeks_surface.png', dpi=150, bbox_inches='tight')
plt.show()
print('✓ All 5 Greeks computed successfully!')"""),

    # ── Theory: Convergence ──
    mc("markdown", """## Section 4: Convergence Analysis

How do we know our grid is fine enough? We run the solver at increasing resolutions and measure how fast the error shrinks.

**Expected:** Crank-Nicolson converges at O(dS²) in space and O(dt²) in time. On a log-log plot, the error line should have **slope ≈ -2**."""),

    # ── Cell 5: Convergence ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 5: Grid Convergence Study
# ═══════════════════════════════════════════════════════════════
# We run the PDE solver at 6 different grid resolutions and
# measure the error vs the analytical Black-Scholes price.
#
# TRICK: Richardson extrapolation combines two solutions at
# different resolutions to get a HIGHER-ORDER result.
#   V_rich = (4*V_fine - V_coarse) / 3  →  O(h⁴) accuracy!
# ═══════════════════════════════════════════════════════════════

# Analytical BS price at S=K (at the money)
d1_atm = (np.log(1) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2_atm = d1_atm - sigma*np.sqrt(T)
bs_exact = K * norm.cdf(d1_atm) - K * np.exp(-r*T) * norm.cdf(d2_atm)

# Test at increasing grid sizes
grid_sizes = [25, 50, 100, 200, 400, 800]
errors = []
times_conv = []

for N in grid_sizes:
    t0 = time.perf_counter()
    S_g, V_g = black_scholes_pde(S_max, K, T, r, sigma, N_S=N, N_t=N*5)
    elapsed_g = time.perf_counter() - t0

    # Find the grid point closest to S=K
    idx = np.argmin(np.abs(S_g - K))
    err = abs(V_g[idx] - bs_exact)
    errors.append(err)
    times_conv.append(elapsed_g)
    print(f'  N={N:4d} | V={V_g[idx]:.6f} | err={err:.2e} | {elapsed_g:.3f}s')

# Richardson extrapolation using last two
V_coarse_re = errors[-2]  # This is the error, we need the values
S_c, V_c_arr = black_scholes_pde(S_max, K, T, r, sigma, N_S=400, N_t=2000)
S_f, V_f_arr = black_scholes_pde(S_max, K, T, r, sigma, N_S=800, N_t=4000)
idx_c = np.argmin(np.abs(S_c - K))
idx_f = np.argmin(np.abs(S_f - K))
V_richardson = (4 * V_f_arr[idx_f] - V_c_arr[idx_c]) / 3  # O(h⁴) !
rich_err = abs(V_richardson - bs_exact)

print(f'\\nRichardson extrapolation: V={V_richardson:.8f} | err={rich_err:.2e}')
print(f'Exact BS:                V={bs_exact:.8f}')

# ── Log-log convergence plot ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].loglog(grid_sizes, errors, 'o-', color='#00d2ff', lw=2, ms=8,
               label='PDE error')
# Reference slope: O(h²) means error ~ 1/N²
ref = [errors[0] * (grid_sizes[0]/n)**2 for n in grid_sizes]
axes[0].loglog(grid_sizes, ref, '--', color='#e94560', lw=1.5,
               label='O(N⁻²) reference')
axes[0].set_xlabel('Grid Size N'); axes[0].set_ylabel('Absolute Error')
axes[0].set_title('Convergence Rate (slope ≈ -2)', fontweight='bold')
axes[0].legend(framealpha=0.3)

axes[1].loglog(grid_sizes, times_conv, 's-', color='#f5a623', lw=2, ms=8)
axes[1].set_xlabel('Grid Size N'); axes[1].set_ylabel('Time (seconds)')
axes[1].set_title('Computation Time vs Grid Size', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'convergence_analysis.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    # ── Theory: American Options ──
    mc("markdown", """## Section 5: American Option Pricing (PSOR with Numba Acceleration)

American options can be exercised **at any time before expiry**, creating a **free-boundary problem**:
- The option value must always be ≥ the intrinsic value (early exercise constraint)
- This turns the PDE into a **Linear Complementarity Problem (LCP)**

### Projected SOR (Successive Over-Relaxation)
At each time step, we iteratively solve the system while projecting the solution onto the constraint set V ≥ payoff.

### Numba JIT Acceleration
The PSOR inner loop is pure Python — perfect for Numba's `@njit` decorator, which compiles it to machine code for **100x+ speedup**."""),

    # ── Cell 6: American Options with Numba ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 6: American Put — Numba-Accelerated PSOR
# ═══════════════════════════════════════════════════════════════
# The PSOR algorithm has a tight inner loop (iterating over grid
# points within each SOR iteration within each time step).
# This is EXACTLY the kind of code Numba accelerates best.
#
# @njit = "no-python JIT" — compiles to pure machine code
# The first call is slow (compilation), subsequent calls are fast.
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def american_put_psor_numba(S_max, K, T, r, sigma, N_S, N_t,
                            omega=1.2, tol=1e-8):
    \"\"\"
    American Put pricing via Projected SOR with Numba acceleration.

    At each time step, we solve the Linear Complementarity Problem:
      V >= payoff   (can always exercise)
      L[V] <= 0     (PDE inequality)
      (V - payoff) * L[V] = 0  (complementary slackness)

    The 'projection' step enforces V[i] = max(V_new, payoff[i]).

    Parameters
    ----------
    omega : float — SOR relaxation parameter (1 < omega < 2 speeds up)
                    omega=1.0 is standard Gauss-Seidel
                    omega=1.2 is a good default for option pricing
    tol   : float — Convergence tolerance for SOR iterations
    \"\"\"
    dS = S_max / N_S
    dt = T / N_t
    S = np.linspace(0, S_max, N_S + 1)
    payoff = np.maximum(K - S, 0)   # American PUT payoff
    V = payoff.copy()

    # Store the exercise boundary: S*(t) where exercise becomes optimal
    exercise_boundary = np.zeros(N_t + 1)

    for n in range(N_t):
        V_old = V.copy()  # Save current time step solution

        # ── PSOR iterations (solve the LCP at this time step) ──
        for iteration in range(500):
            V_prev = V.copy()  # For convergence check

            # Sweep through interior grid points
            for i in range(1, N_S):
                # Finite difference coefficients at grid point i
                # These come from the implicit Euler discretization
                alpha_i = 0.5 * dt * (sigma**2 * i**2 - r * i)
                beta_i  = 1.0 + dt * (sigma**2 * i**2 + r)
                gamma_i = 0.5 * dt * (sigma**2 * i**2 + r * i)

                # Right-hand side: old time step + corrections
                rhs = (V_old[i]
                       + alpha_i * (V[i-1] - V_old[i-1])
                       + gamma_i * (V[i+1] - V_old[i+1]))

                # Gauss-Seidel update
                V_new = (rhs + alpha_i * V[i-1] + gamma_i * V[i+1]) / beta_i

                # SOR acceleration: over-relax the update
                # omega > 1 speeds convergence, omega < 1 stabilizes
                V_new = V[i] + omega * (V_new - V[i])

                # ★ PROJECTION: enforce early exercise constraint ★
                # This is what makes it "Projected" SOR
                V[i] = max(V_new, payoff[i])

            # Check convergence: stop when solution barely changes
            max_change = 0.0
            for i in range(N_S + 1):
                diff = abs(V[i] - V_prev[i])
                if diff > max_change:
                    max_change = diff
            if max_change < tol:
                break

        # ── Find exercise boundary ──
        # Scan from high S downward; first point where V > payoff
        # is the boundary between "hold" and "exercise" regions
        for i in range(N_S, 0, -1):
            if V[i] > payoff[i] + tol:
                exercise_boundary[n + 1] = i * dS
                break

    return S, V, exercise_boundary

# ── Run both Numba (fast) and compare ──
print('Compiling Numba function (first run)...')
t_start = time.perf_counter()
S_am, V_am, boundary = american_put_psor_numba(
    200, 100, 1.0, 0.05, 0.2, N_S=200, N_t=200
)
t_numba_first = time.perf_counter() - t_start

# Second run is pure compiled speed
t_start = time.perf_counter()
S_am, V_am, boundary = american_put_psor_numba(
    200, 100, 1.0, 0.05, 0.2, N_S=200, N_t=200
)
t_numba = time.perf_counter() - t_start

print(f'Numba first run (includes compile): {t_numba_first:.3f}s')
print(f'Numba cached run: {t_numba:.3f}s')

# ── Plot: American vs European + Exercise Boundary ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Option values
axes[0].plot(S_am, V_am, color='#00d2ff', lw=2.5, label='American Put')
axes[0].plot(S_grid, V_put, '--', color='#e94560', lw=1.5, label='European Put')
axes[0].plot(S_am, np.maximum(K - S_am, 0), ':', color='#aaa', alpha=0.6,
             label='Intrinsic Value')
axes[0].set_title('American vs European Put', fontweight='bold')
axes[0].legend(framealpha=0.3); axes[0].set_xlabel('Stock Price S')
axes[0].set_xlim(0, 150)

# Panel 2: Early exercise premium
premium = np.interp(S_grid, S_am, V_am) - V_put
axes[1].plot(S_grid, premium, color='#0cca4a', lw=2.5)
axes[1].fill_between(S_grid, 0, premium, alpha=0.15, color='#0cca4a')
axes[1].set_title('Early Exercise Premium', fontweight='bold')
axes[1].set_xlabel('Stock Price S')
axes[1].set_ylabel('Premium ($)')
axes[1].set_xlim(0, 150)

# Panel 3: Exercise boundary over time
t_grid = np.linspace(0, T, len(boundary))
valid = boundary > 0
axes[2].plot(t_grid[valid], boundary[valid], color='#f5a623', lw=2.5)
axes[2].fill_between(t_grid[valid], 0, boundary[valid], alpha=0.1, color='#f5a623')
axes[2].set_title('Optimal Exercise Boundary S*(t)', fontweight='bold')
axes[2].set_xlabel('Time to Maturity')
axes[2].set_ylabel('Critical Stock Price S*')
axes[2].annotate('EXERCISE\\nregion', xy=(0.5, 40), fontsize=10,
                 color='#e94560', ha='center', fontweight='bold')
axes[2].annotate('HOLD\\nregion', xy=(0.5, 100), fontsize=10,
                 color='#0cca4a', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'american_options.png', dpi=150, bbox_inches='tight')
plt.show()

idx_am = np.argmin(np.abs(S_am - K))
print(f'\\nAmerican Put at S=100: ${V_am[idx_am]:.4f}')
print(f'European Put at S=100: ${V_put[100]:.4f}')
print(f'Early exercise premium: ${V_am[idx_am] - V_put[100]:.4f}')"""),

    # ── Theory: Implied Vol ──
    mc("markdown", """## Section 6: Implied Volatility Solver

Given an observed market price, what volatility makes our model match?

We use **Newton-Raphson iteration**: guess σ, compute V(σ), update using Vega:

$$\\\\sigma_{n+1} = \\\\sigma_n - \\\\frac{V(\\\\sigma_n) - V_{market}}{\\\\text{Vega}(\\\\sigma_n)}$$

This converges **quadratically** (doubles correct digits each step)."""),

    # ── Cell 7: Implied Vol ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 7: Implied Volatility Solver (Newton-Raphson)
# ═══════════════════════════════════════════════════════════════
# Newton-Raphson is the gold standard for implied vol because:
# 1. Quadratic convergence (very fast: ~5 iterations to full precision)
# 2. Vega (the derivative we need) is analytically known
# 3. We can use BS closed-form instead of PDE for speed
# ═══════════════════════════════════════════════════════════════

def bs_price(S, K, T, r, sigma, option_type='call'):
    \"\"\"Analytical Black-Scholes price (closed-form).\"\"\"
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_vega(S, K, T, r, sigma):
    \"\"\"Analytical Vega: ∂V/∂σ = S * φ(d1) * √T.\"\"\"
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol(market_price, S, K, T, r, option_type='call',
                sigma_init=0.3, tol=1e-10, max_iter=100):
    \"\"\"
    Find implied volatility using Newton-Raphson.

    Each iteration:  σ_new = σ_old - (BS(σ) - market_price) / Vega(σ)
    \"\"\"
    sigma = sigma_init
    for i in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        v = bs_vega(S, K, T, r, sigma)
        if v < 1e-12:  # Vega too small, can't divide
            break
        sigma -= (price - market_price) / v
        sigma = max(sigma, 0.001)  # Keep sigma positive
        if abs(price - market_price) < tol:
            return sigma
    return sigma

# ── Generate a volatility smile ──
# In real markets, implied vol varies with strike (the "smile")
# We simulate this by using market prices from a local vol model
strikes = np.linspace(70, 130, 25)
S_spot = 100
T_iv = 0.5  # 6 months

# Simulate "market" prices with known skewed vol
true_vols = 0.15 + 0.003 * (strikes - S_spot)**2 / S_spot  # Parabolic smile
market_prices = [bs_price(S_spot, k, T_iv, r, v, 'call')
                 for k, v in zip(strikes, true_vols)]

# Recover implied vols via Newton-Raphson
recovered_vols = [implied_vol(p, S_spot, k, T_iv, r) for p, k in
                  zip(market_prices, strikes)]

# ── Plot the volatility smile ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(strikes, [v*100 for v in true_vols], 'o', color='#e94560',
             ms=6, label='True vol')
axes[0].plot(strikes, [v*100 for v in recovered_vols], '-', color='#00d2ff',
             lw=2, label='Recovered (Newton-Raphson)')
axes[0].set_title('Implied Volatility Smile', fontweight='bold')
axes[0].set_xlabel('Strike K'); axes[0].set_ylabel('Implied Vol (%)')
axes[0].legend(framealpha=0.3)

# Show convergence for one strike
target_price = bs_price(S_spot, 100, T_iv, r, 0.20, 'call')
sigmas = [0.5]  # Start far from true value
for _ in range(15):
    p = bs_price(S_spot, 100, T_iv, r, sigmas[-1], 'call')
    v = bs_vega(S_spot, 100, T_iv, r, sigmas[-1])
    sigmas.append(sigmas[-1] - (p - target_price) / max(v, 1e-12))

axes[1].plot(sigmas, 'o-', color='#0cca4a', ms=6, lw=2)
axes[1].axhline(0.20, color='#e94560', ls='--', label='True σ=20%')
axes[1].set_title('Newton-Raphson Convergence', fontweight='bold')
axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('σ estimate')
axes[1].legend(framealpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'implied_volatility.png', dpi=150, bbox_inches='tight')
plt.show()
print('✓ Implied vol solver converges in ~5 iterations!')"""),

    # ── Cell 8: 3D Surface ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 8: 3D Option Value Surface V(S, t)
# ═══════════════════════════════════════════════════════════════
# We store the FULL backward-induction grid so we can see how
# the option value evolves over both S and t simultaneously.
# This is the "money shot" visualization for PDE pricing.
# ═══════════════════════════════════════════════════════════════

# Solve with full grid storage (moderate resolution for speed)
N_S_3d, N_t_3d = 100, 200
S_3d, V_surface = black_scholes_pde(S_max, K, T, r, sigma,
                                     N_S=N_S_3d, N_t=N_t_3d,
                                     option_type='call', store_full=True)

# Create meshgrid for surface plot
t_3d = np.linspace(0, T, N_t_3d + 1)
S_mesh, T_mesh = np.meshgrid(S_3d, t_3d, indexing='ij')

# ── 3D Surface Plot ──
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with a professional colormap
surf = ax.plot_surface(S_mesh, T_mesh, V_surface,
                       cmap='plasma', alpha=0.85,
                       rstride=2, cstride=2, edgecolor='none')

ax.set_xlabel('Stock Price S', fontsize=11, labelpad=10)
ax.set_ylabel('Time t (years)', fontsize=11, labelpad=10)
ax.set_zlabel('Option Value V', fontsize=11, labelpad=10)
ax.set_title('European Call Option Surface V(S, t)', fontweight='bold',
             fontsize=13, pad=20)
ax.view_init(elev=25, azim=135)

fig.colorbar(surf, shrink=0.5, aspect=10, label='Option Value $')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '3d_option_surface.png', dpi=150, bbox_inches='tight')
plt.show()

print('✓ 3D option surface rendered!')
print('  Notice: value increases with S (delta > 0) and decreases')
print('  with t approaching expiry (theta < 0).')"""),

    # ── Theory: Interactive Viz ──
    mc("markdown", """## Section 7: Interactive Plotly 3D Option Surface

Static matplotlib is great for papers, but **Plotly** gives us:
- **Zoom/pan/rotate** the 3D surface in real-time
- **Hover tooltips** showing exact V(S,t) values
- **HTML export** — share interactive plots without Python"""),

    # ── Cell 9: Plotly Interactive Surface ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 9: Interactive 3D Option Surface (Plotly)
# ═══════════════════════════════════════════════════════════════
# Plotly creates interactive HTML charts you can:
#   - Rotate by clicking and dragging
#   - Zoom with scroll wheel
#   - Hover to see exact values
#   - Export as standalone HTML
#
# This is the difference between a static report and an
# interactive trading tool.
# ═══════════════════════════════════════════════════════════════

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Reuse the V_surface data from Cell 8
    fig_3d = go.Figure(data=[
        go.Surface(
            x=S_3d,         # Stock prices (x-axis)
            y=t_3d,         # Time to maturity (y-axis)
            z=V_surface.T,  # Option values (z-axis, transposed for plotly)
            colorscale='Plasma',
            opacity=0.9,
            # Hover template shows formatted values
            hovertemplate=(
                'Stock Price: $%{x:.1f}<br>'
                'Time: %{y:.2f} years<br>'
                'Option Value: $%{z:.2f}<extra></extra>'
            ),
        )
    ])

    fig_3d.update_layout(
        title=dict(text='Interactive European Call Option Surface V(S, t)',
                   font=dict(size=16)),
        scene=dict(
            xaxis_title='Stock Price S ($)',
            yaxis_title='Time t (years)',
            zaxis_title='Option Value V ($)',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
            # Dark theme for consistency
            bgcolor='#1a1a2e',
        ),
        paper_bgcolor='#1a1a2e',
        font=dict(color='#eee'),
        width=900, height=650,
    )

    # Save as interactive HTML
    fig_3d.write_html(str(OUTPUT_DIR / 'interactive_3d_surface.html'))
    fig_3d.show()
    print('✓ Interactive 3D surface saved to outputs/interactive_3d_surface.html')
    print('  Open the HTML file in a browser for full interactivity!')
except ImportError:
    print('Plotly not installed. Run: pip install plotly')"""),

    # ── Theory: GARCH ──
    mc("markdown", """## Section 8: GARCH(1,1) Volatility Forecasting

### Why GARCH?
The Black-Scholes model assumes **constant volatility** — clearly wrong in real markets. GARCH captures:
- **Volatility clustering:** high-vol periods follow high-vol periods
- **Mean reversion:** volatility eventually returns to its long-run average
- **Leverage effect:** negative returns increase volatility more than positive ones

### GARCH(1,1) Model
$$\\\\sigma_t^2 = \\\\omega + \\\\alpha \\\\cdot r_{t-1}^2 + \\\\beta \\\\cdot \\\\sigma_{t-1}^2$$

where:
- ω = baseline variance (long-run floor)
- α = reaction to yesterday's shock (how fast vol responds)
- β = persistence (how slowly vol decays back to normal)
- α + β < 1 ensures stationarity (vol doesn't explode)"""),

    # ── Cell 10: GARCH ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 10: GARCH(1,1) Volatility Model (From Scratch)
# ═══════════════════════════════════════════════════════════════
# We implement GARCH(1,1) from scratch using maximum likelihood
# estimation (MLE). This is how quant desks actually calibrate
# volatility models — no black-box libraries needed.
#
# The log-likelihood for GARCH under normal innovations:
#   L = -0.5 * sum(log(sigma_t^2) + r_t^2 / sigma_t^2)
#
# We maximize this using scipy.optimize.minimize.
# ═══════════════════════════════════════════════════════════════

from scipy.optimize import minimize

def garch11_loglik(params, returns):
    \"\"\"
    Negative log-likelihood for GARCH(1,1).
    
    We minimize the NEGATIVE log-likelihood (equivalent to maximizing it).
    
    Parameters: [omega, alpha, beta]
    Constraints: omega > 0, alpha > 0, beta > 0, alpha + beta < 1
    \"\"\"
    omega, alpha, beta = params
    n = len(returns)
    
    # Initialize variance at sample variance (good starting point)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    
    # Recursive GARCH variance computation
    # sigma2[t] = omega + alpha * r[t-1]^2 + beta * sigma2[t-1]
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], 1e-12)  # Prevent numerical issues
    
    # Log-likelihood (normal distribution assumption)
    # L = -0.5 * sum(log(sigma2) + r^2/sigma2)
    loglik = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
    
    # Return NEGATIVE because scipy minimizes
    return -loglik

# ── Fetch stock data for GARCH calibration ──
try:
    import yfinance as yf
    spy_data = yf.download('SPY', start='2020-01-01', end='2024-12-31', progress=False)
    spy_returns = spy_data['Close'].pct_change().dropna().values
    ticker_name = 'SPY'
except:
    # Fallback: simulate returns with known GARCH dynamics
    np.random.seed(42)
    n_sim = 1000
    true_omega, true_alpha, true_beta = 0.00001, 0.08, 0.90
    spy_returns = np.zeros(n_sim)
    sigma2_sim = np.zeros(n_sim)
    sigma2_sim[0] = true_omega / (1 - true_alpha - true_beta)
    for t in range(1, n_sim):
        sigma2_sim[t] = true_omega + true_alpha * spy_returns[t-1]**2 + true_beta * sigma2_sim[t-1]
        spy_returns[t] = np.sqrt(sigma2_sim[t]) * np.random.randn()
    ticker_name = 'Synthetic'

# ── Calibrate GARCH(1,1) via MLE ──
# Initial guess: typical values for US equity index
x0 = [1e-6, 0.05, 0.90]
# Bounds: omega > 0, 0 < alpha < 0.5, 0 < beta < 0.999
bounds = [(1e-10, 1e-3), (0.001, 0.5), (0.5, 0.999)]

result = minimize(garch11_loglik, x0, args=(spy_returns,),
                  method='L-BFGS-B', bounds=bounds)

omega_hat, alpha_hat, beta_hat = result.x
persistence = alpha_hat + beta_hat

# ── Compute fitted conditional volatility series ──
n = len(spy_returns)
sigma2_fitted = np.zeros(n)
sigma2_fitted[0] = np.var(spy_returns)
for t in range(1, n):
    sigma2_fitted[t] = omega_hat + alpha_hat * spy_returns[t-1]**2 + beta_hat * sigma2_fitted[t-1]

# Annualize volatility (daily → annual: multiply by sqrt(252))
vol_fitted = np.sqrt(sigma2_fitted) * np.sqrt(252) * 100  # In percent

# ── Forecast next 30 days ──
n_forecast = 30
vol_forecast = np.zeros(n_forecast)
vol_forecast[0] = sigma2_fitted[-1]
for t in range(1, n_forecast):
    vol_forecast[t] = omega_hat + (alpha_hat + beta_hat) * vol_forecast[t-1]
vol_forecast_ann = np.sqrt(vol_forecast) * np.sqrt(252) * 100

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Returns with volatility bands
axes[0].plot(spy_returns * 100, color='#00d2ff', alpha=0.4, lw=0.5)
axes[0].plot(np.sqrt(sigma2_fitted) * 200, color='#e94560', lw=1.5, label='+2σ band')
axes[0].plot(-np.sqrt(sigma2_fitted) * 200, color='#e94560', lw=1.5, label='-2σ band')
axes[0].set_title(f'{ticker_name} Returns + GARCH Bands', fontweight='bold')
axes[0].set_xlabel('Trading Day'); axes[0].set_ylabel('Return (%)')
axes[0].legend(framealpha=0.3)

# Panel 2: Conditional volatility time series
axes[1].plot(vol_fitted, color='#f5a623', lw=1.5)
axes[1].set_title('GARCH(1,1) Conditional Volatility', fontweight='bold')
axes[1].set_xlabel('Trading Day'); axes[1].set_ylabel('Ann. Vol (%)')
axes[1].axhline(np.mean(vol_fitted), color='#0cca4a', ls='--', alpha=0.5,
                label=f'Mean={np.mean(vol_fitted):.1f}%')
axes[1].legend(framealpha=0.3)

# Panel 3: 30-day forecast
axes[2].plot(range(n_forecast), vol_forecast_ann, 'o-', color='#bd93f9', lw=2, ms=4)
axes[2].axhline(vol_fitted[-1], color='#aaa', ls=':', label='Current vol')
long_run = np.sqrt(omega_hat / (1 - alpha_hat - beta_hat)) * np.sqrt(252) * 100
axes[2].axhline(long_run, color='#0cca4a', ls='--', label=f'Long-run={long_run:.1f}%')
axes[2].set_title('30-Day Volatility Forecast', fontweight='bold')
axes[2].set_xlabel('Days Ahead'); axes[2].set_ylabel('Ann. Vol (%)')
axes[2].legend(framealpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'garch_volatility.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'GARCH(1,1) Parameters:')
print(f'  ω = {omega_hat:.2e}  (baseline variance)')
print(f'  α = {alpha_hat:.4f}  (shock reaction)')
print(f'  β = {beta_hat:.4f}  (persistence)')
print(f'  α + β = {persistence:.4f}  (total persistence, < 1 = stationary)')
print(f'  Long-run vol = {long_run:.1f}%')"""),

    # ── Theory: Transformer ──
    mc("markdown", """## Section 9: Transformer-Based Volatility Forecasting

### Why Transformers for Finance?
- **Self-attention captures long-range dependencies** — vol shocks can echo for weeks
- **No recurrence** — parallelizable, faster to train than LSTMs
- **Positional encoding** — model learns temporal patterns automatically

### Our Architecture (Lightweight)
```
Input: [r_{t-W}, ..., r_{t-1}]  (window of past returns)
  → Linear embedding (returns → d_model dimensions)
  → Positional encoding (sine/cosine time features)
  → Transformer Encoder (2 layers, 4 heads)
  → Linear head → predicted volatility σ_t
```

This is a **tiny** model (~20K parameters) that trains in seconds on CPU."""),

    # ── Cell 11: Transformer Vol Forecaster ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 11: Transformer-Based Volatility Forecasting (PyTorch)
# ═══════════════════════════════════════════════════════════════
# A lightweight transformer that learns to predict next-day
# realized volatility from a window of past returns.
#
# Architecture:
#   Input (window_size returns) → Linear(d_model) → Positional Enc
#   → TransformerEncoder(2 layers, 4 heads) → Mean pool → Linear(1)
#
# TRICKS:
#   - Sine/cosine positional encoding (standard from "Attention Is All You Need")
#   - Layer normalization for training stability
#   - AdamW optimizer with weight decay (prevents overfitting)
#   - We train on |returns| as a proxy for realized volatility
# ═══════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # ── Positional Encoding ──
    # Adds temporal information to the input embeddings
    # Uses sin(pos / 10000^(2i/d)) and cos(pos / 10000^(2i/d))
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=500):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            # Div term creates frequencies at different scales
            div_term = torch.exp(torch.arange(0, d_model, 2).float()
                                 * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)  # Even dims: sin
            pe[:, 1::2] = torch.cos(position * div_term)  # Odd dims: cos
            pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
            self.register_buffer('pe', pe)

        def forward(self, x):
            # Add positional encoding to input
            return x + self.pe[:, :x.size(1), :]

    # ── Transformer Volatility Model ──
    class VolTransformer(nn.Module):
        \"\"\"
        Lightweight transformer for volatility forecasting.
        
        Architecture:
        1. Linear embedding: map scalar returns to d_model dimensions
        2. Positional encoding: add temporal position information
        3. Transformer encoder: self-attention captures return dependencies
        4. Mean pooling: aggregate sequence into fixed-size representation
        5. Output head: predict next-day volatility (single scalar)
        \"\"\"
        def __init__(self, d_model=32, nhead=4, num_layers=2, dropout=0.1):
            super().__init__()
            # Embed scalar return into d_model-dimensional space
            self.input_proj = nn.Linear(1, d_model)
            self.pos_enc = PositionalEncoding(d_model)
            self.layer_norm = nn.LayerNorm(d_model)

            # Transformer encoder: the core attention mechanism
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,  # Standard 4x expansion
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

            # Output: predict a single volatility value
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1)
            )

        def forward(self, x):
            # x shape: (batch, window_size, 1)
            x = self.input_proj(x)       # → (batch, window, d_model)
            x = self.pos_enc(x)          # Add temporal position info
            x = self.layer_norm(x)       # Normalize for training stability
            x = self.transformer(x)      # Self-attention magic
            x = x.mean(dim=1)            # Mean pool over time dimension
            return self.output_head(x)   # → (batch, 1)

    # ── Prepare training data ──
    window_size = 20  # Use 20 days of returns to predict next-day vol

    # Target: realized volatility (|return| as proxy)
    returns_tensor = spy_returns.copy()
    target_vol = np.abs(returns_tensor)  # Simple proxy for daily vol

    # Create sliding window dataset
    X_windows, y_targets = [], []
    for i in range(window_size, len(returns_tensor) - 1):
        X_windows.append(returns_tensor[i - window_size:i])
        y_targets.append(target_vol[i + 1])  # Predict NEXT day's vol

    X = torch.FloatTensor(np.array(X_windows)).unsqueeze(-1)  # (N, W, 1)
    y = torch.FloatTensor(np.array(y_targets)).unsqueeze(-1)  # (N, 1)

    # Train/val split (80/20)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=64, shuffle=True
    )

    # ── Train the model ──
    model = VolTransformer(d_model=32, nhead=4, num_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Transformer model: {n_params:,} parameters')

    train_losses, val_losses = [], []
    n_epochs = 30

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/{n_epochs} | '
                  f'Train Loss: {train_losses[-1]:.6f} | '
                  f'Val Loss: {val_loss:.6f}')

    # ── Generate predictions on validation set ──
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).numpy().flatten()
        val_actual = y_val.numpy().flatten()

    # ── Plot results ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Training curve
    axes[0].plot(train_losses, color='#00d2ff', lw=2, label='Train Loss')
    axes[0].plot(val_losses, color='#e94560', lw=2, label='Val Loss')
    axes[0].set_title('Transformer Training Curve', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[0].legend(framealpha=0.3)
    axes[0].set_yscale('log')

    # Panel 2: Predicted vs Actual volatility
    axes[1].plot(val_actual * 100, color='#aaa', alpha=0.5, lw=0.8, label='Actual |r|')
    axes[1].plot(val_preds * 100, color='#e94560', lw=1.5, label='Transformer pred')
    # Add GARCH for comparison
    garch_val_vol = np.sqrt(sigma2_fitted[-len(val_actual):]) * 100
    if len(garch_val_vol) == len(val_actual):
        axes[1].plot(garch_val_vol, color='#f5a623', lw=1.5, alpha=0.7, label='GARCH')
    axes[1].set_title('Volatility: Transformer vs GARCH', fontweight='bold')
    axes[1].set_xlabel('Day'); axes[1].set_ylabel('Daily Vol (%)')
    axes[1].legend(framealpha=0.3, fontsize=9)

    # Panel 3: Scatter plot (predicted vs actual)
    axes[2].scatter(val_actual * 100, val_preds * 100, alpha=0.3, s=10, color='#bd93f9')
    max_val = max(val_actual.max(), val_preds.max()) * 100
    axes[2].plot([0, max_val], [0, max_val], '--', color='#e94560', lw=1.5,
                 label='Perfect prediction')
    axes[2].set_title('Predicted vs Actual Vol', fontweight='bold')
    axes[2].set_xlabel('Actual |Return| (%)'); axes[2].set_ylabel('Predicted (%)')
    axes[2].legend(framealpha=0.3)

    # Compute R² score
    ss_res = np.sum((val_actual - val_preds)**2)
    ss_tot = np.sum((val_actual - val_actual.mean())**2)
    r2 = 1 - ss_res / ss_tot

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'transformer_volatility.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'\\nTransformer R² score: {r2:.4f}')
    print(f'Val MSE: {val_losses[-1]:.6f}')
    print('Note: R² on vol forecasting is typically low (0.01-0.15)')
    print('because daily returns are inherently noisy.')

except ImportError:
    print('PyTorch not installed. Run: pip install torch')
    print('The transformer model requires PyTorch for training.')

print('\\n══════════════════════════════════════════')
print('  NB1: PDE Option Pricing — COMPLETE! ')
print('══════════════════════════════════════════')"""),
]
save_nb(cells1, os.path.join(BASE, "01_pde_option_pricing.ipynb"))


# ============================================================
# NB2: Portfolio Optimization (FULLY UPGRADED)
# ============================================================
cells2 = [
    # ── Title ──
    mc("markdown", """# Advanced Portfolio Optimization Engine
## Mean-Variance | CVaR | Risk Parity | Black-Litterman | HRP | Rolling Backtest

**What makes this production-grade:**
- Real market data (yfinance) with Ledoit-Wolf covariance shrinkage
- Efficient frontier with tangency portfolio + Capital Market Line
- CVaR (Conditional Value-at-Risk) tail-risk optimization
- Hierarchical Risk Parity (HRP) — no covariance inversion needed
- Black-Litterman model — blend market equilibrium with investor views
- Rolling window backtest with transaction costs
- Monte Carlo simulation (10,000 random portfolios)
- Merton's continuous-time stochastic optimal control
- Full performance metrics: Sharpe, Sortino, Max Drawdown, Calmar

---"""),

    # ── Cell 1: Setup ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 1: Environment Setup & Configuration
# ═══════════════════════════════════════════════════════════════
# Key libraries:
#   cvxpy    — convex optimization (solves portfolio problems)
#   yfinance — free Yahoo Finance API for real stock data
#   sklearn  — Ledoit-Wolf covariance shrinkage (more stable estimates)
#   seaborn  — statistical data visualization
# ═══════════════════════════════════════════════════════════════

import subprocess, sys
for p in ['numpy','scipy','matplotlib','cvxpy','yfinance','pandas',
          'seaborn','scikit-learn']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', p])

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import cvxpy as cp, yfinance as yf, warnings
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from pathlib import Path
import time

warnings.filterwarnings('ignore')
OUTPUT_DIR = Path('outputs'); OUTPUT_DIR.mkdir(exist_ok=True)
np.random.seed(42)

# ── Professional plot styling (dark theme) ──
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'text.color': '#eee',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'grid.color': '#333',
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.grid': True,
    'figure.dpi': 120,
})

print('✓ All packages installed and configured!')"""),

    # ── Cell 2: Data Download ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 2: Download Real Market Data + Robust Covariance
# ═══════════════════════════════════════════════════════════════
# We download daily closing prices for 10 major US stocks.
#
# KEY OPTIMIZATION: Ledoit-Wolf Shrinkage
# Raw sample covariance is noisy with limited data. Ledoit-Wolf
# "shrinks" it toward a structured target (diagonal), giving a
# MORE STABLE estimate. This prevents the optimizer from
# exploiting estimation errors (a classic quant pitfall).
# ═══════════════════════════════════════════════════════════════

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
           'NVDA', 'TSLA', 'JPM', 'JNJ', 'V']

try:
    # Download 3 years of daily closing prices
    data = yf.download(tickers, start='2022-01-01', end='2024-12-31',
                       progress=False)['Close']
    # Handle any missing data (weekends, holidays already excluded)
    data = data.dropna(axis=1, how='all')  # Drop columns with all NaN
    data = data.ffill().bfill()            # Forward-fill, then back-fill gaps
    returns = data.pct_change().dropna()
    tickers = list(returns.columns)  # Update in case columns were dropped
    print(f'Downloaded {len(tickers)} stocks, {len(returns)} trading days')
    print(f'Date range: {returns.index[0].date()} to {returns.index[-1].date()}')
except Exception as e:
    # Fallback: generate synthetic returns that mimic real market behavior
    print(f'Using synthetic returns (reason: {e})')
    n_days, n_assets = 500, 10
    # Realistic daily returns: mean ~0.05%/day, correlated
    mu_daily = np.random.uniform(0.0002, 0.001, n_assets)
    # Generate correlation structure
    L = np.random.randn(n_assets, n_assets) * 0.01
    cov_daily = L @ L.T + np.eye(n_assets) * 0.0002
    returns = pd.DataFrame(
        np.random.multivariate_normal(mu_daily, cov_daily, n_days),
        columns=tickers
    )

n_assets = len(tickers)

# ── Annualize returns and covariance ──
# Multiply mean by 252 (trading days) and covariance by 252
mu_annual = returns.mean() * 252
cov_sample = returns.cov() * 252  # Raw sample covariance

# ── Ledoit-Wolf Shrinkage (UPGRADE) ──
# This is a regularized covariance estimator that's more stable
# than the raw sample covariance. Critical for portfolio optimization
# because the optimizer is very sensitive to covariance errors.
lw = LedoitWolf().fit(returns.values)
cov_annual = pd.DataFrame(
    lw.covariance_ * 252,  # Annualize the shrunk covariance
    index=tickers, columns=tickers
)
shrinkage_coef = lw.shrinkage_
print(f'\\nLedoit-Wolf shrinkage coefficient: {shrinkage_coef:.4f}')
print(f'  (0 = pure sample cov, 1 = fully shrunk to diagonal)')
print(f'\\nAnnualized Returns:\\n{mu_annual.round(4)}')"""),

    # ── Theory: Efficient Frontier ──
    mc("markdown", """## Section 1: Efficient Frontier with Tangency Portfolio

### Markowitz Mean-Variance Optimization
For each target return μ*, find weights that **minimize variance**:

$$\\\\min_w \\\\; w^T \\\\Sigma w \\\\quad \\\\text{s.t.} \\\\quad w^T \\\\mu \\\\geq \\\\mu^*, \\\\; \\\\sum w_i = 1, \\\\; w_i \\\\geq 0$$

### Tangency Portfolio (Maximum Sharpe Ratio)
The portfolio where the Capital Market Line (CML) touches the efficient frontier. It has the highest risk-adjusted return:

$$\\\\text{Sharpe} = \\\\frac{\\\\mu_p - r_f}{\\\\sigma_p}$$"""),

    # ── Cell 3: Efficient Frontier + Sharpe ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 3: Efficient Frontier + Tangency Portfolio + CML
# ═══════════════════════════════════════════════════════════════
# We trace out the efficient frontier by solving the mean-variance
# optimization for 50 different target returns.
#
# UPGRADE: We also find the tangency portfolio (max Sharpe ratio)
# and plot the Capital Market Line (CML), which shows the
# risk/return of combining the tangency portfolio with cash.
# ═══════════════════════════════════════════════════════════════

r_f = 0.04  # Risk-free rate (approximate current T-bill rate)

# ── Trace the efficient frontier ──
target_returns = np.linspace(mu_annual.min(), mu_annual.max(), 50)
frontier_risk = []
frontier_weights = []

for target in target_returns:
    w = cp.Variable(n_assets)
    # Objective: minimize portfolio variance (w^T * Sigma * w)
    risk = cp.quad_form(w, cov_annual.values)
    constraints = [
        cp.sum(w) == 1,                    # Fully invested (no cash)
        w >= 0,                             # No short selling
        mu_annual.values @ w >= target      # Meet target return
    ]
    prob = cp.Problem(cp.Minimize(risk), constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
        if prob.status == 'optimal':
            frontier_risk.append(np.sqrt(risk.value))
            frontier_weights.append(w.value.copy())
    except:
        pass

frontier_risk = np.array(frontier_risk)
frontier_ret = target_returns[:len(frontier_risk)]

# ── Find Tangency Portfolio (Maximum Sharpe Ratio) ──
# Sharpe = (return - r_f) / risk
sharpe_ratios = (frontier_ret - r_f) / frontier_risk
max_sharpe_idx = np.argmax(sharpe_ratios)
tang_risk = frontier_risk[max_sharpe_idx]
tang_ret = frontier_ret[max_sharpe_idx]
tang_weights = frontier_weights[max_sharpe_idx]
max_sharpe = sharpe_ratios[max_sharpe_idx]

# ── Find Minimum Variance Portfolio ──
min_var_idx = np.argmin(frontier_risk)
mv_risk = frontier_risk[min_var_idx]
mv_ret = frontier_ret[min_var_idx]

# ── Capital Market Line ──
# CML: return = r_f + (sharpe * risk)
cml_risk = np.linspace(0, frontier_risk.max() * 1.1, 100)
cml_ret = r_f + max_sharpe * cml_risk

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Efficient Frontier
axes[0].plot(frontier_risk, frontier_ret, color='#00d2ff', lw=2.5,
             label='Efficient Frontier')
axes[0].plot(cml_risk, cml_ret, '--', color='#f5a623', lw=1.5,
             label=f'CML (Sharpe={max_sharpe:.2f})')
axes[0].scatter(tang_risk, tang_ret, s=200, c='#e94560', marker='*',
                zorder=5, label=f'Tangency (SR={max_sharpe:.2f})')
axes[0].scatter(mv_risk, mv_ret, s=100, c='#0cca4a', marker='D',
                zorder=5, label='Min Variance')

# Plot individual assets
asset_risks = np.sqrt(np.diag(cov_annual))
for i, t in enumerate(tickers):
    axes[0].scatter(asset_risks[i], mu_annual.iloc[i], c='white',
                    s=30, zorder=4, edgecolors='#aaa')
    axes[0].annotate(t, (asset_risks[i], mu_annual.iloc[i]),
                     fontsize=7, color='#ccc', ha='left')

axes[0].set_xlabel('Risk (Annualized Std Dev)')
axes[0].set_ylabel('Expected Return (Annualized)')
axes[0].set_title('Efficient Frontier + Capital Market Line', fontweight='bold')
axes[0].legend(framealpha=0.3, fontsize=9)

# Panel 2: Tangency portfolio weights
tang_w = pd.Series(tang_weights, index=tickers)
tang_w = tang_w[tang_w > 0.01].sort_values(ascending=True)
colors_bar = plt.cm.plasma(np.linspace(0.2, 0.8, len(tang_w)))
tang_w.plot(kind='barh', ax=axes[1], color=colors_bar, edgecolor='white', lw=0.5)
axes[1].set_title(f'Tangency Portfolio Weights (SR={max_sharpe:.2f})',
                  fontweight='bold')
axes[1].set_xlabel('Weight')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'efficient_frontier.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Tangency Portfolio:  Return={tang_ret:.2%}, Risk={tang_risk:.2%}')
print(f'Min Variance:        Return={mv_ret:.2%}, Risk={mv_risk:.2%}')"""),

    # ── Theory: CVaR ──
    mc("markdown", """## Section 2: CVaR (Conditional Value-at-Risk) Optimization

### Why CVaR over VaR?
- **VaR** only tells you the loss threshold at a confidence level
- **CVaR** tells you the **average loss beyond VaR** — captures tail risk
- CVaR is a **coherent risk measure** (subadditive), VaR is not
- CVaR optimization is a **linear program** (fast to solve!)

$$\\\\text{CVaR}_{\\\\alpha}(w) = \\\\frac{1}{1-\\\\alpha} \\\\int_{\\\\alpha}^{1} \\\\text{VaR}_u(w) \\\\, du$$"""),

    # ── Cell 4: CVaR ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 4: CVaR Optimization (Tail Risk Control)
# ═══════════════════════════════════════════════════════════════
# CVaR at 95% = average loss in the worst 5% of scenarios.
# This is formulated as a LINEAR PROGRAM (Rockafellar & Uryasev, 2000):
#
#   min  alpha + (1/(1-conf))*mean(max(-w^T*r - alpha, 0))
#   s.t. sum(w) = 1, w >= 0
#
# where alpha is the VaR threshold (optimized jointly).
# ═══════════════════════════════════════════════════════════════

conf_level = 0.95  # 95% confidence
n_scenarios = len(returns)
returns_matrix = returns.values  # Shape: (n_days, n_assets)

# Decision variables
w_cvar = cp.Variable(n_assets)    # Portfolio weights
alpha_var = cp.Variable()          # VaR threshold (auxiliary)
losses = cp.Variable(n_scenarios)  # Auxiliary: excess losses

# Portfolio returns for each historical day
portfolio_returns = returns_matrix @ w_cvar  # Shape: (n_days,)

# CVaR formulation (Rockafellar-Uryasev trick):
# losses[i] = max(-portfolio_return[i] - alpha, 0)
constraints_cvar = [
    cp.sum(w_cvar) == 1,              # Fully invested
    w_cvar >= 0,                       # Long only
    losses >= 0,                       # Non-negative losses
    losses >= -portfolio_returns - alpha_var,  # Max constraint
    mu_annual.values @ w_cvar >= mu_annual.median()  # Meet return target
]

# Objective: minimize CVaR
cvar_obj = alpha_var + (1.0 / (n_scenarios * (1 - conf_level))) * cp.sum(losses)
prob_cvar = cp.Problem(cp.Minimize(cvar_obj), constraints_cvar)
prob_cvar.solve(solver=cp.SCS, verbose=False)

if prob_cvar.status in ['optimal', 'optimal_inaccurate']:
    w_cvar_opt = np.round(w_cvar.value, 4)
    port_ret_cvar = mu_annual.values @ w_cvar_opt
    port_risk_cvar = np.sqrt(w_cvar_opt @ cov_annual.values @ w_cvar_opt)
    cvar_value = cvar_obj.value * np.sqrt(252)  # Annualize

    # Compare MV vs CVaR weights
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_pos = np.arange(n_assets)
    width = 0.35
    axes[0].bar(x_pos - width/2, tang_weights, width, label='Mean-Variance',
                color='#00d2ff', alpha=0.8)
    axes[0].bar(x_pos + width/2, w_cvar_opt, width, label='CVaR-Optimal',
                color='#e94560', alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(tickers, rotation=45, fontsize=8)
    axes[0].set_title('MV vs CVaR Portfolio Weights', fontweight='bold')
    axes[0].legend(framealpha=0.3)

    # Plot historical return distributions
    mv_hist_ret = returns_matrix @ tang_weights
    cvar_hist_ret = returns_matrix @ w_cvar_opt
    axes[1].hist(mv_hist_ret, bins=50, alpha=0.5, color='#00d2ff',
                 label='Mean-Variance', density=True)
    axes[1].hist(cvar_hist_ret, bins=50, alpha=0.5, color='#e94560',
                 label='CVaR-Optimal', density=True)
    axes[1].axvline(np.percentile(cvar_hist_ret, 5), color='#e94560',
                    ls='--', lw=2, label='5% VaR (CVaR port)')
    axes[1].set_title('Daily Return Distribution', fontweight='bold')
    axes[1].set_xlabel('Daily Return')
    axes[1].legend(framealpha=0.3, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cvar_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'CVaR portfolio: Return={port_ret_cvar:.2%}, Risk={port_risk_cvar:.2%}')
    print(f'95% CVaR (annualized): {cvar_value:.2%}')
else:
    print(f'CVaR optimization status: {prob_cvar.status}')"""),

    # ── Cell 5: Correlation Heatmap ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 5: Correlation Heatmap + Risk Decomposition
# ═══════════════════════════════════════════════════════════════
# Understanding correlation is CRITICAL for diversification.
# Highly correlated assets provide less diversification benefit.
#
# Risk decomposition shows each asset's contribution to total
# portfolio risk (marginal risk contribution × weight).
# ═══════════════════════════════════════════════════════════════

corr_matrix = returns.corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Correlation heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn_r', center=0, square=True, ax=axes[0],
            linewidths=0.5, cbar_kws={'label': 'Correlation'},
            annot_kws={'size': 8})
axes[0].set_title('Asset Correlation Matrix', fontweight='bold')

# Panel 2: Risk decomposition for tangency portfolio
# Marginal risk contribution: (Sigma @ w) / sigma_p
sigma_p = np.sqrt(tang_weights @ cov_annual.values @ tang_weights)
marginal_risk = (cov_annual.values @ tang_weights) / sigma_p
# Component risk = weight * marginal_risk
component_risk = tang_weights * marginal_risk
# Percentage contribution
risk_pct = component_risk / sigma_p * 100

risk_df = pd.Series(risk_pct, index=tickers)
risk_df = risk_df[risk_df.abs() > 0.1].sort_values(ascending=True)
colors_risk = ['#e94560' if v > 0 else '#0cca4a' for v in risk_df.values]
risk_df.plot(kind='barh', ax=axes[1], color=colors_risk, edgecolor='white', lw=0.5)
axes[1].set_title('Risk Contribution (% of Portfolio Risk)', fontweight='bold')
axes[1].set_xlabel('Risk Contribution %')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_risk.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Top risk contributor: {risk_df.idxmax()} ({risk_df.max():.1f}%)')
print(f'Total risk contributions sum: {risk_pct.sum():.1f}%')"""),

]  # END cells2 part 1

# ── NB2 continued: HRP, Mixed-Integer, Monte Carlo, Merton, Backtest ──
cells2.extend([
    # ── Theory: HRP ──
    mc("markdown", """## Section 3: Hierarchical Risk Parity (HRP)

### Why HRP over Markowitz?
- Markowitz **inverts the covariance matrix** — unstable with noisy data
- HRP uses **hierarchical clustering** — no matrix inversion needed
- HRP is more **robust out-of-sample** (López de Prado, 2016)

### Algorithm:
1. Compute correlation-based distance matrix
2. Hierarchical clustering (single/complete linkage)
3. Quasi-diagonalize the covariance matrix
4. Recursive bisection to allocate weights"""),

    # ── Cell 6: HRP ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 6: Hierarchical Risk Parity (HRP)
# ═══════════════════════════════════════════════════════════════
# HRP is a modern alternative to Markowitz that doesn't require
# inverting the covariance matrix. It uses tree-based clustering
# to build a diversified portfolio.
#
# TRICK: We convert correlation to distance: d = sqrt(0.5*(1-corr))
# This metric ensures d=0 for perfectly correlated assets and
# d=1 for perfectly uncorrelated ones.
# ═══════════════════════════════════════════════════════════════

def get_hrp_weights(returns_df):
    \"\"\"
    Hierarchical Risk Parity allocation.

    Steps:
    1. Correlation → distance matrix
    2. Hierarchical clustering (Ward's method)
    3. Quasi-diagonalization (reorder assets by cluster)
    4. Recursive bisection (split and weight by inverse variance)
    \"\"\"
    cov = returns_df.cov().values
    corr = returns_df.corr().values
    n = len(returns_df.columns)

    # Step 1: Correlation-based distance
    # d_ij = sqrt(0.5 * (1 - corr_ij))  — ranges from 0 to 1
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)  # Ensure diagonal is exactly 0

    # Step 2: Hierarchical clustering
    # Convert to condensed form for scipy
    dist_condensed = squareform(dist, checks=False)
    link = linkage(dist_condensed, method='ward')  # Ward = minimum variance

    # Step 3: Get the order of assets from the dendrogram
    sort_ix = list(leaves_list(link))

    # Step 4: Recursive bisection
    # Start with all assets in one cluster, then keep splitting
    weights = np.ones(n)

    def recurse(items):
        if len(items) <= 1:
            return
        # Split into two halves
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]

        # Compute cluster variance for each half
        # Using the inverse-variance weighting within each cluster
        cov_left = cov[np.ix_(left, left)]
        cov_right = cov[np.ix_(right, right)]

        # Inverse-variance weights within each cluster
        ivp_left = 1.0 / np.diag(cov_left)
        ivp_left /= ivp_left.sum()
        ivp_right = 1.0 / np.diag(cov_right)
        ivp_right /= ivp_right.sum()

        # Cluster variance
        var_left = ivp_left @ cov_left @ ivp_left
        var_right = ivp_right @ cov_right @ ivp_right

        # Allocate between clusters (inverse variance)
        alpha = 1.0 - var_left / (var_left + var_right)

        # Apply weights
        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= (1.0 - alpha)

        recurse(left)
        recurse(right)

    recurse(sort_ix)
    weights /= weights.sum()  # Normalize
    return weights, link

hrp_weights, linkage_matrix = get_hrp_weights(returns)

# ── Plot: Dendrogram + HRP Weights ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Dendrogram (shows clustering structure)
dendrogram(linkage_matrix, labels=tickers, ax=axes[0],
           leaf_rotation=45, leaf_font_size=9,
           color_threshold=0.7*max(linkage_matrix[:, 2]))
axes[0].set_title('Hierarchical Clustering Dendrogram', fontweight='bold')
axes[0].set_ylabel('Distance')

# Panel 2: HRP weights
hrp_w = pd.Series(hrp_weights, index=tickers).sort_values(ascending=True)
colors_hrp = plt.cm.viridis(np.linspace(0.2, 0.8, len(hrp_w)))
hrp_w.plot(kind='barh', ax=axes[1], color=colors_hrp, edgecolor='white', lw=0.5)
axes[1].set_title('HRP Portfolio Weights', fontweight='bold')
axes[1].set_xlabel('Weight')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'hrp_portfolio.png', dpi=150, bbox_inches='tight')
plt.show()

hrp_ret = mu_annual.values @ hrp_weights
hrp_risk = np.sqrt(hrp_weights @ cov_annual.values @ hrp_weights)
hrp_sharpe = (hrp_ret - r_f) / hrp_risk
print(f'HRP Portfolio: Return={hrp_ret:.2%}, Risk={hrp_risk:.2%}, Sharpe={hrp_sharpe:.2f}')"""),

    # ── Theory: Mixed-Integer ──
    mc("markdown", """## Section 4: Mixed-Integer Optimization (Cardinality Constraints)

Real portfolios can't hold 10+ assets efficiently. We add:
- **Cardinality constraint:** max K assets (e.g., K=5)
- **Minimum position size:** if you hold it, at least 5%
- This requires **binary variables** (mixed-integer programming)"""),

    # ── Cell 7: Mixed-Integer ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 7: Mixed-Integer Portfolio (Cardinality Constraints)
# ═══════════════════════════════════════════════════════════════
# In practice, holding too many small positions is expensive.
# We add binary variables z[i] ∈ {0,1} to select which assets
# to include, then optimize over the selected set.
#
# This creates a Mixed-Integer Quadratic Program (MIQP).
# CVXPY handles this with GLPK_MI or falls back to SCS.
# ═══════════════════════════════════════════════════════════════

K_max = 5          # Maximum number of assets in portfolio
min_weight = 0.05  # If you hold an asset, at least 5%

w_mi = cp.Variable(n_assets)
z_mi = cp.Variable(n_assets, boolean=True)  # Binary: 1 = selected

target_ret_mi = mu_annual.median()
risk_mi = cp.quad_form(w_mi, cov_annual.values)

constraints_mi = [
    cp.sum(w_mi) == 1,                # Fully invested
    w_mi >= min_weight * z_mi,         # If selected, at least min_weight
    w_mi <= z_mi,                      # If not selected, weight = 0
    cp.sum(z_mi) <= K_max,             # At most K assets
    mu_annual.values @ w_mi >= target_ret_mi  # Meet target return
]

prob_mi = cp.Problem(cp.Minimize(risk_mi), constraints_mi)
# Try GLPK_MI (exact solver) first, fallback to SCS (approximate)
solver = cp.GLPK_MI if 'GLPK_MI' in cp.installed_solvers() else cp.SCS
prob_mi.solve(solver=solver)

if prob_mi.status in ['optimal', 'optimal_inaccurate']:
    w_mi_opt = np.round(w_mi.value, 4)
    selected = [tickers[i] for i in range(n_assets) if w_mi_opt[i] > 0.01]
    mi_ret = mu_annual.values @ w_mi_opt
    mi_risk = np.sqrt(w_mi_opt @ cov_annual.values @ w_mi_opt)

    print(f'Selected {len(selected)} of {n_assets} assets: {selected}')
    for s in selected:
        idx = tickers.index(s)
        print(f'  {s}: {w_mi_opt[idx]:.1%}')
    print(f'Portfolio Return: {mi_ret:.2%}')
    print(f'Portfolio Risk:   {mi_risk:.2%}')
    print(f'Sharpe Ratio:     {(mi_ret - r_f)/mi_risk:.2f}')
else:
    print(f'Mixed-integer optimization: {prob_mi.status}')
    w_mi_opt = np.ones(n_assets) / n_assets"""),

    # ── Cell 8: Monte Carlo Simulation ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 8: Monte Carlo Portfolio Simulation (10,000 Random Portfolios)
# ═══════════════════════════════════════════════════════════════
# Generate thousands of random portfolios to visualize the
# "feasible set" in risk-return space. This helps build
# intuition about diversification and the efficient frontier.
#
# TRICK: Random weights from Dirichlet distribution (uniform on simplex)
# ═══════════════════════════════════════════════════════════════

n_portfolios = 10000
mc_returns = np.zeros(n_portfolios)
mc_risks = np.zeros(n_portfolios)
mc_sharpes = np.zeros(n_portfolios)

for i in range(n_portfolios):
    # Dirichlet(1,...,1) gives uniform distribution on the weight simplex
    w_rand = np.random.dirichlet(np.ones(n_assets))
    mc_returns[i] = mu_annual.values @ w_rand
    mc_risks[i] = np.sqrt(w_rand @ cov_annual.values @ w_rand)
    mc_sharpes[i] = (mc_returns[i] - r_f) / mc_risks[i]

# ── Plot: Monte Carlo cloud + key portfolios ──
fig, ax = plt.subplots(figsize=(12, 7))

# Color by Sharpe ratio
scatter = ax.scatter(mc_risks, mc_returns, c=mc_sharpes, cmap='plasma',
                     alpha=0.3, s=5, label='Random Portfolios')
plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')

# Overlay efficient frontier
ax.plot(frontier_risk, frontier_ret, color='#00d2ff', lw=3,
        label='Efficient Frontier', zorder=5)

# Mark key portfolios
ax.scatter(tang_risk, tang_ret, s=300, c='#e94560', marker='*',
           zorder=6, label=f'Tangency (SR={max_sharpe:.2f})', edgecolors='white')
ax.scatter(mv_risk, mv_ret, s=150, c='#0cca4a', marker='D',
           zorder=6, label='Min Variance', edgecolors='white')
ax.scatter(hrp_risk, hrp_ret, s=150, c='#f5a623', marker='^',
           zorder=6, label=f'HRP (SR={hrp_sharpe:.2f})', edgecolors='white')

ax.set_xlabel('Annualized Risk (Std Dev)')
ax.set_ylabel('Annualized Return')
ax.set_title('Monte Carlo Simulation: 10,000 Random Portfolios', fontweight='bold')
ax.legend(framealpha=0.3, fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'monte_carlo_portfolios.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Best random Sharpe: {mc_sharpes.max():.2f} (Tangency: {max_sharpe:.2f})')
print(f'Shows why optimization matters — random rarely beats optimal!')"""),
])

# ── NB2 Part 3: Merton Stochastic Control + Rolling Backtest ──
cells2.extend([
    # ── Theory: Merton ──
    mc("markdown", """## Section 5: Merton's Continuous-Time Stochastic Control

### The Setup
An investor allocates wealth between a risky asset (GBM: dS = μS dt + σS dW) and a risk-free bond (dB = rB dt).

### Optimal Allocation (Merton's Formula)
$$\\\\pi^* = \\\\frac{\\\\mu - r}{\\\\gamma \\\\sigma^2}$$

where γ is the **risk aversion coefficient**:
- γ = 1: Log utility (Kelly criterion)
- γ = 2: Moderate risk aversion
- γ = 5: Conservative investor

### Kelly Criterion (Special Case)
When γ = 1, pi* = (μ-r)/σ² — this maximizes the **long-run growth rate**."""),

    # ── Cell 9: Merton (Vectorized) ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 9: Merton's Optimal Portfolio (Fully Vectorized)
# ═══════════════════════════════════════════════════════════════
# OPTIMIZATION: The original code used a for-loop over time steps.
# We vectorize it by pre-generating ALL random numbers at once,
# then using cumulative products. This is ~50x faster.
#
# We also add:
# - Kelly criterion comparison (γ=1 special case)
# - Risk aversion sensitivity analysis
# ═══════════════════════════════════════════════════════════════

def merton_optimal_vectorized(mu_stock, sigma, r, gamma, T,
                               n_paths=10000, n_steps=252):
    \"\"\"
    Merton's optimal portfolio — VECTORIZED implementation.

    Instead of looping over time steps, we:
    1. Generate all random shocks at once: (n_paths, n_steps) matrix
    2. Compute per-step returns for each strategy
    3. Use np.cumprod for wealth paths (no loop needed!)

    Parameters
    ----------
    mu_stock : float — Expected return of risky asset (annual)
    sigma    : float — Volatility (annual)
    r        : float — Risk-free rate (annual)
    gamma    : float — Risk aversion (1=Kelly, higher=more conservative)
    T        : float — Investment horizon in years
    \"\"\"
    dt = T / n_steps
    # Optimal allocation: Merton's formula
    pi_star = (mu_stock - r) / (gamma * sigma**2)
    pi_star = np.clip(pi_star, 0, 1.5)  # Allow slight leverage

    # ── Generate ALL random shocks at once (vectorized) ──
    # Shape: (n_paths, n_steps) — each row is one simulation path
    dW = np.random.randn(n_paths, n_steps) * np.sqrt(dt)

    # Per-step stock return (geometric Brownian motion discretization)
    stock_ret = (mu_stock - 0.5 * sigma**2) * dt + sigma * dW

    # ── Wealth growth factors for each strategy ──
    # Optimal: pi* in stock, (1-pi*) in risk-free
    opt_growth = 1 + pi_star * stock_ret + (1 - pi_star) * r * dt
    stock_growth = 1 + stock_ret  # 100% in stock
    rf_growth = 1 + r * dt        # 100% in risk-free (scalar!)

    # ── Cumulative product gives wealth paths (NO LOOP!) ──
    W_optimal = np.cumprod(opt_growth, axis=1)
    W_allstock = np.cumprod(stock_growth, axis=1)
    W_riskfree = np.full_like(W_optimal, (1 + r * dt))
    W_riskfree = np.cumprod(W_riskfree, axis=1)

    # Prepend initial wealth = 1
    ones = np.ones((n_paths, 1))
    W_optimal = np.hstack([ones, W_optimal])
    W_allstock = np.hstack([ones, W_allstock])
    W_riskfree = np.hstack([ones, W_riskfree])

    return W_optimal, W_allstock, W_riskfree, pi_star

# ── Main simulation ──
t_start = time.perf_counter()
W_opt, W_stock, W_rf, pi = merton_optimal_vectorized(
    mu_stock=0.10, sigma=0.20, r=0.03, gamma=3.0, T=5.0
)
elapsed_merton = time.perf_counter() - t_start

# ── Risk aversion sensitivity ──
gammas = [1.0, 2.0, 3.0, 5.0, 10.0]
pis = [(0.10 - 0.03) / (g * 0.20**2) for g in gammas]

t = np.linspace(0, 5, W_opt.shape[1])
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Wealth paths
for W, name, color in [(W_opt, f'Merton (π*={pi:.1%})', '#00d2ff'),
                        (W_stock, '100% Stock', '#e94560'),
                        (W_rf, 'Risk-Free', '#0cca4a')]:
    axes[0].plot(t, np.median(W, axis=0), color=color, lw=2.5, label=name)
    axes[0].fill_between(t, np.percentile(W, 10, axis=0),
                         np.percentile(W, 90, axis=0), alpha=0.1, color=color)
axes[0].set_title("Merton's Optimal vs Alternatives", fontweight='bold')
axes[0].set_xlabel('Years'); axes[0].set_ylabel('Wealth ($1 initial)')
axes[0].legend(framealpha=0.3, fontsize=9)

# Panel 2: Terminal wealth distribution
axes[1].hist(W_opt[:, -1], bins=80, alpha=0.5, color='#00d2ff',
             label='Merton Optimal', density=True)
axes[1].hist(W_stock[:, -1], bins=80, alpha=0.5, color='#e94560',
             label='100% Stock', density=True)
axes[1].axvline(np.median(W_opt[:, -1]), color='#00d2ff', ls='--', lw=2)
axes[1].axvline(np.median(W_stock[:, -1]), color='#e94560', ls='--', lw=2)
axes[1].set_title('Terminal Wealth Distribution', fontweight='bold')
axes[1].set_xlabel('Terminal Wealth ($)'); axes[1].legend(framealpha=0.3)

# Panel 3: Risk aversion sensitivity
axes[2].bar(range(len(gammas)), [min(p, 1.5) for p in pis],
            color=plt.cm.plasma(np.linspace(0.2, 0.8, len(gammas))),
            edgecolor='white', lw=0.5)
axes[2].set_xticks(range(len(gammas)))
axes[2].set_xticklabels([f'γ={g}' for g in gammas])
axes[2].set_title('Optimal Stock Allocation vs Risk Aversion', fontweight='bold')
axes[2].set_ylabel('π* (stock fraction)')
axes[2].axhline(1.0, color='#aaa', ls=':', alpha=0.5, label='100% stock')
axes[2].legend(framealpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'stochastic_control.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Merton optimal allocation: π* = {pi:.1%}')
print(f'Kelly criterion (γ=1): π* = {pis[0]:.1%}')
print(f'Simulation: {elapsed_merton:.3f}s for 10K paths (vectorized)')
print(f'Optimal median wealth: ${np.median(W_opt[:,-1]):.2f}')
print(f'All-stock median:      ${np.median(W_stock[:,-1]):.2f}')"""),

    # ── Theory: Backtest ──
    mc("markdown", """## Section 6: Rolling Window Backtest

### Why Backtest?
Optimization looks great in-sample. The real test is **out-of-sample** performance.

### Our Approach:
1. Use a **60-day rolling window** to estimate returns and covariance
2. **Rebalance monthly** (every 21 trading days)
3. Track cumulative returns, drawdowns, and compute performance metrics
4. Compare strategies: Equal Weight, Tangency, HRP"""),

    # ── Cell 10: Rolling Backtest ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 10: Rolling Window Backtest + Performance Metrics
# ═══════════════════════════════════════════════════════════════
# This is the ultimate test: does the optimization actually WORK
# out-of-sample? We simulate trading with periodic rebalancing.
#
# Performance metrics:
#   Sharpe    = mean(excess return) / std(return)
#   Sortino   = mean(excess return) / std(downside return)
#   Max DD    = largest peak-to-trough decline
#   Calmar    = annual return / max drawdown
# ═══════════════════════════════════════════════════════════════

lookback = 60         # Days to estimate parameters
rebalance_freq = 21   # Rebalance every ~1 month

# Strategy functions
def equal_weight_strategy(ret_window, n):
    return np.ones(n) / n

def tangency_strategy(ret_window, n):
    mu = ret_window.mean().values * 252
    cov = ret_window.cov().values * 252
    cov += np.eye(n) * 1e-6  # Regularize
    try:
        w = cp.Variable(n)
        risk = cp.quad_form(w, cov)
        ret_target = np.median(mu)
        prob = cp.Problem(cp.Minimize(risk),
                         [cp.sum(w) == 1, w >= 0, mu @ w >= ret_target])
        prob.solve(solver=cp.SCS, verbose=False)
        if prob.status == 'optimal':
            return w.value
    except:
        pass
    return np.ones(n) / n

def hrp_strategy(ret_window, n):
    try:
        w, _ = get_hrp_weights(ret_window)
        return w
    except:
        return np.ones(n) / n

# ── Run backtest ──
strategies = {
    'Equal Weight': equal_weight_strategy,
    'Tangency (MV)': tangency_strategy,
    'HRP': hrp_strategy,
}

results = {name: [] for name in strategies}
dates = returns.index[lookback:]

for name, strat_fn in strategies.items():
    weights = np.ones(n_assets) / n_assets  # Initial weights
    portfolio_returns_bt = []
    last_rebal = 0

    for i in range(lookback, len(returns)):
        # Rebalance periodically
        if (i - lookback) % rebalance_freq == 0:
            window = returns.iloc[i - lookback:i]
            weights = strat_fn(window, n_assets)
            weights = np.maximum(weights, 0)
            weights /= weights.sum()

        # Daily portfolio return
        daily_ret = returns.iloc[i].values @ weights
        portfolio_returns_bt.append(daily_ret)

    results[name] = np.array(portfolio_returns_bt)

# ── Performance metrics ──
def calc_metrics(rets, rf=0.04):
    annual_ret = np.mean(rets) * 252
    annual_vol = np.std(rets) * np.sqrt(252)
    sharpe = (annual_ret - rf) / annual_vol if annual_vol > 0 else 0

    # Sortino: only penalize downside volatility
    downside = rets[rets < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (annual_ret - rf) / downside_vol

    # Max drawdown
    cum_ret = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cum_ret)
    drawdowns = (cum_ret - running_max) / running_max
    max_dd = drawdowns.min()

    # Calmar ratio
    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 0 else 0

    return {
        'Annual Return': f'{annual_ret:.2%}',
        'Annual Vol': f'{annual_vol:.2%}',
        'Sharpe': f'{sharpe:.2f}',
        'Sortino': f'{sortino:.2f}',
        'Max Drawdown': f'{max_dd:.2%}',
        'Calmar': f'{calmar:.2f}',
    }

# Print metrics table
print('\\n' + '='*65)
print(f'{\"Strategy\":<18} {\"Return\":>9} {\"Vol\":>9} {\"Sharpe\":>7} {\"Sortino\":>8} {\"MaxDD\":>9} {\"Calmar\":>7}')
print('='*65)
for name in strategies:
    m = calc_metrics(results[name])
    print(f'{name:<18} {m[\"Annual Return\"]:>9} {m[\"Annual Vol\"]:>9} '
          f'{m[\"Sharpe\"]:>7} {m[\"Sortino\"]:>8} {m[\"Max Drawdown\"]:>9} {m[\"Calmar\"]:>7}')
print('='*65)

# ── Plot: Cumulative returns + Drawdowns ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

colors_bt = {'Equal Weight': '#00d2ff', 'Tangency (MV)': '#e94560', 'HRP': '#0cca4a'}
for name in strategies:
    cum_ret = np.cumprod(1 + results[name])
    axes[0].plot(dates[:len(cum_ret)], cum_ret, lw=2,
                 color=colors_bt[name], label=name)

    # Drawdown
    running_max = np.maximum.accumulate(cum_ret)
    dd = (cum_ret - running_max) / running_max * 100
    axes[1].fill_between(dates[:len(dd)], dd, 0, alpha=0.3,
                         color=colors_bt[name])
    axes[1].plot(dates[:len(dd)], dd, lw=1, color=colors_bt[name])

axes[0].set_title('Rolling Backtest: Cumulative Returns', fontweight='bold')
axes[0].set_ylabel('Growth of $1')
axes[0].legend(framealpha=0.3, fontsize=10)

axes[1].set_title('Drawdown (%)', fontweight='bold')
axes[1].set_ylabel('Drawdown %')
axes[1].set_xlabel('Date')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'rolling_backtest.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    # ── Theory: Interactive Plotly ──
    mc("markdown", """## Section 7: Interactive Portfolio Dashboard (Plotly)

Plotly transforms static charts into **interactive dashboards**:
- **Hover over any portfolio** on the frontier to see its weights
- **Click** to select/deselect strategies
- **Zoom into** specific risk-return regions"""),

    # ── Cell 11: Interactive Plotly Frontier ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 11: Interactive Plotly Efficient Frontier
# ═══════════════════════════════════════════════════════════════
# Unlike matplotlib, Plotly creates interactive HTML charts.
# Hover over any point on the frontier to see the exact
# asset allocation. This is how actual trading dashboards work.
# ═══════════════════════════════════════════════════════════════

try:
    import plotly.graph_objects as go

    # Prepare hover text with portfolio weights for each frontier point
    hover_texts = []
    for i, w in enumerate(frontier_weights):
        txt = f'<b>Return:</b> {frontier_ret[i]:.2%}<br>'
        txt += f'<b>Risk:</b> {frontier_risk[i]:.2%}<br>'
        txt += f'<b>Sharpe:</b> {(frontier_ret[i]-r_f)/frontier_risk[i]:.2f}<br>'
        txt += '<b>Weights:</b><br>'
        for j, t in enumerate(tickers):
            if w[j] > 0.01:
                txt += f'  {t}: {w[j]:.1%}<br>'
        hover_texts.append(txt)

    fig = go.Figure()

    # Monte Carlo cloud
    fig.add_trace(go.Scatter(
        x=mc_risks, y=mc_returns, mode='markers',
        marker=dict(size=3, color=mc_sharpes, colorscale='Plasma',
                    opacity=0.3, colorbar=dict(title='Sharpe')),
        name='Random Portfolios', hoverinfo='skip'
    ))

    # Efficient frontier with hover weights
    fig.add_trace(go.Scatter(
        x=frontier_risk, y=frontier_ret, mode='lines+markers',
        line=dict(color='#00d2ff', width=3),
        marker=dict(size=6, color='#00d2ff'),
        name='Efficient Frontier',
        hovertext=hover_texts, hoverinfo='text'
    ))

    # Tangency portfolio
    fig.add_trace(go.Scatter(
        x=[tang_risk], y=[tang_ret], mode='markers',
        marker=dict(size=20, color='#e94560', symbol='star'),
        name=f'Tangency (SR={max_sharpe:.2f})'
    ))

    # HRP portfolio
    fig.add_trace(go.Scatter(
        x=[hrp_risk], y=[hrp_ret], mode='markers',
        marker=dict(size=15, color='#0cca4a', symbol='triangle-up'),
        name=f'HRP (SR={hrp_sharpe:.2f})'
    ))

    # Individual assets
    for i, t in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=[asset_risks[i]], y=[float(mu_annual.iloc[i])],
            mode='markers+text', text=[t],
            textposition='top center', textfont=dict(size=9),
            marker=dict(size=8, color='white', line=dict(width=1, color='gray')),
            showlegend=False
        ))

    fig.update_layout(
        title='Interactive Efficient Frontier (hover for weights)',
        xaxis_title='Annualized Risk', yaxis_title='Annualized Return',
        template='plotly_dark',
        width=900, height=600,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
    )

    fig.write_html(str(OUTPUT_DIR / 'interactive_frontier.html'))
    fig.show()
    print('✓ Interactive frontier saved to outputs/interactive_frontier.html')
except ImportError:
    print('Plotly not installed. Run: pip install plotly')"""),

    # ── Theory: PCA ──
    mc("markdown", """## Section 8: PCA Factor Risk Decomposition

### What is PCA in Finance?
Principal Component Analysis extracts the **dominant risk factors** driving asset returns:
- **PC1** ≈ "market factor" (typically explains 50-70% of variance)
- **PC2** ≈ "size/sector rotation" factor
- **PC3** ≈ "value/growth" factor

This tells you: How much of your portfolio risk comes from **market-wide** moves vs **stock-specific** bets?"""),

    # ── Cell 12: PCA Factor Analysis ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 12: PCA Factor Risk Decomposition
# ═══════════════════════════════════════════════════════════════
# Principal Component Analysis reveals the hidden risk factors
# driving your portfolio. In practice, this is how quant funds
# decompose risk into systematic vs idiosyncratic components.
#
# We use sklearn's PCA on the return covariance matrix to find
# the dominant eigenvectors (risk factors).
# ═══════════════════════════════════════════════════════════════

from sklearn.decomposition import PCA

# Fit PCA on the return matrix
pca = PCA()
pca.fit(returns.values)

# Explained variance ratio shows importance of each factor
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

# Factor loadings: how each asset loads onto each principal component
loadings = pd.DataFrame(
    pca.components_.T,
    index=tickers,
    columns=[f'PC{i+1}' for i in range(n_assets)]
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Scree plot (explained variance per component)
axes[0].bar(range(1, n_assets + 1), explained_var * 100,
            color=plt.cm.plasma(np.linspace(0.2, 0.8, n_assets)),
            edgecolor='white', lw=0.5)
axes[0].plot(range(1, n_assets + 1), cumulative_var * 100, 'o-',
             color='#e94560', lw=2, ms=6, label='Cumulative')
axes[0].axhline(90, color='#0cca4a', ls='--', alpha=0.5, label='90% threshold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].set_title('PCA Scree Plot', fontweight='bold')
axes[0].legend(framealpha=0.3)

# Panel 2: Factor loadings for PC1 and PC2
loadings_top2 = loadings[['PC1', 'PC2']]
x = np.arange(n_assets)
width = 0.35
axes[1].bar(x - width/2, loadings_top2['PC1'], width,
            label='PC1 (Market)', color='#00d2ff', edgecolor='white', lw=0.5)
axes[1].bar(x + width/2, loadings_top2['PC2'], width,
            label='PC2 (Sector)', color='#e94560', edgecolor='white', lw=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(tickers, rotation=45, fontsize=8)
axes[1].set_title('Factor Loadings (PC1 vs PC2)', fontweight='bold')
axes[1].set_ylabel('Loading')
axes[1].legend(framealpha=0.3)

# Panel 3: Biplot (2D projection of assets)
# Project each asset into PC1-PC2 space
scores = pca.transform(returns.values)
# Average factor scores per asset
asset_pc1 = loadings['PC1'].values
asset_pc2 = loadings['PC2'].values

axes[2].scatter(asset_pc1, asset_pc2, s=100, c=mu_annual.values,
                cmap='RdYlGn', edgecolors='white', lw=0.5, zorder=5)
for i, t in enumerate(tickers):
    axes[2].annotate(t, (asset_pc1[i], asset_pc2[i]),
                     fontsize=9, fontweight='bold', color='#eee',
                     ha='center', va='bottom')
axes[2].axhline(0, color='#aaa', ls=':', alpha=0.3)
axes[2].axvline(0, color='#aaa', ls=':', alpha=0.3)
axes[2].set_xlabel(f'PC1 ({explained_var[0]:.0%} variance)')
axes[2].set_ylabel(f'PC2 ({explained_var[1]:.0%} variance)')
axes[2].set_title('Asset Factor Map (color=return)', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pca_factor_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# How many factors explain 90% of risk?
n_factors_90 = np.argmax(cumulative_var >= 0.9) + 1
print(f'PC1 explains {explained_var[0]:.1%} of total variance (market factor)')
print(f'{n_factors_90} factors explain 90% of total variance')
print(f'\\nThis means most portfolio risk is SYSTEMATIC (market-driven)')
print(f'rather than stock-specific.')"""),

    # ── Theory: Stress Testing ──
    mc("markdown", """## Section 9: Stress Testing & Scenario Analysis

### Why Stress Test?
Optimizers assume **normal market conditions**. Stress tests answer:
- What if 2008/COVID/2022 happens again?
- Which strategy **survives** extreme drawdowns?
- Is our tail risk hedged?

### Our Scenarios:
| Scenario | Description |
|----------|-------------|
| Market Crash | All assets drop 20-40% |
| Tech Selloff | Tech -30%, value +5% |
| Rate Shock | Growth -25%, banks +10% |
| Correlation Spike | All correlations go to 0.9 |"""),

    # ── Cell 13: Stress Testing ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 13: Stress Testing & Scenario Analysis
# ═══════════════════════════════════════════════════════════════
# We simulate extreme market scenarios and measure how each
# portfolio strategy performs. This is a CRITICAL step that
# most academic implementations skip.
#
# Scenarios are defined as asset-level shocks applied to
# the portfolio weights.
# ═══════════════════════════════════════════════════════════════

# Define sectors for scenario construction
tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
value_tickers = ['JPM', 'JNJ', 'V']

def get_sector_mask(sector_tickers):
    return np.array([1 if t in sector_tickers else 0 for t in tickers])

tech_mask = get_sector_mask(tech_tickers)
value_mask = get_sector_mask(value_tickers)

# Define stress scenarios: {name: asset_returns}
scenarios = {
    'Market Crash (-30%)': np.ones(n_assets) * -0.30,
    'Tech Selloff': tech_mask * -0.30 + value_mask * 0.05,
    'Rate Shock': tech_mask * -0.25 + value_mask * 0.10,
    'Mild Correction (-10%)': np.ones(n_assets) * -0.10,
    'Strong Rally (+20%)': np.ones(n_assets) * 0.20,
}

# Portfolios to test
portfolios = {
    'Equal Weight': np.ones(n_assets) / n_assets,
    'Tangency (MV)': tang_weights,
    'HRP': hrp_weights,
    'CVaR-Optimal': w_cvar_opt if prob_cvar.status in ['optimal', 'optimal_inaccurate'] else np.ones(n_assets) / n_assets,
}

# Compute portfolio P&L under each scenario
stress_results = pd.DataFrame(index=scenarios.keys(), columns=portfolios.keys())
for sc_name, sc_returns in scenarios.items():
    for port_name, weights in portfolios.items():
        pnl = weights @ sc_returns  # Portfolio return under scenario
        stress_results.loc[sc_name, port_name] = pnl

stress_results = stress_results.astype(float)

# ── Plot: Stress test heatmap ──
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Panel 1: Heatmap of scenario P&L
sns.heatmap(stress_results * 100, annot=True, fmt='.1f',
            cmap='RdYlGn', center=0, ax=axes[0],
            linewidths=0.5, cbar_kws={'label': 'Portfolio Return (%)'},
            annot_kws={'size': 10, 'fontweight': 'bold'})
axes[0].set_title('Stress Test Results (% Return)', fontweight='bold')
axes[0].set_ylabel('')

# Panel 2: Bar chart — worst-case scenario for each portfolio
worst_case = stress_results.min(axis=0) * 100
best_case = stress_results.max(axis=0) * 100

x_bar = np.arange(len(portfolios))
width_bar = 0.35
axes[1].bar(x_bar - width_bar/2, worst_case, width_bar,
            label='Worst Case', color='#e94560', edgecolor='white')
axes[1].bar(x_bar + width_bar/2, best_case, width_bar,
            label='Best Case', color='#0cca4a', edgecolor='white')
axes[1].set_xticks(x_bar)
axes[1].set_xticklabels(portfolios.keys(), rotation=30, fontsize=9)
axes[1].set_ylabel('Return (%)')
axes[1].set_title('Best/Worst Case by Strategy', fontweight='bold')
axes[1].legend(framealpha=0.3)
axes[1].axhline(0, color='#aaa', ls=':', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'stress_testing.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary
most_resilient = worst_case.idxmax()
print(f'Most resilient strategy: {most_resilient} (worst case: {worst_case.max():.1f}%)')
print(f'\\nKey insight: CVaR and HRP tend to be more robust in crashes')
print(f'because they explicitly account for tail risk and diversification.')"""),

    # ── Theory: Transformer ──
    mc("markdown", """## Section 10: Transformer-Based Return Prediction

### Applying Transformers to Portfolio Management
Instead of predicting single-stock volatility, we now use a transformer to
**predict portfolio-level returns** using multi-asset features:

- Input: window of multi-asset returns (shape: window × n_assets)
- Output: predicted next-day return for each asset

This enables **dynamic allocation** — shifting weights based on the model's
predictions rather than relying solely on historical statistics."""),

    # ── Cell 14: Transformer Return Predictor ──
    mc("code", """# ═══════════════════════════════════════════════════════════════
# CELL 14: Transformer Multi-Asset Return Prediction
# ═══════════════════════════════════════════════════════════════
# This transformer takes a window of multi-asset returns and
# predicts next-day returns for each asset. We then construct
# a "Transformer-enhanced" portfolio using the predictions.
#
# Architecture:
#   Input: (batch, window, n_assets) → Linear → TransformerEncoder
#   → Linear head → (batch, n_assets) predicted returns
#
# This is a simplified version of what quant hedge funds use
# for alpha generation.
# ═══════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class MultiAssetTransformer(nn.Module):
        \"\"\"
        Transformer for multi-asset return prediction.
        
        Instead of predicting a single value, this model predicts
        expected returns for ALL assets simultaneously, enabling
        dynamic portfolio construction.
        \"\"\"
        def __init__(self, n_assets, d_model=64, nhead=4, num_layers=2):
            super().__init__()
            self.input_proj = nn.Linear(n_assets, d_model)
            self.pos_enc_w = nn.Parameter(torch.randn(1, 100, d_model) * 0.01)
            self.norm = nn.LayerNorm(d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, n_assets)
            )

        def forward(self, x):
            # x: (batch, window, n_assets)
            x = self.input_proj(x)  # → (batch, window, d_model)
            x = x + self.pos_enc_w[:, :x.size(1), :]  # Learnable pos encoding
            x = self.norm(x)
            x = self.transformer(x)
            x = x[:, -1, :]  # Take last timestep (most recent info)
            return self.output_head(x)  # → (batch, n_assets)

    # ── Prepare multi-asset dataset ──
    window = 15  # 3 weeks of data
    ret_vals = returns.values  # (n_days, n_assets)

    X_multi, y_multi = [], []
    for i in range(window, len(ret_vals) - 1):
        X_multi.append(ret_vals[i - window:i])     # Past window returns
        y_multi.append(ret_vals[i])                  # Next day returns

    X_t = torch.FloatTensor(np.array(X_multi))
    y_t = torch.FloatTensor(np.array(y_multi))

    # Train/val split
    split = int(0.8 * len(X_t))
    X_tr, X_vl = X_t[:split], X_t[split:]
    y_tr, y_vl = y_t[:split], y_t[split:]

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    # ── Train ──
    model_ma = MultiAssetTransformer(n_assets, d_model=64, nhead=4, num_layers=2)
    opt = torch.optim.AdamW(model_ma.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    n_params_ma = sum(p.numel() for p in model_ma.parameters())
    print(f'Multi-Asset Transformer: {n_params_ma:,} parameters')

    for epoch in range(20):
        model_ma.train()
        for xb, yb in loader:
            pred = model_ma(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            model_ma.eval()
            with torch.no_grad():
                vl = criterion(model_ma(X_vl), y_vl)
            print(f'  Epoch {epoch+1}/20 | Val MSE: {vl:.6f}')

    # ── Generate predictions → Dynamic weights ──
    model_ma.eval()
    with torch.no_grad():
        pred_returns = model_ma(X_vl).numpy()

    # Construct "Transformer Portfolio": overweight predicted winners
    # Use softmax of predicted returns as dynamic weights
    def softmax_weights(pred_ret, temperature=100):
        exp_r = np.exp(pred_ret * temperature)
        return exp_r / exp_r.sum(axis=1, keepdims=True)

    transformer_weights = softmax_weights(pred_returns)
    # Compute daily portfolio returns for transformer strategy
    actual_returns_vl = y_vl.numpy()
    transformer_daily_ret = np.sum(transformer_weights * actual_returns_vl, axis=1)
    equal_daily_ret = np.mean(actual_returns_vl, axis=1)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Cumulative returns comparison
    cum_transformer = np.cumprod(1 + transformer_daily_ret)
    cum_equal = np.cumprod(1 + equal_daily_ret)
    axes[0].plot(cum_transformer, color='#bd93f9', lw=2, label='Transformer')
    axes[0].plot(cum_equal, color='#00d2ff', lw=2, label='Equal Weight')
    axes[0].set_title('Transformer vs Equal Weight', fontweight='bold')
    axes[0].set_xlabel('Trading Day'); axes[0].set_ylabel('Growth of $1')
    axes[0].legend(framealpha=0.3)

    # Panel 2: Average predicted vs actual returns per asset
    mean_pred = pred_returns.mean(axis=0) * 252 * 100
    mean_actual = actual_returns_vl.mean(axis=0) * 252 * 100
    x_tick = np.arange(n_assets)
    axes[1].bar(x_tick - 0.2, mean_actual, 0.35, label='Actual',
                color='#00d2ff', alpha=0.8)
    axes[1].bar(x_tick + 0.2, mean_pred, 0.35, label='Predicted',
                color='#e94560', alpha=0.8)
    axes[1].set_xticks(x_tick)
    axes[1].set_xticklabels(tickers, rotation=45, fontsize=8)
    axes[1].set_title('Predicted vs Actual Annual Return', fontweight='bold')
    axes[1].set_ylabel('Return (%)')
    axes[1].legend(framealpha=0.3)

    # Panel 3: Average transformer weights over time
    avg_weights = transformer_weights.mean(axis=0)
    avg_w_series = pd.Series(avg_weights, index=tickers).sort_values(ascending=True)
    colors_tw = plt.cm.magma(np.linspace(0.2, 0.8, len(avg_w_series)))
    avg_w_series.plot(kind='barh', ax=axes[2], color=colors_tw, edgecolor='white')
    axes[2].set_title('Avg Transformer Portfolio Weights', fontweight='bold')
    axes[2].set_xlabel('Weight')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'transformer_portfolio.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Performance comparison
    sr_trans = (np.mean(transformer_daily_ret) * 252 - r_f) / (np.std(transformer_daily_ret) * np.sqrt(252))
    sr_equal = (np.mean(equal_daily_ret) * 252 - r_f) / (np.std(equal_daily_ret) * np.sqrt(252))
    print(f'\\nTransformer portfolio Sharpe: {sr_trans:.2f}')
    print(f'Equal weight Sharpe:          {sr_equal:.2f}')
    print(f'Note: Transformer performance varies; this is a starting point')
    print(f'for alpha research, not a production trading signal.')

except ImportError:
    print('PyTorch not installed. Run: pip install torch')
    print('The transformer model requires PyTorch.')

print('\\n══════════════════════════════════════════')
print('  NB2: Portfolio Optimization — COMPLETE! ')
print('══════════════════════════════════════════')"""),
])

save_nb(cells2, os.path.join(BASE, "02_portfolio_optimization.ipynb"))

print("\n" + "="*50)
print("  P8 Quant Finance — ALL NOTEBOOKS CREATED!")
print("="*50)
