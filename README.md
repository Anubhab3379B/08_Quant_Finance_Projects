# Advanced Quant Finance Projects

Production-grade quantitative finance implementations with detailed explanatory comments, interactive visualizations, and ML-powered forecasting.

## Notebooks

### 1. PDE Option Pricing (`01_pde_option_pricing.ipynb`)
- **Crank-Nicolson FDM** — unconditionally stable, O(h²) accurate PDE solver
- **Put-Call Parity Verification** — validates solver accuracy
- **Full Greeks Surface** — Delta, Gamma, Theta, Vega, Rho via FDM + bump-and-revalue
- **Convergence Analysis** — grid convergence study with Richardson extrapolation
- **American Options (PSOR)** — Numba JIT-accelerated Projected SOR with exercise boundary
- **Implied Volatility Solver** — Newton-Raphson with volatility smile visualization
- **3D Option Surface** — V(S, t) rendered as publication-quality 3D plot
- **Interactive Plotly 3D** — rotatable, hoverable HTML export of option surface
- **GARCH(1,1) Volatility** — MLE-calibrated from scratch with 30-day forecast
- **Transformer Vol Forecaster** — PyTorch attention-based volatility prediction

### 2. Portfolio Optimization (`02_portfolio_optimization.ipynb`)
- **Enhanced Data Pipeline** — yfinance + Ledoit-Wolf covariance shrinkage
- **Efficient Frontier** — with tangency portfolio + Capital Market Line
- **CVaR Optimization** — Rockafellar-Uryasev tail-risk minimization
- **Correlation Heatmap** — with risk decomposition analysis
- **Hierarchical Risk Parity (HRP)** — Lopez de Prado's clustering-based allocation
- **Mixed-Integer Optimization** — cardinality constraints, minimum position sizes
- **Monte Carlo Simulation** — 10,000 random portfolios visualization
- **Merton Stochastic Control** — fully vectorized with Kelly criterion comparison
- **Rolling Backtest** — 60-day window, monthly rebalancing, Sharpe/Sortino/MaxDD/Calmar
- **Interactive Plotly Frontier** — hover to see portfolio weights, zoomable
- **PCA Factor Decomposition** — risk factor analysis with scree plot and biplot
- **Stress Testing** — scenario analysis (crash, tech selloff, rate shock)
- **Transformer Return Predictor** — multi-asset PyTorch model for dynamic allocation

## Quick Start
```bash
pip install -r requirements.txt
python build_all.py
```

## Key Optimizations
- **Numba JIT** for PSOR inner loop (100x+ speedup)
- **Vectorized Monte Carlo** (no Python loops for simulation)
- **Ledoit-Wolf shrinkage** for robust covariance estimation
- **Sparse linear algebra** for efficient tridiagonal solves
- **GARCH(1,1)** volatility forecasting from scratch (MLE)
- **Transformer models** (PyTorch) for volatility & return prediction
- **Interactive Plotly** charts with hover tooltips + HTML export
- **PCA factor analysis** for systematic vs idiosyncratic risk
- Professional dark-theme visualizations throughout

## Dependencies
| Package | Purpose |
|---------|---------|
| numpy, scipy | Core numerical computing |
| matplotlib, seaborn | Static visualization |
| plotly | Interactive visualization |
| cvxpy | Convex optimization |
| yfinance | Market data API |
| numba | JIT compilation |
| scikit-learn | Covariance shrinkage, PCA |
| torch | Transformer neural networks |

## License
MIT