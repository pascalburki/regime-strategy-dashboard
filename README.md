# Regime Strategy Dashboard

## Status: Complete

An interactive dashboard combining two prior projects into one interface, presenting real market risk analysis alongside a validated, walk-forward tested trading strategy.

1. [Regime-Based Risk Instability in Financial Markets](https://github.com/pascalburki/regime-risk-instability) — analysis of how correlation and volatility shift across market regimes, and why diversification breaks down under stress.

2. [K-Means Market Regime Detection](https://github.com/pascalburki/kmeans-regime-detection) — unsupervised clustering and Hidden Markov Model regime detection, compared against a manual, rule-based classification, extended to real energy market data.

## What's Actually Built

**Risk Analysis Tab:**
- Correlation matrix across a 5-asset portfolio (SPY, QQQ, GLD, TLT, XOM)
- Annualized volatility by asset
- Cumulative returns by asset over the full 2018-2023 period
- The core finding: diversification benefit collapses once uniform correlation exceeds 0.879, regardless of volatility

**Strategy Performance Tab:**
- A regime-based trading strategy using HMM regime detection on SPY, applied across the full 5-asset portfolio (120% exposure in calm regimes, 100% in moderate, 0% in stress)
- Walk-forward validated with zero look-ahead bias — the model retrains monthly using only data available up to that point
- Real, confirmed results: strategy return 0.57 vs. buy-and-hold's 0.44; Sharpe ratio 0.84 vs. 0.69
- Results are reproducible: identical output across repeated runs, despite some monthly HMM refits requiring many iterations to stabilize

## Key Findings Along the Way

- An initial, biased backtest (HMM trained on the full dataset at once) showed dramatic outperformance, but this was substantially inflated by look-ahead bias
- A genuinely honest train/test split reversed the result entirely — a real, important lesson about validating quant strategies properly before trusting them
- A walk-forward approach (monthly retraining, testing only on unseen data) restored real, credible outperformance once HMM's convergence issues were fixed with explicit model initialization

## Tools Used

Python, pandas, numpy, hmmlearn (Gaussian HMM), scikit-learn (K-means), yfinance, Streamlit, Plotly

## Related Work

Builds directly on [Regime-Based Risk Instability in Financial Markets](https://github.com/pascalburki/regime-risk-instability) and [K-Means Market Regime Detection](https://github.com/pascalburki/kmeans-regime-detection).
