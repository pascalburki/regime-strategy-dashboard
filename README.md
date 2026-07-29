# Regime Strategy Dashboard

## Status: In Progress

This repository will combine two prior projects into a single dashboard:

1. [Regime-Based Risk Instability in Financial Markets](https://github.com/pascalburki/regime-risk-instability) — analysis of how correlation and volatility shift across market regimes, and why diversification breaks down under stress.

2. [K-Means Market Regime Detection](https://github.com/pascalburki/kmeans-regime-detection) — unsupervised clustering used to detect market regimes directly from data, compared against a manual, rule-based classification.

## Planned Contents

- **Analysis view**: correlation matrix, volatility by regime, Sharpe ratio, and the core diversification-failure findings from the Regime Instability project.
- **Strategy view**: a simple trading strategy built on the regime signal from the K-Means project (e.g., adjusting exposure depending on detected regime), backtested against real historical data, with real performance metrics (returns, Sharpe ratio, max drawdown).
- **Combined dashboard**: an interactive interface (built with Streamlit) presenting both the underlying analysis and the strategy's backtested performance in one place.

More detail will be added here as each piece is built.
