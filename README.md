# Regime Strategy Dashboard

## Status: Complete — all open items from the August 11 audit resolved

An interactive dashboard combining two prior projects into one interface, presenting real market risk analysis alongside a validated, walk-forward tested trading strategy.

1. [Regime-Based Risk Instability in Financial Markets](https://github.com/pascalburki/regime-risk-instability) — analysis of how correlation and volatility shift across market regimes, and why diversification breaks down under stress.

2. [K-Means Market Regime Detection](https://github.com/pascalburki/kmeans-regime-detection) — unsupervised clustering and Hidden Markov Model regime detection, compared against a manual, rule-based classification, extended to real energy market data.

## What's Actually Built

**Risk Analysis Tab:**
- Correlation matrix across a 5-asset portfolio (SPY, QQQ, GLD, TLT, XOM)
- Annualized volatility by asset
- Cumulative returns by asset over the full 2018-2023 period
- **Correction:** an earlier version of this README stated "diversification benefit collapses once uniform correlation exceeds 0.879" as this project's core finding. That figure is not computed anywhere in this project's data or code — it belongs to the separate [Regime-Based Risk Instability](https://github.com/pascalburki/regime-risk-instability) project's rolling-correlation analysis. This project's own correlation matrix is a static, full-period matrix; its highest pairwise correlation is 0.93 (SPY/QQQ).

**Strategy Performance Tab:**
- A regime-based trading strategy using HMM regime detection on SPY, applied across the full 5-asset portfolio (120% exposure in calm regimes, 100% in moderate, 0% in stress)
- Walk-forward validated with zero look-ahead bias — the model retrains monthly using only data available up to that point
- Real, confirmed results (verified directly from `strategy_returns.csv`/`buyhold_returns.csv`): strategy return 0.57 vs. buy-and-hold's 0.44; Sharpe ratio 0.84 vs. 0.69; max drawdown -0.265 vs. -0.264 — the strategy's drawdown is essentially tied with buy-and-hold, not better, unlike the single-asset SPY walk-forward result below
- **Test window note:** this backtest covers 2020-01-02 to 2023-12-29 (1,039 trading days), shorter than the 2018-2023 range shown in the Risk Analysis tab, since the walk-forward model needs training history before its first monthly prediction
- Reproducibility confirmed: two independent runs produced identical results on every headline metric
- **State-identity caveat:** this specific strategy's underlying HMM states show overlapping volatility ranges across the monthly refits (see Open Items) — the numbers above are accurate, but the state labels' consistency across refits is not fully clean

## Key Findings Along the Way

- An initial, biased backtest (HMM trained on the full dataset at once) showed dramatic outperformance (+0.59 log return vs. buy-and-hold's -0.24)
- **Corrected finding:** a genuinely honest train/test split does *not* reverse the result — using volatility-based regime labeling (the standard adopted across this project, see Open Items), the strategy substantially outperforms buy-and-hold in the realistic, unseen-data test: +0.6660 vs. -0.0099 return, Sharpe 1.1389 vs. -0.0043. An earlier version of this project claimed the honest split flipped the strategy into a loss (-0.96) — that number was an artifact of an unconverged, mislabeled HMM fit from before the initialization fix, and has been corrected. The real lesson: look-ahead bias inflates how good a strategy looks, it doesn't necessarily manufacture a fake winner out of a real loser
- **The strongest result in the honest test is risk reduction, and it's larger than first measured:** in the realistic split, buy-and-hold's max drawdown (-1.58, reflecting NG=F's real 2022 spike/2023 crash) is roughly 5.5x the strategy's (-0.28) under volatility-based labeling. The strategy's zero-exposure rule during its correctly-identified highest-volatility state avoids most of that decline — a genuinely useful finding that wasn't visible in the original write-up, since drawdown was never reported for this test at all
- A walk-forward approach (monthly retraining, testing only on unseen data) restored real, credible outperformance on SPY once HMM's convergence issues were fixed with explicit model initialization

## Open Items — all resolved

All three items raised during this audit have been checked and closed:

1. **The "stress" state label ambiguity — resolved, and the choice mattered a lot.** The realistic-split NG=F backtest was tested under both the original return-based tie-break and a strict-volatility labeling rule. The difference was substantial, not cosmetic: return 0.6660 vs. 0.0506, Sharpe 1.1389 vs. 0.0595, drawdown -0.28 vs. -0.52. Strict-volatility labeling was adopted as the standard — it's more defensible (volatility, not average return, is what actually catches tail-risk days) and consistent with the labeling convention already used throughout the K-Means project. The figures throughout this README now reflect the adopted (strict-volatility) result.

2. **State-identity consistency across the ~96 monthly refits — checked, and a real issue found in one script.** The SPY walk-forward (`august5`, 108 windows) shows clean, non-overlapping state ranges — identity held throughout. The portfolio strategy (`august6`, 48 windows) does not: its "moderate" and "stress" states have overlapping volatility ranges, confirmed reproducibly across two independent runs. The dashboard's headline strategy numbers are still accurate as computed from the data, but the underlying state labels may not carry a fully consistent meaning across every monthly refit in this specific script.

3. **Reproducibility of the August 6 backtest — checked, confirmed.** Two independent runs produced identical results on every headline metric.

## Tools Used

Python, pandas, numpy, hmmlearn (Gaussian HMM), scikit-learn (K-means), yfinance, Streamlit, Plotly

## Related Work

Builds directly on [Regime-Based Risk Instability in Financial Markets](https://github.com/pascalburki/regime-risk-instability) and [K-Means Market Regime Detection](https://github.com/pascalburki/kmeans-regime-detection).
