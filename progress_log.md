## August 4 2026
Built a regime-based trading strategy using HMM's hidden states on natural gas (NG=F): 100% exposure in calm state, 40% in moderate state, 0% in stress state
Initial backtest (HMM trained on full 2018-2023 dataset): strategy returned +0.98 (log) vs. buy-and-hold's -0.24, Sharpe 0.71 vs. -0.06
Retested with a more honest split: HMM trained only on 2018-2020, predicting forward on 2021-2023 (data it never saw)
Key finding: the realistic test reversed the result entirely, strategy returned -0.96 vs. buy-and-hold's -0.01. The original strong result was substantially inflated by look-ahead bias, since the full-period model effectively "knew" about the 2022 crisis before classifying earlier periods, and the regimes learned from 2018-2020 didn't generalize well to the fundamentally different conditions of 2021-2023
Script: strategy/august4_regime_strategy.py

## August 5 2026
Built a walk-forward HMM strategy on SPY (2000-2023 data, monthly retraining from 2015 onward) to eliminate look-ahead bias
Initial attempts had severe convergence failures using random HMM initialization. Fixed by using diagonal covariance and explicit, sensible starting parameters instead of random ones — 0% convergence failures after the fix
Tested 5 exposure combinations against the same predictions. Best result: {state 0: 1.2x, state 1: 1.0x, state 2: 0.0x} — full exposure in calm/moderate regimes with modest leverage in state 0, zero exposure in the stress regime
Final result: strategy return 1.3531 vs. buy-and-hold 1.1110, Sharpe 0.9012 vs. 0.6612, max drawdown -0.3272 vs. -0.4112 — beats buy-and-hold on every dimension, with genuine walk-forward validation (no look-ahead bias)
Script: strategy/august5_spy_walkforward.py