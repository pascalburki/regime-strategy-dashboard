## August 4 2026
Built a regime-based trading strategy using HMM's hidden states on natural gas (NG=F): 100% exposure in calm state, 40% in moderate state, 0% in stress state
Initial backtest (HMM trained on full 2018-2023 dataset): strategy returned +0.98 (log) vs. buy-and-hold's -0.24, Sharpe 0.71 vs. -0.06
Retested with a more honest split: HMM trained only on 2018-2020, predicting forward on 2021-2023 (data it never saw)
Key finding: the realistic test reversed the result entirely, strategy returned -0.96 vs. buy-and-hold's -0.01. The original strong result was substantially inflated by look-ahead bias, since the full-period model effectively "knew" about the 2022 crisis before classifying earlier periods, and the regimes learned from 2018-2020 didn't generalize well to the fundamentally different conditions of 2021-2023
Script: strategy/august4_regime_strategy.py