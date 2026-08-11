import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

df = yf.download('SPY', start='2000-01-01', end='2023-12-31')
df.columns = df.columns.get_level_values(0)
close = df["Close"]

returns = np.log(close / close.shift(1)).dropna()
vol = returns.rolling(window=20).std().dropna()
df["returns"] = returns
df["vol"] = vol
df_clean = df.dropna().copy()

exposure = {0: 1.2, 1: 1.0, 2: 0.0}

start_date = pd.Timestamp('2015-01-01')
current_date = start_date
convergence_failures = 0
total_windows = 0

all_states = []
all_returns = []
all_fitted_means = []  # FIX: track per-window fitted means to verify state identity held across refits

while current_date < df_clean.index[-1]:
    next_date = current_date + pd.DateOffset(months=1)

    train_data = df_clean.loc[:current_date, ["returns", "vol"]]
    test_data = df_clean.loc[current_date:next_date, ["returns", "vol"]]

    if len(train_data) < 100 or len(test_data) == 0:
        current_date = next_date
        continue

    total_windows += 1

    model = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        random_state=3,
        n_iter=200,
        tol=0.01,
        init_params=""
    )
    model.startprob_ = np.array([0.34, 0.33, 0.33])
    model.transmat_ = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9]
    ])
    model.means_ = np.array([
        [0.0, 0.01],
        [0.001, 0.02],
        [-0.001, 0.03]
    ])
    model.covars_ = np.array([
        [0.0001, 0.0001],
        [0.0001, 0.0001],
        [0.0001, 0.0001]
    ])

    try:
        model.fit(train_data)
    except Exception:
        # FIX-NOTE: this silently drops the entire test month from the backtest.
        # At 0% observed failure rate this doesn't currently affect results, but
        # if it ever fires, whichever months fail (plausibly the most volatile
        # ones) simply vanish from both the strategy and buy-and-hold series.
        convergence_failures += 1
        current_date = next_date
        continue

    if not model.monitor_.converged:
        convergence_failures += 1

    all_fitted_means.append(model.means_.copy())  # FIX: record for post-hoc identity check

    predicted_states = model.predict(test_data)
    test_returns = df_clean.loc[test_data.index, "returns"]

    all_states.append(pd.Series(predicted_states, index=test_data.index))
    all_returns.append(test_returns)

    current_date = next_date

all_predicted_states = pd.concat(all_states)
all_test_returns = pd.concat(all_returns)

test_exposure = all_predicted_states.map(exposure)
strategy_returns = all_test_returns * test_exposure

strategy_total_return = strategy_returns.sum()
strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
strategy_max_dd = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()

buyhold_total_return = all_test_returns.sum()
buyhold_sharpe = all_test_returns.mean() / all_test_returns.std() * np.sqrt(252)
buyhold_max_dd = (all_test_returns.cumsum() - all_test_returns.cumsum().cummax()).min()

print(f"Total windows: {total_windows}")
print(f"Convergence failures: {convergence_failures} ({100*convergence_failures/total_windows:.1f}%)")
print()
print(f"Strategy total return: {strategy_total_return:.4f}")
print(f"Strategy Sharpe: {strategy_sharpe:.4f}")
print(f"Strategy max drawdown: {strategy_max_dd:.4f}")
print()
print(f"Buy-and-hold total return: {buyhold_total_return:.4f}")
print(f"Buy-and-hold Sharpe: {buyhold_sharpe:.4f}")
print(f"Buy-and-hold max drawdown: {buyhold_max_dd:.4f}")

# FIX: verify state identity actually held across all monthly refits, rather than
# assuming the fixed initialization guaranteed it. If state 0 ever drifted away
# from "calm" (e.g. its return mean went strongly negative in some window), the
# exposure mapping would have been silently wrong for that window.
means_array = np.array(all_fitted_means)  # shape: (n_windows, 3 states, 2 features)
print("\nState identity check across all windows:")
for state in range(3):
    returns_range = (means_array[:, state, 0].min(), means_array[:, state, 0].max())
    vol_range = (means_array[:, state, 1].min(), means_array[:, state, 1].max())
    print(f"  State {state}: return range {returns_range}, vol range {vol_range}")
print("If any state's range crosses zero or overlaps heavily with another state's range,")
print("state identity may not have held consistently — investigate before trusting results.")