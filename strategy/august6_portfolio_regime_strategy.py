import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

assets = ['SPY', 'QQQ', 'GLD', 'TLT', 'XOM']
data = {}
for asset in assets:
    df = yf.download(asset, '2018-01-01', '2023-12-31')
    df.columns = df.columns.get_level_values(0)
    data[asset] = df["Close"]

combined = pd.concat(data.values(), axis=1)
combined.columns = assets
combined_clean = combined.dropna()

log_returns = np.log(combined_clean / combined_clean.shift(1)).dropna()
portfolio_returns = log_returns[assets].mean(axis=1)

spy_returns = log_returns['SPY']
spy_vol = spy_returns.rolling(window=20).std().dropna()

hmm_data = pd.DataFrame({"returns": spy_returns, "vol": spy_vol}).dropna()

start_date = pd.Timestamp('2020-01-01')
current_date = start_date
convergence_failures = 0
total_windows = 0

all_states = []

while current_date < hmm_data.index[-1]:
    next_date = current_date + pd.DateOffset(months=1)

    train_data = hmm_data.loc[:current_date]
    test_data = hmm_data.loc[current_date:next_date]

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
    model.means_ = np.array([[0.0, 0.01], [0.001, 0.02], [-0.001, 0.03]])
    model.covars_ = np.array([[0.0001, 0.0001], [0.0001, 0.0001], [0.0001, 0.0001]])

    try:
        model.fit(train_data)
    except Exception:
        convergence_failures += 1
        current_date = next_date
        continue

    if not model.monitor_.converged:
        convergence_failures += 1

    if model.monitor_.iter >= 190:  # close to the 200 limit
        print(f"Window ending {current_date}: used {model.monitor_.iter} iterations (near limit)")    

    predicted_states = model.predict(test_data)
    all_states.append(pd.Series(predicted_states, index=test_data.index))
    current_date = next_date

regime_signal = pd.concat(all_states)
print(f"Convergence failures: {convergence_failures}/{total_windows}")

exposure = {0: 1.2, 1: 1.0, 2: 0.0}

test_period_returns = portfolio_returns.loc[regime_signal.index]
portfolio_exposure = regime_signal.map(exposure)
strategy_returns = test_period_returns * portfolio_exposure

buyhold_return = test_period_returns.sum()
strategy_return = strategy_returns.sum()

print(f"Portfolio buy-and-hold return: {buyhold_return:.4f}")
print(f"Portfolio strategy return: {strategy_return:.4f}")

strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
buyhold_sharpe = test_period_returns.mean() / test_period_returns.std() * np.sqrt(252)

strategy_max_dd = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
buyhold_max_dd = (test_period_returns.cumsum() - test_period_returns.cumsum().cummax()).min()

print(f"Strategy Sharpe: {strategy_sharpe:.4f}")
print(f"Buy-and-hold Sharpe: {buyhold_sharpe:.4f}")
print(f"Strategy max drawdown: {strategy_max_dd:.4f}")
print(f"Buy-and-hold max drawdown: {buyhold_max_dd:.4f}")

strategy_returns.to_csv("strategy_returns.csv")
test_period_returns.to_csv("buyhold_returns.csv")
log_returns.to_csv("asset_returns.csv")
corr_matrix_export = log_returns.corr()
corr_matrix_export.to_csv("correlation_matrix.csv")