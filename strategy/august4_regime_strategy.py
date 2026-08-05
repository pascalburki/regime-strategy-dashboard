import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

df = yf.download('NG=F', start='2018-01-01', end='2023-12-31')
df.columns = df.columns.get_level_values(0)
close = df["Close"]

returns = np.log(close / close.shift(1)).dropna()
vol = returns.rolling(window=20).std().dropna()
df["returns"] = returns
df["vol"] = vol
df_clean = df.dropna().copy()
x = df_clean[["returns", "vol"]]

model = GaussianHMM(n_components=3, covariance_type="full", random_state=3)
model.fit(x)
hidden_states = model.predict(x)
df_clean["hidden_states"] = hidden_states

exposure = {1 : 1.0, 0 : 0.4, 2 : 0.0}
df_clean["exposure"] = df_clean["hidden_states"].map(exposure)
df_clean["strategy_returns"] = df_clean["returns"] * df_clean["exposure"]

total_strategy_return = df_clean["strategy_returns"].sum()
total_buy_hold_return = df_clean["returns"].sum()

strategy_sharpe = df_clean["strategy_returns"].mean() / df_clean["strategy_returns"].std() * np.sqrt(252)
buyhold_sharpe = df_clean["returns"].mean() / df_clean["returns"].std() * np.sqrt(252)

df_clean["cumulative_strategy"] = df_clean["strategy_returns"].cumsum()
df_clean["cumulative_buyhold"] = df_clean["returns"].cumsum()

print(f"Strategy total return (log): {total_strategy_return:.4f}")
print(f"Buy-and-hold total return (log): {total_buy_hold_return:.4f}")
print(f"Strategy Sharpe ratio: {strategy_sharpe:.4f}")
print(f"Buy-and-hold Sharpe ratio: {buyhold_sharpe:.4f}")

train = df_clean.loc[:'2020-12-31']
test = df_clean.loc['2021-01-01':].copy()

model_realistic = GaussianHMM(n_components=3, covariance_type="full", random_state=3)
model_realistic.fit(train[["returns", "vol"]])

test_states = model_realistic.predict(test[["returns", "vol"]])
test["hidden_states"] = test_states

test["exposure"] = test["hidden_states"].map(exposure)
test["strategy_returns"] = test["returns"] * test["exposure"]

total_strategy_realistic = test["strategy_returns"].sum()
total_buyhold_realistic = test["returns"].sum()

print(f"\n--- Realistic (train 2018-2020, test 2021-2023) ---")
print(f"Strategy total return (log): {total_strategy_realistic:.4f}")
print(f"Buy-and-hold total return (log): {total_buyhold_realistic:.4f}")

plt.plot(df_clean.index, df_clean["cumulative_strategy"], label="Regime Strategy", color="green")
plt.plot(df_clean.index, df_clean["cumulative_buyhold"], label="Buy and Hold", color="gray")
plt.title("Cumulative Returns: Regime Strategy vs. Buy and Hold (NG=F)")
plt.xlabel("Date")
plt.ylabel("Cumulative Log Return")
plt.legend()
plt.show()