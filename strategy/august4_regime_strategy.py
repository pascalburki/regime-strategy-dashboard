import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

df = yf.download('NG=F', start='2018-01-01', end='2023-12-31')
df.columns = df.columns.get_level_values(0)
close = df["Close"]

returns = np.log(close / close.shift(1)).dropna()
vol = returns.rolling(window=20).std().dropna()
df["returns"] = returns
df["vol"] = vol
df_clean = df.dropna().copy()

# --- FIX: explicit, stable HMM initialization (same approach as august5/6) ---
# Initial guesses are in SCALED units (mean 0, std 1 after StandardScaler):
# state 0 = calm (below-average vol), state 1 = moderate, state 2 = stress (negative return, high vol)
INIT_MEANS = np.array([
    [0.0, -0.8],
    [0.3, 0.2],
    [-0.5, 1.2],
])
INIT_COVARS = np.array([
    [0.3, 0.3],
    [0.3, 0.3],
    [0.3, 0.3],
])
INIT_STARTPROB = np.array([0.34, 0.33, 0.33])
INIT_TRANSMAT = np.array([
    [0.9, 0.05, 0.05],
    [0.05, 0.9, 0.05],
    [0.05, 0.05, 0.9],
])


def fit_stable_hmm(x_scaled, label=""):
    """Fit a GaussianHMM with explicit stable init instead of random init,
    and report whether it actually converged."""
    model = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        random_state=3,
        n_iter=200,
        tol=0.01,
        init_params="",
    )
    model.startprob_ = INIT_STARTPROB.copy()
    model.transmat_ = INIT_TRANSMAT.copy()
    model.means_ = INIT_MEANS.copy()
    model.covars_ = INIT_COVARS.copy()
    model.fit(x_scaled)
    converged = model.monitor_.converged
    print(f"[{label}] converged: {converged} (iterations used: {model.monitor_.iter})")
    if not converged:
        print(f"[{label}] WARNING: model did not converge — results below are unreliable.")
    return model


def derive_labels(df_with_states, state_col, return_col="returns", vol_col="vol"):
    """Rank states by vol to find 'calm' (lowest vol), then rank the remaining
    two by return to find 'stress' (most negative) vs 'moderate'.
    Returns a dict {state_number: label} and prints the means used to derive it."""
    means = df_with_states.groupby(state_col)[[return_col, vol_col]].mean()
    print(means)
    vol_rank = means[vol_col].sort_values()
    calm_state = vol_rank.index[0]
    remaining = vol_rank.index[1:]
    remaining_returns = means.loc[remaining, return_col].sort_values()
    stress_state = remaining_returns.index[0]   # most negative return among the two higher-vol states
    moderate_state = remaining_returns.index[1]
    label_map = {calm_state: "calm", moderate_state: "moderate", stress_state: "stress"}
    print(f"Derived label map: {label_map}")
    return label_map


def derive_labels_strict_vol(df_with_states, state_col, return_col="returns", vol_col="vol"):
    """
    Alternative to derive_labels(): ranks ALL THREE states purely by
    volatility (ascending), rather than using return direction to break
    the tie between the two higher-vol states. This tests whether the
    drawdown finding depends on the current return-based tie-break rule.
    """
    means = df_with_states.groupby(state_col)[[return_col, vol_col]].mean()
    vol_rank = means[vol_col].sort_values()
    calm_state, moderate_state, stress_state = vol_rank.index[0], vol_rank.index[1], vol_rank.index[2]
    label_map = {calm_state: "calm", moderate_state: "moderate", stress_state: "stress"}
    print(f"Derived label map (STRICT VOLATILITY RANK): {label_map}")
    return label_map


exposure_by_label = {"calm": 1.0, "moderate": 0.4, "stress": 0.0}

# ============================================================
# PART 1: Full-period model (the intentionally biased backtest)
# ============================================================
scaler_full = StandardScaler()
x_full_scaled = scaler_full.fit_transform(df_clean[["returns", "vol"]])

model_full = fit_stable_hmm(x_full_scaled, label="full-period")
df_clean["hidden_states"] = model_full.predict(x_full_scaled)

label_map_full = derive_labels(df_clean, "hidden_states")
df_clean["regime_label"] = df_clean["hidden_states"].map(label_map_full)
df_clean["exposure"] = df_clean["regime_label"].map(exposure_by_label)
df_clean["strategy_returns"] = df_clean["returns"] * df_clean["exposure"]

total_strategy_return = df_clean["strategy_returns"].sum()
total_buy_hold_return = df_clean["returns"].sum()
strategy_sharpe = df_clean["strategy_returns"].mean() / df_clean["strategy_returns"].std() * np.sqrt(252)
buyhold_sharpe = df_clean["returns"].mean() / df_clean["returns"].std() * np.sqrt(252)

print(f"\n[Full-period / biased] Strategy total return (log): {total_strategy_return:.4f}")
print(f"[Full-period / biased] Buy-and-hold total return (log): {total_buy_hold_return:.4f}")
print(f"[Full-period / biased] Strategy Sharpe: {strategy_sharpe:.4f}")
print(f"[Full-period / biased] Buy-and-hold Sharpe: {buyhold_sharpe:.4f}")

df_clean["cumulative_strategy"] = df_clean["strategy_returns"].cumsum()
df_clean["cumulative_buyhold"] = df_clean["returns"].cumsum()

# ============================================================
# PART 2: Realistic split — train on 2018-2020 only, test on 2021-2023
# ============================================================
train = df_clean.loc[:'2020-12-31'].copy()
test = df_clean.loc['2021-01-01':].copy()

# --- FIX: scaler fit ONLY on train, then applied to test (no leakage) ---
scaler_train = StandardScaler()
x_train_scaled = scaler_train.fit_transform(train[["returns", "vol"]])
x_test_scaled = scaler_train.transform(test[["returns", "vol"]])

model_realistic = fit_stable_hmm(x_train_scaled, label="train-only (2018-2020)")

train["hidden_states"] = model_realistic.predict(x_train_scaled)
test["hidden_states"] = model_realistic.predict(x_test_scaled)

# --- FIX: derive labels from the TRAIN-ONLY model's own state means, don't reuse label_map_full ---
label_map_realistic = derive_labels(train, "hidden_states")

# --- Sanity check: do the two models' label derivations at least look structurally similar? ---
# (Not a guarantee of correctness, but catches an obviously broken mapping.)
if set(label_map_full.values()) != set(label_map_realistic.values()):
    print("WARNING: label sets differ between the two models — something is structurally wrong.")

test["regime_label"] = test["hidden_states"].map(label_map_realistic)
test["exposure"] = test["regime_label"].map(exposure_by_label)
test["strategy_returns"] = test["returns"] * test["exposure"]

total_strategy_realistic = test["strategy_returns"].sum()
total_buyhold_realistic = test["returns"].sum()
strategy_sharpe_realistic = test["strategy_returns"].mean() / test["strategy_returns"].std() * np.sqrt(252)
buyhold_sharpe_realistic = test["returns"].mean() / test["returns"].std() * np.sqrt(252)
strategy_dd_realistic = (test["strategy_returns"].cumsum() - test["strategy_returns"].cumsum().cummax()).min()
buyhold_dd_realistic = (test["returns"].cumsum() - test["returns"].cumsum().cummax()).min()

print(f"\n[Realistic: train 2018-2020, test 2021-2023]")
print(f"Strategy total return (log): {total_strategy_realistic:.4f}")
print(f"Buy-and-hold total return (log): {total_buyhold_realistic:.4f}")
print(f"Strategy Sharpe: {strategy_sharpe_realistic:.4f}")
print(f"Buy-and-hold Sharpe: {buyhold_sharpe_realistic:.4f}")
print(f"Strategy max drawdown: {strategy_dd_realistic:.4f}")
print(f"Buy-and-hold max drawdown: {buyhold_dd_realistic:.4f}")

# ============================================================
# PART 3: Alternative labeling test — does the drawdown finding survive
# under strict-volatility labels instead of the return-based tie-break?
# ============================================================
label_map_realistic_strict = derive_labels_strict_vol(train, "hidden_states")

test["regime_label_strict"] = test["hidden_states"].map(label_map_realistic_strict)
test["exposure_strict"] = test["regime_label_strict"].map(exposure_by_label)
test["strategy_returns_strict"] = test["returns"] * test["exposure_strict"]

total_strategy_strict = test["strategy_returns_strict"].sum()
strategy_sharpe_strict = test["strategy_returns_strict"].mean() / test["strategy_returns_strict"].std() * np.sqrt(252)
strategy_dd_strict = (test["strategy_returns_strict"].cumsum() - test["strategy_returns_strict"].cumsum().cummax()).min()

print(f"\n[Realistic, STRICT VOLATILITY LABELING] Strategy total return (log): {total_strategy_strict:.4f}")
print(f"[Realistic, STRICT VOLATILITY LABELING] Strategy Sharpe: {strategy_sharpe_strict:.4f}")
print(f"[Realistic, STRICT VOLATILITY LABELING] Strategy max drawdown: {strategy_dd_strict:.4f}")
print(f"\n(Compare against the return-based labeling above: return {total_strategy_realistic:.4f}, "
      f"Sharpe {strategy_sharpe_realistic:.4f}, drawdown {strategy_dd_realistic:.4f})")

plt.plot(df_clean.index, df_clean["cumulative_strategy"], label="Regime Strategy (biased, full-period)", color="green")
plt.plot(df_clean.index, df_clean["cumulative_buyhold"], label="Buy and Hold", color="gray")
plt.title("Cumulative Returns: Regime Strategy vs. Buy and Hold (NG=F)")
plt.xlabel("Date")
plt.ylabel("Cumulative Log Return")
plt.legend()
plt.show()
