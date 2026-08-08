import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Regime Strategy Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400..600&family=Inter:wght@300;400;500;600&display=swap');

    * { color: #E8E4DA; }

    .stApp {
        background: linear-gradient(180deg, #0A0E17 0%, #0F1420 100%);
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header {visibility: hidden;}

    h1, h2, h3 {
        font-family: 'Fraunces', serif !important;
        color: #5B9BD5 !important;
        font-weight: 500 !important;
    }

    .main-header {
        text-align: center;
        padding: 20px 0 30px 0;
    }

    div[data-testid="stMetric"] {
        background-color: #131826;
        border: 1px solid #253048;
        border-radius: 12px;
        padding: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #7C8AA5 !important;
        font-size: 12px !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    div[data-testid="stMetricValue"] {
        color: #5B9BD5 !important;
        font-family: 'Fraunces', serif !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #6FCF97 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #131826;
        border-radius: 8px;
        padding: 10px 20px;
        border: 1px solid #253048;
        color: #E8E4DA;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1A2338 !important;
        border-color: #5B9BD5 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #131826;
        border: 1px solid #253048 !important;
        border-radius: 12px !important;
    }

    div[data-testid="stMarkdownContainer"] p {
        font-size: 15px;
        line-height: 1.7;
        color: #E8E4DA;
    }

    div[data-testid="stMarkdownContainer"] strong {
        color: #5B9BD5;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>Regime-Based Portfolio Strategy</h1>
    <p style="color:#7C8AA5; letter-spacing:2px; text-transform:uppercase; font-size:12px;">Risk Analysis · Regime Detection · Walk-Forward Strategy</p>
</div>
""", unsafe_allow_html=True)

strategy_returns = pd.read_csv("data/strategy_returns.csv", index_col=0, parse_dates=True).squeeze()
buyhold_returns = pd.read_csv("data/buyhold_returns.csv", index_col=0, parse_dates=True).squeeze()
asset_returns = pd.read_csv("data/asset_returns.csv", index_col=0, parse_dates=True)
corr_matrix = pd.read_csv("data/correlation_matrix.csv", index_col=0)

tab1, tab2 = st.tabs(["📈 Risk Analysis", "🎯 Strategy Performance"])

with tab1:
    st.subheader("Asset Correlation Matrix")
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto"
    )
    fig_corr.update_layout(paper_bgcolor="#0F1420", plot_bgcolor="#0F1420", font_color="#E8E4DA")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Annualized Volatility by Asset")
    vols = asset_returns.std() * np.sqrt(252)
    fig_vol = px.bar(
        x=vols.index, y=vols.values,
        labels={"x": "Asset", "y": "Annualized Volatility"},
        color_discrete_sequence=["#5B9BD5"]
    )
    fig_vol.update_layout(paper_bgcolor="#0F1420", plot_bgcolor="#0F1420", font_color="#E8E4DA")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Cumulative Returns by Asset")
    cumulative_by_asset = asset_returns.cumsum()
    fig_assets = px.line(
        cumulative_by_asset,
        labels={"value": "Cumulative Log Return", "index": "Date"},
        color_discrete_sequence=["#5B9BD5", "#6FCF97", "#E0A857", "#DB6A61", "#9B7EDE"]
    )
    fig_assets.update_layout(paper_bgcolor="#0F1420", plot_bgcolor="#0F1420", font_color="#E8E4DA", legend_title="")
    st.plotly_chart(fig_assets, use_container_width=True)

    with st.container(border=True):
        st.markdown("""
        **Core finding:** diversification benefit collapses once uniform correlation exceeds 0.879,
        regardless of volatility. Correlation, not volatility alone, is the primary driver of
        diversification failure under stress.
        """)

with tab2:
    strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    buyhold_sharpe = buyhold_returns.mean() / buyhold_returns.std() * np.sqrt(252)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Strategy Return", f"{strategy_returns.sum():.2%}")
    col2.metric("Buy & Hold Return", f"{buyhold_returns.sum():.2%}")
    col3.metric("Strategy Sharpe", f"{strategy_sharpe:.2f}", f"{strategy_sharpe - buyhold_sharpe:+.2f}")
    col4.metric("Buy & Hold Sharpe", f"{buyhold_sharpe:.2f}")

    st.subheader("Strategy vs. Buy-and-Hold — Cumulative Returns")
    cumulative_strategy = strategy_returns.cumsum()
    cumulative_buyhold = buyhold_returns.cumsum()

    chart_df = pd.DataFrame({"Strategy": cumulative_strategy, "Buy and Hold": cumulative_buyhold})
    fig_perf = px.line(
        chart_df, labels={"value": "Cumulative Log Return", "index": "Date"},
        color_discrete_map={"Strategy": "#5B9BD5", "Buy and Hold": "#7C8AA5"}
    )
    fig_perf.update_layout(paper_bgcolor="#0F1420", plot_bgcolor="#0F1420", font_color="#E8E4DA", legend_title="")
    st.plotly_chart(fig_perf, use_container_width=True)

    with st.container(border=True):
        st.markdown("""
        **Strategy:** HMM regime detection on SPY drives exposure across a 5-asset portfolio
        (SPY, QQQ, GLD, TLT, XOM) — 120% exposure in calm regimes, 100% in moderate, 0% in stress.

        Walk-forward validated with zero look-ahead bias: the model retrains monthly using only
        data available up to that point.
        """)