import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="RoT Strategy Dashboard", layout="wide")

st.title("Risk of Target (RoT) Live Dashboard")

SECTORS = {
    "AI and Biotech": ["ABSI", "SDGR", "RXRX", "ABCL", "RLAY", "XNCR", "KRRO", "QSI", "TEM"],
    "Space": ["ASTS", "LUNR", "RKLB", "PL", "SPIR", "BKSY", "RDW", "MYNA", "SATX"],
    "Quantum Technology": ["IONQ", "QSI", "QUBT", "QBTS", "RGTI", "QMCO", "ARQQ"],
    "Nuclear": ["UUUU", "UEC", "OKLO", "SMR", "VST", "BWXT", "CEG", "GEV", "NNE", "SILXY"],
    "Materials": ["ASPI", "AXTA", "RYAM", "FTK", "HUN", "MTRN", "HXL"]
}

BENCHMARK = 'SPY'
configs = {
    'Return': {'weight': 0.25, 'direction': 1},
    'Volatility': {'weight': 0.25, 'direction': -1},
    'Sharpe': {'weight': 0.25, 'direction': 1},
    'MaxDD': {'weight': 0.25, 'direction': -1}
}

with st.sidebar:
    st.header("Configuration")
    selected_sector = st.selectbox("Select Sector", list(SECTORS.keys()))
    lookback = st.slider("Lookback Period (Days)", 100, 500, 252)
    refresh = st.button("Refresh Data")

tickers = SECTORS[selected_sector]
full_list = list(set(tickers + [BENCHMARK]))

# Caching function is required for dashboard performance
@st.cache_data(ttl=300) 
def get_data(ticker_list):
    try:
        df = yf.download(ticker_list, period="2y", auto_adjust=True, progress=False)
        # Handle multi-index if returned
        if isinstance(df.columns, pd.MultiIndex):
            try:
                return df['Close']
            except KeyError:
                return df.iloc[:, :len(ticker_list)] 
        return df['Close']
    except:
        return pd.DataFrame()

raw_data = get_data(full_list)

if raw_data.empty:
    st.error("Failed to download data.")
    st.stop()

raw_data = raw_data.dropna(axis=1, how='all')
data_window = raw_data.iloc[-lookback:]
returns = data_window.pct_change(fill_method=None).dropna()

metrics = pd.DataFrame(index=returns.columns)
metrics['Return'] = returns.mean() * 252
metrics['Volatility'] = returns.std() * np.sqrt(252)
metrics['Sharpe'] = metrics['Return'] / metrics['Volatility']

cum_returns = (1 + returns).cumprod()
running_max = cum_returns.cummax()
drawdown = (cum_returns / running_max) - 1
metrics['MaxDD'] = drawdown.min().abs()

normalized = pd.DataFrame(index=metrics.index)
for col, config in configs.items():
    min_val = metrics[col].min()
    max_val = metrics[col].max()
    rng = max_val - min_val
    
    if rng == 0:
        normalized[col] = 0.5
    elif config['direction'] == 1:
        normalized[col] = (metrics[col] - min_val) / rng
    else:
        normalized[col] = (max_val - metrics[col]) / rng

scores = pd.DataFrame(index=metrics.index)
scores['EI'] = 0.0
for col, config in configs.items():
    scores['EI'] += normalized[col] * config['weight']

if BENCHMARK in scores.index:
    bench_ei = scores.loc[BENCHMARK, 'EI']
    scores['RoT'] = (scores['EI'] - bench_ei) / bench_ei
else:
    scores['RoT'] = np.nan

final_df = pd.concat([metrics, scores], axis=1).sort_values('EI', ascending=False)

for col in final_df.columns:
    if col != 'Ticker':
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

final_df = final_df.replace([np.inf, -np.inf], np.nan).dropna(how='any')

if final_df.empty:
    st.warning(f"Warning: No valid data available for the '{selected_sector}' sector after cleaning. Check ticker symbols or lookback period.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Efficient Frontier (RoT Analysis)")
    fig_scatter = px.scatter(
        final_df, 
        x="Volatility", 
        y="Return", 
        color="EI", 
        size="EI", 
        hover_name=final_df.index,
        color_continuous_scale="Viridis",
        title=f"{selected_sector} vs {BENCHMARK}",
        height=500
    )
    
    # Add crosshair lines for Benchmark
    if BENCHMARK in final_df.index:
        bench_vol = final_df.loc[BENCHMARK, 'Volatility']
        bench_ret = final_df.loc[BENCHMARK, 'Return']
        fig_scatter.add_vline(x=bench_vol, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_hline(y=bench_ret, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_annotation(x=bench_vol, y=bench_ret, text="Benchmark", showarrow=True, arrowhead=1)

    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("RoT Rankings")
    final_df['Color'] = np.where(final_df['RoT'] >= 0, 'green', 'red')
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=final_df.index,
        x=final_df['RoT'],
        orientation='h',
        marker_color=final_df['Color'],
        text=final_df['RoT'].apply(lambda x: f"{x:.1%}"),
        textposition='auto'
    ))
    fig_bar.update_layout(title="Risk of Target %", height=500)
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Detailed Metrics")
st.dataframe(final_df.style.format("{:.4f}").background_gradient(subset=['EI'], cmap='Greens'), use_container_width=True)
