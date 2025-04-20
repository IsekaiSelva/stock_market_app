# Formalized Streamlit App with Enhanced Reasoning and Visuals

import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go

# Page Config
st.set_page_config(page_title="NEAT-DDPG Stock Forecasting", layout="centered")

# Header
st.title("📈 Intelligent Stock Forecasting with Evolutionary Deep Reinforcement Learning")
st.markdown("""
**Final Year B.Tech Project - Team G098**  
**Guided by: Ms. Anisha Radhakrishnan**  
This application demonstrates a hybrid deep learning system that integrates NEAT (NeuroEvolution of Augmenting Topologies) with DDPG (Deep Deterministic Policy Gradient) to forecast stock trends using technical, fundamental, and sentiment indicators.
""")

# Objective
st.header("🎯 Objective")
st.markdown("""
To develop an adaptive stock market forecasting framework capable of:
- Ingesting multimodal data (technical, fundamental, and sentiment-based inputs).
- Dynamically evolving its network topology using neuroevolution techniques.
- Optimizing trading decisions through deep reinforcement learning principles.
""")

# Problem Statement
st.header("❓ Problem Statement")
st.markdown("""
Conventional stock forecasting models are often constrained by fixed architectures, linear assumptions, and lack of adaptability to non-stationary market conditions. Furthermore, they fail to account for qualitative insights such as market sentiment embedded in financial news. Our system addresses these gaps by leveraging NEAT-DDPG and LLM-derived sentiment analysis.
""")

# System Architecture
st.header("🧠 System Architecture")
st.image("Picture1.png", caption="Overview of the NEAT-DDPG Hybrid Forecasting System", use_container_width=True)

# Reward Function
st.header("🎯 Reinforcement Learning Reward Design")
st.markdown("""
The reward mechanism incorporates:
- 📈 Maximization of cumulative returns.
- 🔻 Penalty for high drawdowns and risk exposure.
- 💰 Trading cost regularization.
- ⏱️ Time-based reward shaping to prevent excessive holding.
""")

# Dataset Viewer
st.header("📑 Dataset Viewer")
stock_choice = st.selectbox("Select a stock dataset:", ["Amazon", "Apple", "Microsoft"])
if stock_choice == "Amazon":
    try:
        df = pd.read_excel("final_dataset_converted.xlsx")
        st.success("Amazon dataset loaded successfully.")
        st.dataframe(df.head(50))
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
else:
    st.info(f"The dataset for {stock_choice} will be available in subsequent versions.")

# RL Performance Table
st.header("⚖️ RL Model Benchmark Comparison")
rl_data = {
    "Model": ["SAC", "A2C", "DDPG", "TD3"],
    "Sharpe Ratio": [0.73, 1.23, 1.40, 1.47],
    "Max Drawdown": ["-23.12%", "-35.40%", "-11.33%", "-10.87%"],
    "Annualized Return": [13.7, 21.0, 41.0, 40.3],
    "Stability": ["High", "Medium", "Medium", "High"]
}
st.dataframe(pd.DataFrame(rl_data))

# Interactive Metrics
st.header("📊 Comparative Performance Metrics")
models = ["Buy-and-Hold", "SAC", "A2C", "DDPG", "TD3"]
sharpe = [0.58, 0.73, 1.23, 1.40, 1.47]
returns = [7.6, 13.7, 21.0, 41.0, 40.3]
drawdowns = [-22.4, -23.12, -35.4, -11.33, -10.87]
volatility = [15.2, 18.5, 20.2, 14.6, 13.9]

# Bar Charts with Annotations
fig1 = px.bar(x=models, y=sharpe, color=models, title="Sharpe Ratio Comparison")
fig1.update_traces(text=sharpe, textposition='outside')
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(x=models, y=returns, color=models, title="Annualized Return (%)")
fig2.update_traces(text=returns, textposition='outside')
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.bar(x=models, y=drawdowns, color=models, title="Max Drawdown (%)")
fig3.update_traces(text=drawdowns, textposition='outside')
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.bar(x=models, y=volatility, color=models, title="Annualized Volatility (%)")
fig4.update_traces(text=volatility, textposition='outside')
st.plotly_chart(fig4, use_container_width=True)

# NEAT-DDPG vs Baseline Table
st.header("📈 NEAT-DDPG vs Buy-and-Hold Strategy")
comp = {
    "Metric": ["Sharpe Ratio", "Max Drawdown", "Annualized Return", "Volatility"],
    "Buy-and-Hold": [0.58, "-22.4%", "7.6%", "15.2%"],
    "NEAT-DDPG": [2.04, "-10.5%", "19.4%", "12.8%"]
}
st.dataframe(pd.DataFrame(comp))

# Portfolio Simulation
st.header("📈 Simulated Portfolio Performance")
time = np.arange(0, 225)
buy_hold = 10000 + np.cumsum(np.random.normal(5, 50, len(time)))
ddpg = 10000 + np.cumsum(np.random.normal(10, 40, len(time)))

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=buy_hold, mode='lines', name='Buy-and-Hold', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=time, y=ddpg, mode='lines', name='DDPG Portfolio', line=dict(color='green', dash='dash')))
fig.update_layout(title='Portfolio Value Over Time', xaxis_title='Time Steps', yaxis_title='Portfolio Value')
st.plotly_chart(fig, use_container_width=True)

""# Performance Summary
st.markdown("""
<div style='
    background-color: #2e2e2e;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 25px;
'>
<div style='text-align: center; color: limegreen; font-size: 26px; font-weight: bold; margin-bottom: 20px;'>
📈 Performance Summary
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style='text-align: center; font-size: 20px; color: white;'>
        ✅ <b>Sharpe Ratio</b><br>
        <span style='font-size: 28px; color: limegreen;'>2.04</span><br>
        <span style='font-size: 14px; color: lightgray;'>↑ +39% vs TD3<br>↑ +252% vs Buy-and-Hold</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='text-align: center; font-size: 20px; color: white;'>
        📉 <b>Max Drawdown</b><br>
        <span style='font-size: 28px; color: limegreen;'>-10.5%</span><br>
        <span style='font-size: 14px; color: lightgray;'>↓ 3.4% vs TD3<br>↓ 53% vs Buy-and-Hold</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='text-align: center; font-size: 20px; color: white;'>
        🚀 <b>Annualized Return</b><br>
        <span style='font-size: 28px; color: limegreen;'>19.4%</span><br>
        <span style='font-size: 14px; color: lightgray;'>⚖️ More stable than DDPG<br>↑ 155% vs Buy-and-Hold</span>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style='text-align: center; font-size: 20px; color: white;'>
        📊 <b>Volatility</b><br>
        <span style='font-size: 28px; color: limegreen;'>12.8%</span><br>
        <span style='font-size: 14px; color: lightgray;'>↓ 16% vs Buy-and-Hold</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.header("⚠️ Limitations & Future Work")
st.markdown("""
While our NEAT-DDPG framework achieves high accuracy and low risk, we recognize the following limitations:

- 🔄 **Retraining Time**: Neuroevolution increases convergence time, especially with large networks.
- 🧠 **Interpretability Challenge**: While SHAP helps, evolved topologies make full interpretability harder.
- 🌍 **Scalability to Multiple Stocks**: Currently trained per stock; multi-stock portfolio training is future work.
- 💬 **Sentiment Domain Dependence**: FinBERT works well, but generalized LLM-based sentiment may improve accuracy.

These will be addressed in upcoming work through multi-agent RL, faster evolution schemes, and LLM fusion.
""")
# Downloads""
st.header("📥 Supplementary Downloads")
st.download_button("Download Dataset (CSV)", data=pd.DataFrame({'Time': time, 'Buy-and-Hold': buy_hold, 'DDPG': ddpg}).to_csv(index=False), file_name="portfolio_sim.csv")
st.download_button("Download Report (DOCX)", open("Updated_Paper_with_all_refernces.docx", "rb"), file_name="Research_Report.docx")

# Footer
st.markdown("---")
st.caption("Developed by Arvind P, Harinandan N, Selvakumaran K, Moin Ashvath | Guided by Ms. Anisha Radhakrishnan")
