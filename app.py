import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px

# Page Config
st.set_page_config(page_title="NEAT-DDPG Stock Forecasting", layout="centered")

# --- HEADER ---
st.title("📈 Intelligent Stock Forecasting with Evolutionary Deep Reinforcement Learning")
st.markdown("""
Built by **Team 2** | Guided by **Ms. Anisha Radhakrishnan**  
This project presents a hybrid AI framework combining **NEAT** and **DDPG** to forecast stock trends and optimize trading strategies.
""")

# --- OBJECTIVE ---
st.header("🎯 Objective")
st.markdown("""
Build a predictive model using:
- 📊 Stock indicators
- 📰 News sentiment
- 🌐 Economic signals  
Integrated using **NEAT-enhanced DDPG** for adaptive decision making.
""")

# --- PROBLEM STATEMENT ---
st.header("❓ Problem Statement")
st.write("""
Traditional models fail to adapt to volatile market conditions. Our hybrid RL-evolutionary system overcomes this using multimodal inputs and self-evolving networks.
""")

# --- FEEDBACK ---
st.header("🔁 What We Improved")
st.write("""
- Clearer architecture explanation  
- Real-world evaluation metrics  
- Improved visuals and SHAP integration  
""")

# --- ARCHITECTURE ---
st.header("🧠 System Architecture")
st.image("system_architecture.jpg", caption="NEAT-DDPG System Architecture", use_container_width=True)

# --- REWARD FUNCTION ---
st.header("🎯 Reward Function")
st.markdown("""
Our reward balances:
- ✅ Profit
- 📉 Risk (Sharpe)
- 💸 Trading cost
- ⏳ Time decay
""")

# --- DATASET SELECTOR ---
st.header("📑 View Stock Dataset")
stock_choice = st.selectbox("Choose a dataset:", ["Amazon", "Apple", "Microsoft"])
if stock_choice == "Amazon":
    try:
        df = pd.read_csv("final_dataset_converted.csv")
        st.success("Amazon dataset loaded successfully.")
        st.dataframe(df.head(50))
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info(f"{stock_choice} dataset is currently in progress.")


# --- RL ALGORITHM TABLE ---
st.header("⚖️ RL Algorithm Performance Table")
rl_data = {
    "Model": ["SAC", "A2C", "DDPG", "TD3"],
    "Sharpe Ratio": [0.73, 1.23, 1.40, 1.47],
    "Max Drawdown": ["-23.12%", "-35.40%", "-11.33%", "-10.87%"],
    "Annualized Return": [1370.41, 21023.70, 41012.00, 40311.28],
    "Stability": ["High", "Medium", "Medium", "High"]
}
st.dataframe(pd.DataFrame(rl_data))

# --- INTERACTIVE CHARTS ---
st.header("📊 Model Metric Comparisons (Interactive)")
models_all = ["Buy-and-Hold", "SAC", "A2C", "DDPG", "TD3"]
sharpe = [0.58, 0.73, 1.23, 1.40, 1.47]
returns = [7.6, 13.7, 21.0, 41.0, 40.3]
drawdowns = [-22.4, -23.12, -35.4, -11.33, -10.87]
volatility = [15.2, 18.5, 20.2, 14.6, 13.9]

fig1 = px.bar(x=models_all, y=sharpe, color=models_all, title="Sharpe Ratio Comparison")
fig1.update_traces(text=sharpe, textposition='outside')
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(x=models_all, y=returns, color=models_all, title="Annualized Return (%)")
fig2.update_traces(text=returns, textposition='outside')
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.bar(x=models_all, y=drawdowns, color=models_all, title="Max Drawdown (%)")
fig3.update_traces(text=drawdowns, textposition='outside')
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.bar(x=models_all, y=volatility, color=models_all, title="Volatility (%)")
fig4.update_traces(text=volatility, textposition='outside')
st.plotly_chart(fig4, use_container_width=True)

# --- STRATEGY COMPARISON TABLE ---
st.header("📈 NEAT-DDPG vs Buy-and-Hold")
comp = {
    "Metric": ["Sharpe Ratio", "Max Drawdown", "Annualized Return", "Annualized Volatility"],
    "Buy-and-Hold": [0.58, "-22.4%", "7.6%", "15.2%"],
    "NEAT-DDPG": [2.04, "-10.5%", "19.4%", "12.8%"]
}
st.dataframe(pd.DataFrame(comp))
# --- RESULTS ---
st.header("📊 Model Results")
st.subheader("📈 Portfolio Performance")
# Generate simulation

import numpy as np
import plotly.graph_objs as go

np.random.seed(42)
time = np.arange(0, 225)
buy_hold = 10000 + np.cumsum(np.random.normal(5, 50, len(time)))
ddpg = 10000 + np.cumsum(np.random.normal(10, 40, len(time)))

# Plotly line chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=buy_hold, mode='lines', name='Buy-and-Hold', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=time, y=ddpg, mode='lines', name='DDPG Portfolio', line=dict(color='green', dash='dash')))

fig.update_layout(
    title='Simulated Portfolio Value Over Time',
    xaxis_title='Time Steps',
    yaxis_title='Portfolio Value',
    legend=dict(x=0.01, y=0.99),
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# Create and offer download of dataset
import pandas as pd
sim_df = pd.DataFrame({'Time': time, 'Buy-and-Hold': buy_hold, 'DDPG Portfolio': ddpg})
csv_data = sim_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Simulated Portfolio Data", data=csv_data, file_name="simulated_portfolio.csv", mime="text/csv")

# --- SHAP ---
st.header("🧠 Explainability")
st.image("shap.png", caption="Feature Attribution (Buy Action)", use_container_width=True)

# --- PERFORMANCE SUMMARY (IN GRAY BOX) ---
# --- PERFORMANCE SUMMARY (FULLY WRAPPED IN GRAY BOX) ---
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
        <span style='font-size: 14px; color: lightgray;'>+39% vs TD3<br>+252% vs Buy-and-Hold</span>
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
        🚀 <b>Return</b><br>
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

# --- DIFFERENTIATORS ---
st.header("🚀 What Sets Us Apart")
st.markdown("""
1. 🧬 **NEAT evolves actor-critic topology dynamically**  
2. 💬 **Sentiment shaping with FinBERT & news mapping**  
3. 📉 **Real-world metrics: Sharpe, MDD, volatility**  
4. 🔁 **Episodic self-improvement with evolution**
""")

# --- WHY OUR MODEL IS BETTER ---
st.header("📌 Why Our Model is Better Than Others")
st.markdown("""
Our proposed **NEAT-DDPG hybrid model** offers significant advantages over both traditional reinforcement learning models and conventional strategies like Buy-and-Hold.

### 🧬 1. Dynamic Policy Evolution with NEAT
- Adapts network architecture to market conditions
- Improves feature selection and learning flexibility

### 📈 2. Proven Superior Performance

| Metric               | Buy-and-Hold | NEAT-DDPG |
|----------------------|--------------|-----------|
| Sharpe Ratio         | 0.58         | 2.04      |
| Max Drawdown (%)     | -22.4%       | -10.5%    |
| Annualized Return (%)| 7.6%         | 19.4%     |
| Volatility (%)       | 15.2%        | 12.8%     |

### 💬 3. Multimodal Feature Fusion
Includes:
- Technical indicators
- Sentiment scores (FinBERT)
- Piotroski fundamental scores

### 🔍 4. Explainability via SHAP
Clear interpretability using SHAP values ensures transparency in decisions.

### 📊 5. Outperforms RL Benchmarks

| Model | Sharpe | Drawdown | Return | Stability |
|-------|--------|----------|--------|-----------|
| SAC   | 0.73   | -23.12%  | 13.7%  | High      |
| A2C   | 1.23   | -35.40%  | 21.0%  | Medium    |
| DDPG  | 1.40   | -11.33%  | 41.0%  | Medium    |
| TD3   | 1.47   | -10.87%  | 40.3%  | High      |
| **NEAT-DDPG** | **2.04** | **-10.5%** | **19.4%** | **High + Adaptive** |

### 🔁 6. Generalizable & Future-Ready
Can scale to multi-agent RL and LLM-integrated financial pipelines.
""")

# --- LIMITATIONS SECTION ---
st.header("⚠️ Limitations & Future Work")
st.markdown("""
While our NEAT-DDPG framework achieves high accuracy and low risk, we recognize the following limitations:

- 🔄 **Retraining Time**: Neuroevolution increases convergence time, especially with large networks.
- 🧠 **Interpretability Challenge**: While SHAP helps, evolved topologies make full interpretability harder.
- 🌍 **Scalability to Multiple Stocks**: Currently trained per stock; multi-stock portfolio training is future work.
- 💬 **Sentiment Domain Dependence**: FinBERT works well, but generalized LLM-based sentiment may improve accuracy.

These will be addressed in upcoming work through multi-agent RL, faster evolution schemes, and LLM fusion.
""")

# # --- COLAB OUTPUT RESULTS ---
# st.header("📘 Colab Output Highlights")
# st.image("colab_outputs/reward_curve.png", caption="Reward vs Training", use_container_width=True)
# st.image("colab_outputs/portfolio_performance.png", caption="Portfolio Growth", use_container_width=True)
# st.image("colab_outputs/action_dist.png", caption="Action Distribution", use_container_width=True)
# st.image("colab_outputs/metrics_table.png", caption="Metrics Overview", use_container_width=True)

# --- DOWNLOADS ---
st.header("📥 Downloads")
st.download_button("📉 Final Dataset (CSV)", open("final_dataset_converted.csv", "rb"), file_name="final_dataset.csv")
st.download_button("📄 Research Report (DOCX)", open("Updated_Paper_with_all_refernces.docx", "rb"), file_name="NEAT_DDPG_Final_Paper.docx")

# --- PUBLICATION ---
st.header("📢 Publication")
st.markdown("Submitted to **ICACCP 2025 – International Conference on Advanced Computational and Communication Paradigm**.")


# --- FOOTER ---
st.markdown("---")
st.caption("👨‍💻 Built by Team 2 | 📘 Guided by Ms. Anisha Radhakrishnan | 🗓️ March 2025")
