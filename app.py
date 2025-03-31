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

# Sharpe
fig1 = px.bar(x=models_all, y=sharpe, color=models_all, title="Sharpe Ratio Comparison", labels={"x": "Model", "y": "Sharpe Ratio"})
fig1.update_traces(text=sharpe, textposition='outside')
st.plotly_chart(fig1, use_container_width=True)

# Return
fig2 = px.bar(x=models_all, y=returns, color=models_all, title="Annualized Return (%)", labels={"x": "Model", "y": "Return"})
fig2.update_traces(text=returns, textposition='outside')
st.plotly_chart(fig2, use_container_width=True)

# Drawdown
fig3 = px.bar(x=models_all, y=drawdowns, color=models_all, title="Max Drawdown (%)", labels={"x": "Model", "y": "Drawdown"})
fig3.update_traces(text=drawdowns, textposition='outside')
st.plotly_chart(fig3, use_container_width=True)

# Volatility
fig4 = px.bar(x=models_all, y=volatility, color=models_all, title="Volatility (%)", labels={"x": "Model", "y": "Volatility"})
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

# --- DIFFERENTIATORS ---
st.header("🚀 What Sets Us Apart")
st.markdown("""
1. 🧬 **NEAT evolves actor-critic topology dynamically**  
2. 💬 **Sentiment shaping with FinBERT & news mapping**  
3. 📉 **Real-world metrics: Sharpe, MDD, volatility**  
4. 🔁 **Episodic self-improvement with evolution**
""")


# --- RESULTS ---
st.header("📊 Model Results")

st.subheader("Key Metrics")
st.markdown("""
- Sharpe Ratio: `2.68`  
- Max Drawdown: `-12.7%`  
- Annualized Return: `18.3%`
""")

st.subheader("📈 Portfolio Performance")
st.image("portfolio.png", caption="NEAT-DDPG vs Buy-and-Hold", use_container_width=True)

# --- SHAP ---
st.header("🧠 Explainability")
st.image("shap.png", caption="Feature Attribution (Buy Action)", use_container_width=True)


# --- COLAB RESULTS ---
st.header("📘 Colab Output Highlights")

st.subheader("Reward Curve")
st.image("colab_outputs/reward_curve.png", use_container_width=True)

st.subheader("Portfolio Growth")
st.image("colab_outputs/portfolio_performance.png", use_container_width=True)

st.subheader("Action Distribution")
st.image("colab_outputs/action_dist.png", use_container_width=True)

st.subheader("Metric Snapshot")
st.image("colab_outputs/metrics_table.png", use_container_width=True)

# --- DOWNLOADS ---
st.header("📥 Downloads")
st.download_button("📉 Final Dataset (CSV)", open("final_dataset.csv", "rb"), file_name="final_dataset.csv")
st.download_button("📄 Research Report (DOCX)", open("Updated_Paper_with_all_refernces.docx", "rb"), file_name="NEAT_DDPG_Final_Paper.docx")

# --- PUBLICATION ---
st.header("📢 Publication")
st.markdown("Submitted to **ICACCP 2025 – International Conference on Advanced Computational and Communication Paradigm**.")

# --- FOOTER ---
st.markdown("---")
st.caption("👨‍💻 Built by Team 2 | 📘 Guide: Ms. Anisha Radhakrishnan | 🗓️ March 2025")
