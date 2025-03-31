import streamlit as st
from PIL import Image
import pandas as pd


# Page Config
st.set_page_config(page_title="NEAT-DDPG Stock Forecasting", layout="centered")

# --- HEADER ---
st.title("📈 Intelligent Stock Forecasting with Evolutionary Deep Reinforcement Learning")
st.markdown("""
Built by **Team 2** | Guided by **Ms. Anisha Radhakrishnan**  
This project presents a hybrid AI framework combining **NEAT** and **DDPG** to forecast stock trends and optimize trading strategies.
""")

# --- OBJECTIVE ---
with st.container():
    st.header("🎯 Objective")
    st.write("""
    Our goal is to build a predictive model using:
    - 📊 Stock indicators
    - 📰 News & sentiment
    - 🌐 Economic signals

    We integrate these with **NEAT-enhanced DDPG** to make adaptive, real-world trading decisions.
    """)

# --- PROBLEM STATEMENT ---
with st.container():
    st.header("❓ Problem Statement")
    st.write("""
    Traditional models fail in volatile, nonlinear financial environments.  
    Our model combines **reinforcement learning** with **neuroevolution** to dynamically learn optimal strategies using multimodal data.
    """)

# --- FEEDBACK RESPONSE ---
with st.container():
    st.header("🔁 Review Feedback: What We Improved")
    st.write("""
    - ✔️ Clarified architecture & RL-NEAT integration  
    - ✔️ Added detailed evaluation metrics  
    - ✔️ Improved visual presentation of pipeline & outputs  
    """)

# --- ARCHITECTURE ---
st.header("🧠 NEAT-DDPG Architecture")
st.image("system_architecture.jpg", caption="System Architecture Diagram", use_container_width=True)

# --- REWARD FUNCTION ---
st.header("🎯 Reward Function Design")
st.markdown("""
Our custom reward function balances:
- 📈 Profitability (returns)
- 📉 Risk (Sharpe ratio)
- 💰 Transaction costs
- ⏳ Time preference
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
        st.error(f"Failed to load dataset: {e}")
else:
    st.info(f"{stock_choice} dataset is currently being prepared.")

# --- RESULTS ---
st.header("📊 Model Results")

with st.container():
    st.subheader("Key Metrics (from Colab Output)")
    st.markdown("""
    - ✅ **Sharpe Ratio**: `2.68`  
    - 📉 **Max Drawdown**: `-12.7%`  
    - 💹 **Annualized Return**: `18.3%`  
    """)

st.subheader("📈 Portfolio Comparison")
st.image("portfolio.png", caption="NEAT-DDPG vs Buy-and-Hold", use_container_width=True)

# --- SHAP OUTPUT ---
st.header("🧠 Explainability (SHAP)")
st.image("shap.png", caption="Feature Importance for Buy Action", use_container_width=True)

# --- COMPARISON TABLE ---
st.header("⚖️ RL Algorithm Comparison Table")
st.markdown("This table compares key performance metrics across multiple reinforcement learning models.")

comparison_data = {
    "Model": ["SAC", "A2C", "DDPG", "TD3"],
    "Sharpe Ratio": [0.73, 1.23, 1.40, 1.47],
    "Max Drawdown": ["-23.12%", "-35.40%", "-11.33%", "-10.87%"],
    "Annualized Return": [1370.41, 21023.70, 41012.00, 40311.28]
}
df_comparison = pd.DataFrame(comparison_data)
st.dataframe(df_comparison)

# --- DIFFERENTIATORS ---
st.header("🚀 Differentiators: What Makes Our Model Unique")

st.markdown("""
Our NEAT-DDPG framework brings significant innovations compared to traditional trading models and baseline RL agents.

1. 🧬 **Neuroevolution-enhanced Policy Learning**  
   NEAT evolves both the **topology and weights** of actor-critic networks.  
   👉 [Colab: NEAT-DDPG Training](https://colab.research.google.com/drive/1TKZhWtvGgTrz3oOBuJ8ZXQo4fQPGk9pU)

2. 💬 **Sentiment-Driven Reward Shaping**  
   FinBERT sentiment scores directly affect trade decisions.  
   👉 [Colab: Sentiment Pipeline](https://colab.research.google.com/drive/1EmWftskl893qTyIFaVMg5sINS4qGJH_Y)

3. 📊 **Realistic Evaluation Metrics**  
   Includes Sharpe Ratio, MDD, Volatility — not just accuracy.  
   👉 [Colab: Evaluation Notebook](https://colab.research.google.com/drive/1whYH9if3Tu18TEM4tdk2uTSGIUgdN4z-)

4. 🔁 **Episodic Policy Fine-Tuning**  
   Genetic algorithm improves exploration vs gradient-only updates.
""")

# --- COLAB OUTPUT RESULTS ---
st.header("🧾 Visual Results from Colab")

st.subheader("1️⃣ Reward Curve")
st.image("colab_outputs/reward_curve.png", caption="Reward vs Training Episode", use_container_width=True)

st.subheader("2️⃣ Portfolio Performance")
st.image("colab_outputs/portfolio_performance.png", caption="NEAT-DDPG Portfolio Over Time", use_container_width=True)

st.subheader("3️⃣ Action Distribution")
st.image("colab_outputs/action_dist.png", caption="Buy/Hold/Sell Action Distribution", use_container_width=True)

st.subheader("4️⃣ Metric Snapshot Table")
st.image("colab_outputs/metrics_table.png", caption="Sharpe Ratio, MDD, CAGR, Volatility", use_container_width=True)

# --- DOWNLOADS ---
st.header("📥 Downloads")
st.download_button("📉 Download Final Dataset (CSV)", open("final_dataset.csv", "rb"), file_name="final_dataset.csv")
st.download_button("📄 Download Project Report (DOCX)", open("Updated_Paper_with_all_refernces.docx", "rb"), file_name="NEAT_DDPG_Final_Paper.docx")

# --- PUBLICATION ---
st.header("📢 Publication")
st.markdown("📎 Submitted to **ICACCP 2025 – International Conference on Advanced**")