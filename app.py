
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
from tensorflow import keras

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Demand Forecasting AI",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


# =========================
# LOAD MODELS AND DATA
# =========================

@st.cache_resource
def load_models():
    preprocessor = joblib.load(BASE_DIR / "preprocessor.pkl")
    ml_model = joblib.load(BASE_DIR / "best_model.pkl")
    dl_model = keras.models.load_model(BASE_DIR / "demand_dl_model.keras")

    return preprocessor, ml_model, dl_model


@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "dashboard_data.csv")


@st.cache_data
def load_metrics():
    with open(BASE_DIR / "metrics.json", "r") as file:
        return json.load(file)


try:
    preprocessor, ml_model, dl_model = load_models()
    df = load_data()
    metrics = load_metrics()

except Exception as e:
    st.error("Unable to load project files.")
    st.exception(e)
    st.stop()


# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.stApp {
    background-color: #080D1C;
    color: #F1F5F9;
}

.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}

h1, h2, h3 {
    color: #F1F5F9 !important;
}

/* Input column names in white */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #FFFFFF !important;
}

[data-testid="stMetric"] {
    background: #141E33;
    border: 1px solid #263653;
    padding: 22px;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #60A5FA;
    font-weight: bold;
}

[data-testid="stMetricLabel"] {
    color: #CBD5E1;
}

.stTextInput input,
.stNumberInput input {
    background-color: #141E33;
    color: white;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: #2563EB;
    color: white;
    border: none;
    border-radius: 9px;
    height: 48px;
    font-weight: bold;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #1D4ED8;
    color: white;
}

hr {
    border-color: #263653;
}

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================

st.title("📊 Demand Forecasting AI")
st.caption("Product Demand Prediction using Machine Learning & Deep Learning")

st.divider()


# =========================
# PRODUCT INPUTS
# =========================

st.subheader("📦 Product Information")
st.caption("Enter product details to predict its demand.")

feature_columns = [
    col for col in df.columns if col != "Demand"
]

cat_columns = df[feature_columns].select_dtypes(
    include=["object", "bool", "category"]
).columns.tolist()

input_values = {}

with st.form("prediction_form"):

    input_columns = st.columns(2)

    for i, col in enumerate(feature_columns):

        with input_columns[i % 2]:

            label = col.replace("_", " ").title()

            if col in cat_columns:

                options = sorted(
                    df[col].dropna().unique().tolist(),
                    key=str
                )

                input_values[col] = st.selectbox(
                    label,
                    options
                )

            elif col == "Month":

                input_values[col] = st.selectbox(
                    "Month",
                    range(1, 13),
                    format_func=lambda x: pd.Timestamp(
                        2026, x, 1
                    ).strftime("%B")
                )

            elif col == "Year":

                input_values[col] = st.number_input(
                    "Year",
                    min_value=2000,
                    max_value=2100,
                    value=int(df[col].median())
                )

            elif col == "Marketing_Effect":

                input_values[col] = st.number_input(
                    label,
                    value=0.20
                )

            else:

                input_values[col] = st.number_input(
                    label,
                    value=float(df[col].median())
                )

    predict_button = st.form_submit_button(
        "🔍 Predict Demand",
        use_container_width=True
    )


# =========================
# PREDICTION RESULTS
# =========================

if predict_button:

    input_data = pd.DataFrame(
        [input_values],
        columns=feature_columns
    )

    input_scaled = preprocessor.transform(input_data)

    if hasattr(input_scaled, "toarray"):
        input_scaled = input_scaled.toarray()

    ml_prediction = float(
        ml_model.predict(input_scaled)[0]
    )

    dl_prediction = float(
        dl_model.predict(input_scaled, verbose=0)[0][0]
    )

    st.session_state["ml_prediction"] = ml_prediction
    st.session_state["dl_prediction"] = dl_prediction


# =========================
# DISPLAY RESULTS
# =========================

if "ml_prediction" in st.session_state:

    ml_prediction = st.session_state["ml_prediction"]
    dl_prediction = st.session_state["dl_prediction"]

    st.divider()

    st.subheader("🎯 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Machine Learning Prediction",
            f"{ml_prediction:,.2f}"
        )

    with col2:
        st.metric(
            "Deep Learning Prediction",
            f"{dl_prediction:,.2f}"
        )

    # Demand status and recommendations

    low = df["Demand"].quantile(0.33)
    high = df["Demand"].quantile(0.66)

    if ml_prediction < 0:

        st.warning("Negative demand prediction. Review the input values.")

    elif ml_prediction < low:

        st.info("📉 Low Expected Demand")

        st.markdown("### 💡 Recommendation")
        st.write("""
        - Reduce production to avoid extra inventory.
        - Offer discounts and promotional campaigns.
        - Avoid purchasing unnecessary stock.
        - Monitor sales before increasing production.
        """)

    elif ml_prediction < high:

        st.warning("📊 Moderate Expected Demand")

        st.markdown("### 💡 Recommendation")
        st.write("""
        - Maintain normal production levels.
        - Keep sufficient stock available.
        - Monitor daily sales and customer demand.
        - Avoid unnecessary increases in production.
        - Prepare to increase production if demand rises.
        """)

    else:

        st.success("📈 High Expected Demand")

        st.markdown("### 💡 Recommendation")
        st.write("""
        - Increase production to meet expected demand.
        - Arrange additional stock and raw materials.
        - Ensure enough staff and resources are available.
        - Monitor inventory to prevent stock shortages.
        - Prepare for increased customer orders.
        """)

    # =========================
    # PREDICTION GRAPH
    # =========================

    st.subheader("📊 Prediction Comparison")

    fig, ax = plt.subplots(figsize=(9, 4))

    fig.patch.set_facecolor("#080D1C")
    ax.set_facecolor("#141E33")

    models = [
        "Machine Learning",
        "Deep Learning"
    ]

    predictions = [
        ml_prediction,
        dl_prediction
    ]

    bars = ax.bar(
        models,
        predictions,
        width=0.45,
        color=["#3B82F6", "#8B5CF6"],
        edgecolor="none"
    )

    ax.set_ylabel("Predicted Demand", color="white")
    ax.set_title("ML vs DL Predicted Demand", color="white", pad=15)

    ax.tick_params(colors="white")
    ax.grid(axis="y", alpha=0.15)

    for spine in ax.spines.values():
        spine.set_color("#263653")

    for bar in bars:
        height = bar.get_height()

        ax.annotate(
            f"{height:,.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color="white",
            fontweight="bold"
        )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

else:

    st.info("Enter product information above and click Predict Demand.")


# =========================
# MODEL PERFORMANCE
# =========================

st.divider()

st.subheader("🤖 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Best ML Model",
    metrics["Best ML Model"]
)

col2.metric(
    "ML R² Score",
    f'{metrics["ML R2"]:.2%}'
)

col3.metric(
    "DL R² Score",
    f'{metrics["DL R2"]:.2%}'
)


# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "Demand Forecasting AI | Machine Learning & Deep Learning"
)
