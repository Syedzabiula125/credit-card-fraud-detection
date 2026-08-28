import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "creditcard.csv"
MODEL_PATH = BASE_DIR / "fraud_detection_model.keras"
SCALER_PATH = BASE_DIR / "fraud_scaler.joblib"
FEATURES_PATH = BASE_DIR / "fraud_features.joblib"


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(66, 133, 244, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(168, 85, 247, 0.10),
                transparent 30%
            ),
            #080b12;
    }

    /* Main content */
    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0c1019 0%,
                #090c13 100%
            );
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.07),
                rgba(255,255,255,0.025)
            );
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
        box-shadow:
            0 10px 30px rgba(0,0,0,0.18);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        background:
            linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6
            );
        color: white;
        font-weight: 700;
        padding: 0.7rem 1rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 10px 25px rgba(99,102,241,0.30);
    }

    /* Inputs */
    div[data-baseweb="input"],
    div[data-baseweb="select"],
    div[data-baseweb="textarea"] {
        border-radius: 12px;
    }

    /* Cards */
    .glass-card {
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.07),
                rgba(255,255,255,0.025)
            );
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow:
            0 15px 45px rgba(0,0,0,0.18);
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background:
            linear-gradient(
                90deg,
                #ffffff,
                #a78bfa
            );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .risk-low {
        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(34,197,94,0.30);
        color: #86efac;
        padding: 18px;
        border-radius: 16px;
        font-size: 1.4rem;
        font-weight: 700;
    }

    .risk-high {
        background: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.30);
        color: #fca5a5;
        padding: 18px;
        border-radius: 16px;
        font-size: 1.4rem;
        font-weight: 700;
    }

    .info-text {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_fraud_model():

    if not MODEL_PATH.exists():
        return None

    return load_model(MODEL_PATH)


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    if not SCALER_PATH.exists():
        return None

    return joblib.load(SCALER_PATH)


# ============================================================
# LOAD FEATURE LIST
# ============================================================

@st.cache_resource
def load_features():

    if FEATURES_PATH.exists():

        return joblib.load(FEATURES_PATH)

    df = load_dataset()

    if df is not None:
        return [c for c in df.columns if c != "Class"]

    return []


# ============================================================
# LOAD EVERYTHING
# ============================================================

df = load_dataset()
model = load_fraud_model()
scaler = load_scaler()
FEATURES = load_features()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:15px 0 25px 0;">
            <div style="font-size:3.5rem;">🛡️</div>
            <div style="
                font-size:1.5rem;
                font-weight:800;
                color:white;
            ">
                FraudShield AI
            </div>
            <div style="
                color:#94a3b8;
                font-size:0.85rem;
                margin-top:5px;
            ">
                Credit Card Fraud Detection
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### Navigation")

    page = st.radio(
        "",
        [
            "🏠 Dashboard",
            "🔍 Fraud Prediction",
            "📊 Model Performance",
            "⚙️ Model Information",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="info-text">
        <b>Model:</b> Neural Network<br>
        <b>Framework:</b> TensorFlow / Keras<br>
        <b>Balancing:</b> SMOTE<br>
        <b>Optimization:</b> Optuna
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="hero-title">Fraud Detection Command Center</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'AI-powered credit card transaction risk analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    if df is None:

        st.error(
            "creditcard.csv was not found in the project folder."
        )

        st.stop()

    total_transactions = len(df)

    fraud_count = int(
        df["Class"].sum()
    )

    legitimate_count = (
        total_transactions - fraud_count
    )

    fraud_rate = (
        fraud_count / total_transactions
    ) * 100

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Transactions",
        f"{total_transactions:,}",
    )

    c2.metric(
        "Fraud Transactions",
        f"{fraud_count:,}",
    )

    c3.metric(
        "Legitimate Transactions",
        f"{legitimate_count:,}",
    )

    c4.metric(
        "Fraud Rate",
        f"{fraud_rate:.3f}%",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Transaction Distribution"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Legitimate", "Fraud"],
                y=[
                    legitimate_count,
                    fraud_count,
                ],
                marker=dict(
                    color=[
                        "#6366f1",
                        "#ef4444",
                    ]
                ),
                text=[
                    legitimate_count,
                    fraud_count,
                ],
                textposition="auto",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        st.markdown(
            "### Fraud Percentage"
        )

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        "Legitimate",
                        "Fraud",
                    ],
                    values=[
                        legitimate_count,
                        fraud_count,
                    ],
                    hole=0.62,
                    marker=dict(
                        colors=[
                            "#6366f1",
                            "#ef4444",
                        ]
                    ),
                )
            ]
        )

        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="glass-card">
            <h3>🔐 Security Insight</h3>
            <p class="info-text">
            Credit card fraud is a highly imbalanced classification
            problem. The project addresses this using SMOTE during
            training and evaluates the neural network using
            precision, recall and F1-score rather than relying only
            on accuracy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FRAUD PREDICTION
# ============================================================

elif page == "🔍 Fraud Prediction":

    st.markdown(
        '<div class="hero-title">Transaction Risk Scanner</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Enter transaction information to estimate fraud probability.'
        '</div>',
        unsafe_allow_html=True,
    )

    if model is None or scaler is None:

        st.error(
            "Model or scaler not found. "
            "Save fraud_detection_model.keras and "
            "fraud_scaler.joblib from your notebook first."
        )

        st.stop()

    if not FEATURES:

        st.error(
            "Feature list could not be loaded."
        )

        st.stop()

    with st.form("fraud_form"):

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### 💳 Transaction Details"
        )

        values = {}

        # Show inputs in 3-column layout
        for start in range(
            0,
            len(FEATURES),
            3,
        ):

            columns = st.columns(3)

            for j, col in enumerate(columns):

                index = start + j

                if index >= len(FEATURES):
                    break

                feature = FEATURES[index]

                values[feature] = col.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                    key=f"input_{feature}",
                )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        submitted = st.form_submit_button(
            "🚨 Analyze Transaction",
            use_container_width=True,
        )

    if submitted:

        try:

            input_df = pd.DataFrame(
                [[
                    values[feature]
                    for feature in FEATURES
                ]],
                columns=FEATURES,
            )

            # Correct preprocessing:
            # use the scaler fitted during training
            scaled_input = scaler.transform(
                input_df
            )

            probability = float(
                model.predict(
                    scaled_input,
                    verbose=0,
                )[0][0]
            )

            fraud_probability = probability * 100

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            if probability >= 0.5:

                st.markdown(
                    f"""
                    <div class="risk-high">
                        🚨 HIGH FRAUD RISK
                        <br>
                        Fraud Probability:
                        {fraud_probability:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    f"""
                    <div class="risk-low">
                        ✅ LOW FRAUD RISK
                        <br>
                        Fraud Probability:
                        {fraud_probability:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ------------------------------------------------
            # PROBABILITY GAUGE
            # ------------------------------------------------

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=fraud_probability,
                    number={
                        "suffix": "%",
                    },
                    title={
                        "text": "Fraud Probability",
                    },
                    gauge={
                        "axis": {
                            "range": [
                                0,
                                100,
                            ]
                        },
                        "bar": {
                            "color": (
                                "#ef4444"
                                if probability >= 0.5
                                else "#22c55e"
                            )
                        },
                        "steps": [
                            {
                                "range": [
                                    0,
                                    30,
                                ],
                                "color": (
                                    "rgba(34,197,94,0.18)"
                                ),
                            },
                            {
                                "range": [
                                    30,
                                    70,
                                ],
                                "color": (
                                    "rgba(234,179,8,0.18)"
                                ),
                            },
                            {
                                "range": [
                                    70,
                                    100,
                                ],
                                "color": (
                                    "rgba(239,68,68,0.18)"
                                ),
                            },
                        ],
                    },
                )
            )

            fig.update_layout(
                template="plotly_dark",
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            # ------------------------------------------------
            # TRANSACTION SUMMARY
            # ------------------------------------------------

            st.markdown(
                "### Transaction Summary"
            )

            st.dataframe(
                input_df.T.rename(
                    columns={
                        0: "Value",
                    }
                ),
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.exception(error)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="hero-title">Model Performance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Evaluation of the trained neural network on unseen transactions'
        '</div>',
        unsafe_allow_html=True,
    )

    if (
        model is None
        or scaler is None
        or df is None
    ):

        st.error(
            "Required model, scaler or dataset is missing."
        )

        st.stop()

    X = df.drop(
        "Class",
        axis=1,
    )

    y = df["Class"]

    # Same test split as notebook
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    predictions = model.predict(
        X_test_scaled,
        verbose=0,
    ).ravel()

    y_pred = (
        predictions >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{accuracy:.2%}",
    )

    c2.metric(
        "Precision",
        f"{precision:.2%}",
    )

    c3.metric(
        "Recall",
        f"{recall:.2%}",
    )

    c4.metric(
        "F1 Score",
        f"{f1:.2%}",
    )

    c5.metric(
        "ROC-AUC",
        f"{roc_auc:.4f}",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
        )

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[
                    "Predicted Legitimate",
                    "Predicted Fraud",
                ],
                y=[
                    "Actual Legitimate",
                    "Actual Fraud",
                ],
                colorscale=[
                    [0, "#111827"],
                    [1, "#6366f1"],
                ],
                text=cm,
                texttemplate="%{text}",
            )
        )

        fig_cm.update_layout(
            template="plotly_dark",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    with col2:

        st.markdown(
            "### ROC Curve"
        )

        fpr, tpr, _ = roc_curve(
            y_test,
            predictions,
        )

        fig_roc = go.Figure()

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Model AUC = {roc_auc:.4f}",
                line=dict(
                    width=3,
                    color="#8b5cf6",
                ),
            )
        )

        fig_roc.add_trace(
            go.Scatter(
                x=[
                    0,
                    1,
                ],
                y=[
                    0,
                    1,
                ],
                mode="lines",
                name="Random Classifier",
                line=dict(
                    dash="dash",
                    color="#64748b",
                ),
            )
        )

        fig_roc.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig_roc,
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="glass-card">
            <h3>Why Recall Matters</h3>
            <p class="info-text">
            In fraud detection, missing a fraudulent transaction
            can be more costly than incorrectly flagging a legitimate
            transaction. Therefore, recall, precision and F1-score
            should be considered together rather than relying only
            on accuracy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "⚙️ Model Information":

    st.markdown(
        '<div class="hero-title">Model Architecture</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Neural network architecture and project pipeline'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="glass-card">

            <h3>🧠 Neural Network</h3>

            <p class="info-text">

            <b>Input:</b> Credit card transaction features

            <br><br>

            <b>Hidden Layer 1:</b> 37 neurons — ReLU

            <br>

            Batch Normalization + Dropout

            <br><br>

            <b>Hidden Layer 2:</b> 27 neurons — ReLU

            <br>

            Batch Normalization + Dropout

            <br><br>

            <b>Output:</b> 1 neuron — Sigmoid

            <br><br>

            <b>Optimizer:</b> RMSprop

            <br>

            <b>Loss:</b> Binary Crossentropy

            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="glass-card">

            <h3>⚙️ ML Pipeline</h3>

            <p class="info-text">

            Dataset

            ↓

            Train / Validation / Test Split

            ↓

            StandardScaler

            ↓

            SMOTE

            ↓

            Neural Network

            ↓

            Optuna Hyperparameter Optimization

            ↓

            Final Model

            ↓

            Streamlit Deployment

            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "### Project Statistics"
    )

    if df is not None:

        stat1, stat2, stat3 = st.columns(3)

        stat1.metric(
            "Features",
            len(
                [
                    c
                    for c in df.columns
                    if c != "Class"
                ]
            ),
        )

        stat2.metric(
            "Dataset Records",
            f"{len(df):,}",
        )

        stat3.metric(
            "Fraud Cases",
            f"{int(df['Class'].sum()):,}",
        )

    st.markdown(
        """
        <div class="glass-card">

        <h3>🎯 Project Goal</h3>

        <p class="info-text">

        The objective is to identify potentially fraudulent
        credit card transactions using a neural-network-based
        binary classification system while addressing the
        strong class imbalance present in fraud detection.

        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )
