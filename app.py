"""
app.py
Streamlit Web Application for Bank Customer Churn Prediction & Risk Scoring.

Modules:
  1. Customer Churn Risk Calculator
  2. Probability Distribution Visualization
  3. Feature Importance Dashboard
  4. What-If Scenario Simulator
"""

import pickle
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Bank Customer Churn Risk Intelligence",
    page_icon="🏦",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Load artifacts (cached so the app doesn't reload the model on every click)
# ----------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open("best_model_name.json", "r") as f:
        best_model_name = json.load(f)["best_model_name"]
    importance_df = pd.read_csv("feature_importance.csv")
    comparison_df = pd.read_csv("model_comparison.csv")
    test_preds = pd.read_csv("test_predictions.csv")
    return model, scaler, feature_cols, best_model_name, importance_df, comparison_df, test_preds


model, scaler, feature_cols, best_model_name, importance_df, comparison_df, test_preds = load_artifacts()

USES_SCALER = best_model_name == "Logistic Regression"


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Build a single-row dataframe matching the training feature schema
    from raw user inputs, applying the same feature engineering used in
    preprocessing.py."""
    balance = inputs["Balance"]
    salary = inputs["EstimatedSalary"]
    tenure = inputs["Tenure"]
    num_products = inputs["NumOfProducts"]
    is_active = inputs["IsActiveMember"]
    age = inputs["Age"]

    row = {
        "CreditScore": inputs["CreditScore"],
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": inputs["HasCrCard"],
        "IsActiveMember": is_active,
        "EstimatedSalary": salary,
        "BalanceSalaryRatio": (balance / salary) if salary else 0,
        "ProductDensity": num_products / (tenure + 1),
        "EngagementProductInteraction": is_active * num_products,
        "AgeTenureInteraction": age * tenure,
        "IsZeroBalance": 1 if balance == 0 else 0,
        "Geography_Germany": 1 if inputs["Geography"] == "Germany" else 0,
        "Geography_Spain": 1 if inputs["Geography"] == "Spain" else 0,
        "Gender_Male": 1 if inputs["Gender"] == "Male" else 0,
    }
    df = pd.DataFrame([row])[feature_cols]
    return df


def predict_probability(feature_row: pd.DataFrame) -> float:
    if USES_SCALER:
        scaled = scaler.transform(feature_row)
        return model.predict_proba(scaled)[0, 1]
    return model.predict_proba(feature_row)[0, 1]


def risk_band(prob: float) -> tuple:
    if prob < 0.3:
        return "Low Risk", "🟢"
    elif prob < 0.6:
        return "Medium Risk", "🟡"
    else:
        return "High Risk", "🔴"


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------

st.sidebar.title("🏦 Churn Intelligence")
st.sidebar.caption(f"Best model in use: **{best_model_name}**")
page = st.sidebar.radio(
    "Navigate",
    [
        "Churn Risk Calculator",
        "What-If Scenario Simulator",
        "Probability Distribution",
        "Feature Importance Dashboard",
        "Model Performance",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Predictive Modeling and Risk Scoring for Bank Customer Churn — "
    "European Central Bank project."
)

# ----------------------------------------------------------------------------
# Shared input widget (used by both the calculator and the simulator)
# ----------------------------------------------------------------------------

def customer_input_form(defaults=None, key_prefix="calc"):
    defaults = defaults or {}
    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.slider(
            "Credit Score", 300, 900, defaults.get("CreditScore", 650), key=f"{key_prefix}_cs"
        )
        geography = st.selectbox(
            "Geography", ["France", "Spain", "Germany"],
            index=["France", "Spain", "Germany"].index(defaults.get("Geography", "France")),
            key=f"{key_prefix}_geo",
        )
        gender = st.selectbox(
            "Gender", ["Male", "Female"],
            index=["Male", "Female"].index(defaults.get("Gender", "Male")),
            key=f"{key_prefix}_gender",
        )
    with col2:
        age = st.slider("Age", 18, 92, defaults.get("Age", 40), key=f"{key_prefix}_age")
        tenure = st.slider("Tenure (years with bank)", 0, 10, defaults.get("Tenure", 5), key=f"{key_prefix}_tenure")
        balance = st.number_input(
            "Account Balance (€)", 0.0, 300000.0, float(defaults.get("Balance", 50000.0)),
            step=1000.0, key=f"{key_prefix}_balance",
        )
    with col3:
        num_products = st.slider(
            "Number of Products", 1, 4, defaults.get("NumOfProducts", 2), key=f"{key_prefix}_products"
        )
        has_card = st.selectbox(
            "Has Credit Card?", ["Yes", "No"],
            index=0 if defaults.get("HasCrCard", 1) == 1 else 1,
            key=f"{key_prefix}_card",
        )
        is_active = st.selectbox(
            "Active Member?", ["Yes", "No"],
            index=0 if defaults.get("IsActiveMember", 1) == 1 else 1,
            key=f"{key_prefix}_active",
        )
        salary = st.number_input(
            "Estimated Salary (€)", 0.0, 300000.0, float(defaults.get("EstimatedSalary", 100000.0)),
            step=1000.0, key=f"{key_prefix}_salary",
        )

    return {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
    }


# ----------------------------------------------------------------------------
# Page 1: Churn Risk Calculator
# ----------------------------------------------------------------------------

if page == "Churn Risk Calculator":
    st.title("Customer Churn Risk Calculator")
    st.write(
        "Enter a customer's profile to get their real-time churn probability "
        "and risk classification."
    )

    inputs = customer_input_form(key_prefix="calc")

    if st.button("Calculate Churn Risk", type="primary"):
        feature_row = build_feature_row(inputs)
        prob = predict_probability(feature_row)
        band, emoji = risk_band(prob)

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob*100:.1f}%")
        c2.metric("Risk Category", f"{emoji} {band}")
        c3.metric("Retention Priority", "Immediate" if prob > 0.6 else ("Monitor" if prob > 0.3 else "Low"))

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Churn Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred" if prob > 0.6 else ("orange" if prob > 0.3 else "green")},
                "steps": [
                    {"range": [0, 30], "color": "#d4f7dc"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
            },
        ))
        st.plotly_chart(fig, use_container_width=True)

        if prob > 0.6:
            st.warning(
                "**Recommendation:** High churn risk. Consider a personalized retention "
                "offer, proactive outreach from a relationship manager, and a review of "
                "product fit / pricing."
            )
        elif prob > 0.3:
            st.info(
                "**Recommendation:** Moderate risk. Consider targeted engagement "
                "campaigns (e.g., cross-sell relevant product, activity nudges)."
            )
        else:
            st.success("**Recommendation:** Low risk. Maintain standard engagement.")

# ----------------------------------------------------------------------------
# Page 2: What-If Scenario Simulator
# ----------------------------------------------------------------------------

elif page == "What-If Scenario Simulator":
    st.title("What-If Scenario Simulator")
    st.write(
        "Start from a baseline customer profile, then adjust engagement and "
        "product values to see how churn probability changes in real time."
    )

    st.subheader("Baseline Profile")
    baseline_inputs = customer_input_form(key_prefix="base")
    baseline_prob = predict_probability(build_feature_row(baseline_inputs))

    st.markdown("---")
    st.subheader("Adjusted Scenario")
    st.caption("Modify the levers below (engagement & product usage) to simulate an intervention.")

    scenario_inputs = dict(baseline_inputs)
    col1, col2 = st.columns(2)
    with col1:
        scenario_inputs["NumOfProducts"] = st.slider(
            "Scenario: Number of Products", 1, 4, baseline_inputs["NumOfProducts"], key="sim_products"
        )
    with col2:
        scenario_active = st.selectbox(
            "Scenario: Active Member?", ["Yes", "No"],
            index=0 if baseline_inputs["IsActiveMember"] == 1 else 1,
            key="sim_active",
        )
        scenario_inputs["IsActiveMember"] = 1 if scenario_active == "Yes" else 0

    scenario_prob = predict_probability(build_feature_row(scenario_inputs))

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Churn Probability", f"{baseline_prob*100:.1f}%")
    c2.metric("Scenario Churn Probability", f"{scenario_prob*100:.1f}%")
    delta = (scenario_prob - baseline_prob) * 100
    c3.metric("Change", f"{delta:+.1f} pts", delta=f"{delta:+.1f} pts", delta_color="inverse")

    fig = go.Figure(data=[
        go.Bar(name="Baseline", x=["Churn Probability"], y=[baseline_prob * 100], marker_color="steelblue"),
        go.Bar(name="Scenario", x=["Churn Probability"], y=[scenario_prob * 100], marker_color="orange"),
    ])
    fig.update_layout(barmode="group", yaxis_title="Churn Probability (%)")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Page 3: Probability Distribution Visualization
# ----------------------------------------------------------------------------

elif page == "Probability Distribution":
    st.title("Churn Probability Distribution")
    st.write(
        "Distribution of predicted churn probabilities on the held-out test "
        "set, split by actual outcome."
    )

    fig = px.histogram(
        test_preds,
        x="ChurnProbability",
        color=test_preds["ActualChurn"].map({0: "Retained", 1: "Churned"}),
        nbins=40,
        barmode="overlay",
        opacity=0.7,
        labels={"color": "Actual Outcome", "ChurnProbability": "Predicted Churn Probability"},
        color_discrete_map={"Retained": "steelblue", "Churned": "crimson"},
    )
    st.plotly_chart(fig, use_container_width=True)

    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
    flagged = (test_preds["ChurnProbability"] >= threshold).sum()
    st.write(
        f"At a threshold of **{threshold:.2f}**, **{flagged}** of "
        f"{len(test_preds)} test customers ({flagged/len(test_preds)*100:.1f}%) "
        "would be flagged as churn risks."
    )

    st.subheader("Risk Segmentation")
    seg = pd.cut(
        test_preds["ChurnProbability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    seg_counts = seg.value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"])
    fig2 = px.pie(values=seg_counts.values, names=seg_counts.index,
                  color=seg_counts.index,
                  color_discrete_map={"Low Risk": "#28a745", "Medium Risk": "#ffc107", "High Risk": "#dc3545"})
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------------
# Page 4: Feature Importance Dashboard
# ----------------------------------------------------------------------------

elif page == "Feature Importance Dashboard":
    st.title("Feature Importance Dashboard")
    st.write(f"Global feature importance for the deployed model: **{best_model_name}**")

    fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Churn Drivers — Interpretation")
    top5 = importance_df.head(5)["Feature"].tolist()
    st.write(
        "The most influential features for churn prediction, in order, are: "
        + ", ".join(top5) + "."
    )
    st.dataframe(importance_df, use_container_width=True)

    try:
        with open("shap_values.pkl", "rb") as f:
            shap_data = pickle.load(f)
        st.subheader("SHAP Summary (sample of test customers)")
        st.caption(
            "SHAP values show how much each feature pushed an individual "
            "customer's prediction up (toward churn) or down (toward retention)."
        )
        import shap
        shap_values = shap_data["shap_values"]
        sample = shap_data["sample"]
        # Handle binary classifier shap output shapes
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs_shap = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({"Feature": sample.columns, "Mean |SHAP|": mean_abs_shap}) \
            .sort_values("Mean |SHAP|", ascending=False)
        fig3 = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h")
        st.plotly_chart(fig3, use_container_width=True)
    except FileNotFoundError:
        st.info("SHAP values not available. Re-run train_model.py with the `shap` package installed.")

# ----------------------------------------------------------------------------
# Page 5: Model Performance
# ----------------------------------------------------------------------------

elif page == "Model Performance":
    st.title("Model Comparison & Evaluation")
    st.write(
        "All candidate models were trained on a stratified train/test split "
        "and evaluated on the held-out test set."
    )
    st.dataframe(
        comparison_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="lightgreen"
        ),
        use_container_width=True,
    )

    fig = px.bar(
        comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric", barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"**Deployed model: {best_model_name}** (selected by highest ROC-AUC)")
