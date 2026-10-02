"""InterpretAI: Explainable AI & Model Interpretability Studio.

Interactive XAI dashboard featuring SHAP local waterfall attributions,
Logistic Regression log-odds, Random Forest MDI, and Neural Net Permutation Importance.
"""

import streamlit as st
import numpy as np

from src.interpretai.config import InterpretAIConfig, get_default_config
from src.interpretai.engine import InterpretAIEngine

# Configure Streamlit Page
st.set_page_config(
    page_title="InterpretAI — XAI Model Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .shap-pos {
        background-color: #fee2e2;
        color: #991b1b;
        border-left: 4px solid #ef4444;
        padding: 0.4rem 0.8rem;
        margin-bottom: 0.4rem;
        border-radius: 0 6px 6px 0;
    }
    .shap-neg {
        background-color: #dbeafe;
        color: #1e40af;
        border-left: 4px solid #3b82f6;
        padding: 0.4rem 0.8rem;
        margin-bottom: 0.4rem;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    """Cache engine to maintain trained models in memory."""
    return InterpretAIEngine()

def main():
    engine = load_engine()
    pipe = engine.pipeline

    # Sidebar: Model and Applicant Inputs
    with st.sidebar:
        st.header("⚙️ XAI Model & Profile")
        model_name = st.selectbox(
            "Target Machine Learning Model:",
            ["Random Forest", "Logistic Regression", "Neural Network (MLP)"],
            index=0
        )

        st.markdown("---")
        st.subheader("Applicant Demographics")

        age = st.slider("Age", min_value=18, max_value=90, value=38)
        edu_num = st.slider("Education Years (Num)", min_value=1, max_value=16, value=13)
        hours = st.slider("Hours per Week", min_value=1, max_value=99, value=40)

        workclass_options = pipe.category_levels.get("WORKCLASS", ["Private", "State-gov", "Self-emp-not-inc"])
        workclass = st.selectbox("Workclass:", workclass_options, index=0)

        marital_options = pipe.category_levels.get("MARITAL_STATUS", ["Married-civ-spouse", "Never-married", "Divorced"])
        marital = st.selectbox("Marital Status:", marital_options, index=0)

        occ_options = pipe.category_levels.get("OCCUPATION", ["Exec-managerial", "Prof-specialty", "Sales", "Craft-repair"])
        occupation = st.selectbox("Occupation:", occ_options, index=0)

        rel_options = pipe.category_levels.get("RELATIONSHIP", ["Husband", "Wife", "Not-in-family", "Own-child"])
        relationship = st.selectbox("Relationship:", rel_options, index=0)

        race_options = pipe.category_levels.get("RACE", ["White", "Black", "Asian-Pac-Islander"])
        race = st.selectbox("Race:", race_options, index=0)

        sex_options = pipe.category_levels.get("SEX", ["Male", "Female"])
        sex = st.selectbox("Sex:", sex_options, index=0)

        applicant_profile = {
            "AGE": age,
            "EDUCATION_NUM": edu_num,
            "HOURS_PER_WEEK": hours,
            "WORKCLASS": workclass,
            "MARITAL_STATUS": marital,
            "OCCUPATION": occupation,
            "RELATIONSHIP": relationship,
            "RACE": race,
            "SEX": sex,
            "NATIONALITY": "United-States",
            "EDUCATION": "Bachelors"
        }

    # Main Header
    st.markdown('<div class="main-title">InterpretAI 🔍</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Production Explainable AI Suite — SHAP Local Attribution, Feature Importance & Auditing</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Individual Prediction & SHAP",
        "📊 Global Model Interpretability",
        "⚖️ Fairness & Bias Audit",
        "📈 Model Benchmarks (ROC-AUC)",
    ])

    # Tab 1: Local SHAP
    with tab1:
        st.subheader("Local Instance Prediction & Shapley Value Decomposition")
        shap_res = engine.explain_individual(applicant_profile, model_name=model_name, top_n=10)

        prob = shap_res["prediction_probability"]
        base_val = shap_res["base_value"]
        delta = shap_res["total_delta"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Income Probability (>50K)", f"{prob * 100:.1f}%")
        with col2:
            st.metric("Base Population Expectation E[f(x)]", f"{base_val * 100:.1f}%")
        with col3:
            st.metric("Net Model Delta", f"{delta * 100:+.1f}%")

        if prob >= 0.5:
            st.success(f"Outcome: **Likely >$50K/year** (Confidence: {prob*100:.1f}%)")
        else:
            st.info(f"Outcome: **Likely <=$50K/year** (Confidence: {(1.0-prob)*100:.1f}%)")

        st.markdown("---")
        st.subheader("Top Contributing Features (Local SHAP Waterfall)")
        st.caption("Game-theoretic Shapley values quantify each feature's contribution to moving the prediction from the base value to the final score.")

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### 🔺 Pushing Towards >$50K (Positive Attribution)")
            pos_found = False
            for item in shap_res["top_contributions"]:
                if item["shap_value"] > 0:
                    pos_found = True
                    st.markdown(
                        f'<div class="shap-pos">'
                        f'<b>{item["feature"]}</b>: <code>+{item["shap_value"]:.4f}</code><br/>'
                        f'<small>Increases odds towards high income</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            if not pos_found:
                st.write("No strong positive pushing features for this instance.")

        with c_right:
            st.markdown("#### 🔻 Pulling Towards <=$50K (Negative Attribution)")
            neg_found = False
            for item in shap_res["top_contributions"]:
                if item["shap_value"] < 0:
                    neg_found = True
                    st.markdown(
                        f'<div class="shap-neg">'
                        f'<b>{item["feature"]}</b>: <code>{item["shap_value"]:.4f}</code><br/>'
                        f'<small>Pulls prediction towards low income</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            if not neg_found:
                st.write("No strong negative pulling features for this instance.")

    # Tab 2: Global Interpretability
    with tab2:
        st.subheader("Global Feature Importance & Model Mechanics")
        st.markdown(f"Inspecting global explanations for **{model_name}**.")

        global_exp = engine.get_global_explanations(model_name=model_name, top_n=12)

        if "logistic" in model_name.lower():
            st.markdown("#### Logistic Regression Log-Odds Coefficients & Odds Ratios")
            st.write(f"Model Intercept (Bias): `{global_exp.get('intercept', 0.0)}`")

            rows = []
            for item in global_exp.get("top_features", []):
                rows.append({
                    "Feature": item["feature"],
                    "Coefficient (Log-Odds)": item["coefficient"],
                    "Odds Ratio exp(β)": item["odds_ratio"],
                    "Odds Impact (%)": f"{item['pct_change']:+0.1f}%",
                    "Direction": item["direction"]
                })
            st.table(rows)

        elif "random" in model_name.lower():
            st.markdown("#### Random Forest Gini Impurity (MDI) Importance")
            st.caption("Normalized sum of impurity reductions across all decision trees.")
            rows = []
            for item in global_exp.get("top_features", []):
                rows.append({
                    "Feature": item["feature"],
                    "Importance (MDI)": item["importance"],
                    "Relative Weight (%)": f"{item['percentage']}%"
                })
            st.table(rows)

        elif "neural" in model_name.lower():
            st.markdown("#### Neural Network (MLP) Permutation Feature Importance")
            st.caption("Accuracy degradation when feature column values are randomly shuffled.")
            rows = []
            for item in global_exp.get("top_features", []):
                rows.append({
                    "Feature": item["feature"],
                    "Accuracy Drop (Mean)": item["importance_mean"],
                    "Std Dev": item["importance_std"]
                })
            st.table(rows)

    # Tab 3: Fairness & Bias Audit
    with tab3:
        st.subheader("Algorithmic Fairness & Protected Attribute Audit")
        st.markdown("""
        Explainability is incomplete without **fairness verification**. We evaluate model predictions across demographic subgroups
        to ensure compliance with the **Four-Fifths (80%) Rule** for disparate impact.
        """)

        # Subgroup analysis on test set
        X_test = engine.X_test
        y_test = engine.y_test
        preds = engine.rf_model.predict(X_test)

        # Sex column index in one-hot features
        male_idx = pipe.feature_names.index("SEX_Male") if "SEX_Male" in pipe.feature_names else None
        female_idx = pipe.feature_names.index("SEX_Female") if "SEX_Female" in pipe.feature_names else None

        if male_idx is not None and female_idx is not None:
            male_mask = X_test[:, male_idx] == 1.0
            female_mask = X_test[:, female_idx] == 1.0

            rate_male = float(np.mean(preds[male_mask])) if np.sum(male_mask) > 0 else 0.0
            rate_female = float(np.mean(preds[female_mask])) if np.sum(female_mask) > 0 else 0.0
            disparity = (rate_female / rate_male) if rate_male > 0 else 1.0

            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                st.metric("Male High-Income Rate (>50K)", f"{rate_male * 100:.1f}%")
            with c_f2:
                st.metric("Female High-Income Rate (>50K)", f"{rate_female * 100:.1f}%")
            with c_f3:
                st.metric("Disparate Impact Ratio", f"{disparity:.2f}", help="Threshold >= 0.80 for 4/5ths rule")

            if disparity < 0.80:
                st.warning(f"⚠️ Disparate impact detected ({disparity:.2f} < 0.80). Reflects historical demographic wage gap in Census data. Mitigate using re-weighting or adversarial debiasing.")
            else:
                st.success("✅ Meets 80% Four-Fifths disparate impact threshold.")

    # Tab 4: Performance Benchmarks
    with tab4:
        st.subheader("Model Classification Benchmarks on Test Holdout")
        metrics = engine.evaluate_all_models()

        table_rows = []
        for name, m in metrics.items():
            table_rows.append({
                "Model": name,
                "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Precision": f"{m['precision'] * 100:.2f}%",
                "Recall": f"{m['recall'] * 100:.2f}%",
                "F1 Score": f"{m['f1_score'] * 100:.2f}%",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
            })
        st.table(table_rows)

if __name__ == "__main__":
    main()
