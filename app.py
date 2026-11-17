
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import hashlib
import io
from pathlib import Path

from src.data import generate_demo_data
from src.models import train_models
from src.uploaded_model import train_uploaded_model, explain_prediction

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

MODEL_STORAGE = Path(__file__).resolve().parent / "uploaded_fraud_model.joblib"


# -------------------- SAVE AND RESTORE --------------------

def save_uploaded_state():
    save_data = {
        "data": st.session_state.uploaded_data,
        "model": st.session_state.uploaded_model,
        "results": st.session_state.uploaded_results,
        "fingerprint": st.session_state.uploaded_file_id
    }

    temporary_file = MODEL_STORAGE.with_name(
        MODEL_STORAGE.stem + ".tmp" + MODEL_STORAGE.suffix
    )

    joblib.dump(save_data, temporary_file)
    temporary_file.replace(MODEL_STORAGE)


def restore_uploaded_state():
    if not MODEL_STORAGE.exists():
        return

    try:
        saved_data = joblib.load(MODEL_STORAGE)

        st.session_state.uploaded_data = saved_data.get("data")
        st.session_state.uploaded_model = saved_data.get("model")
        st.session_state.uploaded_results = saved_data.get("results")
        st.session_state.uploaded_file_id = saved_data.get("fingerprint")

        results = st.session_state.uploaded_results

        if isinstance(results, dict):
            outdated = (
                "y_test" not in results
                or "probabilities" not in results
            )
        else:
            outdated = results is not None

        st.session_state.uploaded_results_outdated = outdated

    except Exception as error:
        st.error(f"Could not restore saved model: {error}")


# -------------------- THRESHOLD METRICS --------------------

def calculate_threshold_metrics(y_true, probabilities, threshold):
    actual = pd.Series(y_true).reset_index(drop=True).astype(int)

    predicted = (
        pd.Series(probabilities).reset_index(drop=True) >= threshold
    ).astype(int)

    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    true_negative = int(((actual == 0) & (predicted == 0)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0 else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return {
        "Threshold": f"{threshold:.0%}",
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100,
        "True Positives": true_positive,
        "True Negatives": true_negative,
        "False Positives": false_positive,
        "False Negatives": false_negative
    }

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}
[data-testid="stSidebar"] {
    background-color: #101D32;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.main-title {
    color: #14243A;
    font-size: 32px;
    font-weight: 750;
}
.sub-title {
    color: #718096;
    font-size: 14px;
}
.metric-card {
    background: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #E5EAF1;
    box-shadow: 0 2px 8px rgba(20, 36, 58, 0.04);
}
.metric-label {
    color: #718096;
    font-size: 12px;
    font-weight: 700;
}
.metric-value {
    color: #14243A;
    font-size: 27px;
    font-weight: 750;
    margin-top: 8px;
}
.section-title {
    color: #14243A;
    font-size: 21px;
    font-weight: 700;
}
div.stButton > button {
    background-color: #1769AA;
    color: white;
    border: none;
    border-radius: 7px;
    padding: 9px 20px;
    font-weight: 600;
}
div.stButton > button:hover {
    background-color: #10548C;
    color: white;
}
</style>
""", unsafe_allow_html=True)
for key, default in {
    "analysis_history": [],
    "uploaded_data": None,
    "uploaded_model": None,
    "uploaded_results": None,
    "uploaded_file_id": None,
    "uploaded_results_outdated": False,
    "fraud_example_loaded": False,
    "fraud_example_values": None,
    "fraud_example_label": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


if st.session_state.uploaded_data is None and MODEL_STORAGE.exists():
    restore_uploaded_state()


# -------------------- DEMONSTRATION MODELS --------------------

@st.cache_resource
def load_system():
    demo_data = generate_demo_data()
    demo_models, demo_results = train_models(demo_data)
    return demo_data, demo_models, demo_results


with st.spinner("Loading FraudShield AI..."):
    data, models, results = load_system()


# -------------------- SIDEBAR --------------------

st.sidebar.markdown("# 🛡️ FraudShield AI")
st.sidebar.caption("Intelligent Transaction Security")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "Overview",
    "Fraud Detection",
    "Live Prediction",
    "Transactions",
    "Model Analytics",
    "Dataset Upload"
])

st.sidebar.markdown("---")
st.sidebar.caption("AI-powered fraud monitoring system")
st.sidebar.caption("Educational demonstration project")


# -------------------- DASHBOARD COUNTS --------------------

analysis_history = st.session_state.analysis_history

analysis_count = len(analysis_history)

detected_fraud = sum(
    1 for item in analysis_history
    if item["Prediction"] == "Fraud"
)

detected_genuine = analysis_count - detected_fraud

total = len(data)
fraud_count = int(data["Class"].sum())
normal_count = total - fraud_count

dashboard_total = total + analysis_count
dashboard_fraud = fraud_count + detected_fraud
dashboard_genuine = normal_count + detected_genuine

dashboard_fraud_rate = (
    dashboard_fraud / dashboard_total * 100
    if dashboard_total > 0 else 0
)
if page == "Overview":

    st.markdown(
        '<div class="main-title">Fraud Monitoring Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Monitor transactions and review model predictions in real time.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, "TOTAL TRANSACTIONS", f"{dashboard_total:,}"),
        (c2, "FRAUD / FLAGGED", f"{dashboard_fraud:,}"),
        (c3, "GENUINE / UNFLAGGED", f"{dashboard_genuine:,}"),
        (c4, "FRAUD / FLAGGED RATE", f"{dashboard_fraud_rate:.2f}%")
    ]

    for col, label, value in metrics:
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
            """, unsafe_allow_html=True)

    st.caption(
        f"Additional transactions analyzed this session: {analysis_count}"
    )

    left, right = st.columns([1.6, 1])

    with left:
        st.markdown(
            '<div class="section-title">Transaction Overview</div>',
            unsafe_allow_html=True
        )

        chart_data = pd.DataFrame({
            "Status": ["Genuine / Unflagged", "Fraud / Flagged"],
            "Count": [dashboard_genuine, dashboard_fraud]
        })

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.barplot(
            data=chart_data,
            x="Status",
            y="Count",
            hue="Status",
            legend=False,
            ax=ax,
            palette=["#2878C8", "#E34D59"]
        )

        ax.set_xlabel("")
        ax.set_ylabel("Number of transactions")
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown(
            '<div class="section-title">Risk Summary</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">FRAUD / FLAGGED TRANSACTIONS</div>
                <div class="metric-value" style="color:#E34D59">{dashboard_fraud:,}</div>
            </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">GENUINE / UNFLAGGED TRANSACTIONS</div>
                <div class="metric-value" style="color:#2878C8">{dashboard_genuine:,}</div>
            </div>
        """, unsafe_allow_html=True)

        st.info(
            "Dashboard figures combine synthetic demo labels and predictions made during this session."
        )

    st.markdown("### Recent Analyses")

    if analysis_history:
        st.table(pd.DataFrame(analysis_history).tail(5))
    else:
        st.info(
            "No transactions analyzed yet. Open Fraud Detection to test a transaction."
        )


# ==========================================================
# FRAUD DETECTION
# ==========================================================

elif page == "Fraud Detection":

    st.markdown(
        '<div class="main-title">Fraud Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Analyze a demonstration transaction using the demo ML and NLP models.</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.markdown("### Transaction Details")

        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input(
                "Transaction Amount (₹)",
                min_value=1.0,
                value=2500.0,
                step=100.0
            )

            hour = st.slider(
                "Transaction Hour",
                min_value=0,
                max_value=23,
                value=14
            )

        with col2:
            merchant = st.selectbox(
                "Merchant Category",
                ["Grocery", "Shopping", "Restaurant", "Travel", "Online"]
            )

            model_choice = st.selectbox(
                "Detection Model",
                ["Machine Learning", "NLP", "Hybrid"]
            )

        description = st.text_area(
            "Transaction Description",
            placeholder="Example: Online payment for electronic products"
        )

        analyze = st.button(
            "Analyze Transaction",
            use_container_width=True
        )

    if analyze:

        if not description.strip():
            st.warning("Please enter a transaction description.")

        else:

            transaction = pd.DataFrame({
                "Amount": [amount],
                "Time": [hour],
                "Merchant": [merchant],
                "Description": [description]
            })

            selected_model = models[model_choice]

            if model_choice == "NLP":
                prediction = selected_model.predict(
                    transaction["Description"]
                )[0]

                probabilities = selected_model.predict_proba(
                    transaction["Description"]
                )[0]

            else:
                prediction = selected_model.predict(transaction)[0]
                probabilities = selected_model.predict_proba(transaction)[0]

            fraud_index = list(selected_model.classes_).index(1)

            fraud_probability = float(probabilities[fraud_index])

            prediction_label = (
                "Fraud" if prediction == 1 else "Genuine"
            )

            result = {
                "Amount": amount,
                "Merchant": merchant,
                "Description": description,
                "Model": model_choice,
                "Prediction": prediction_label,
                "Fraud Probability": round(
                    fraud_probability * 100, 2
                )
            }

            st.session_state.analysis_history.append(result)

            if prediction == 1:
                st.error("Potential Fraudulent Transaction Detected")
            else:
                st.success("Transaction Classified as Genuine")

            r1, r2 = st.columns(2)

            with r1:
                st.metric(
                    "Fraud Probability",
                    f"{fraud_probability * 100:.2f}%"
                )

            with r2:
                st.metric("Selected Model", model_choice)

            st.progress(fraud_probability)

            st.caption(
                "This prediction uses synthetic educational data and is not a verified financial risk assessment."
            )

            st.markdown("### Prediction Details")
            st.json(result)

elif page == "Live Prediction":

    st.markdown(
        '<div class="main-title">Live Fraud Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Test the trained Random Forest model using transaction data.</div>',
        unsafe_allow_html=True
    )

    uploaded_data = st.session_state.uploaded_data
    uploaded_model = st.session_state.uploaded_model

    if uploaded_data is None or uploaded_model is None:

        st.warning(
            "Upload your credit card dataset and train the model first in Dataset Upload."
        )

        if uploaded_data is not None:
            st.info(
                "Your dataset is loaded, but no trained model is available. Open Dataset Upload and train it."
            )

    else:

        feature_columns = [
            col for col in uploaded_data.columns
            if col != "Class"
        ]

        non_numeric = [
            col for col in feature_columns
            if not pd.api.types.is_numeric_dtype(uploaded_data[col])
        ]

        if non_numeric:
            st.error(
                "Live prediction requires numerical feature columns: "
                + ", ".join(non_numeric)
            )

        elif uploaded_data[feature_columns].isna().any().any():
            st.error("Your dataset contains missing feature values.")

        else:

            st.markdown("### Test a Transaction")

            st.info(
                "Enter values manually or load an actual fraud example from your uploaded dataset."
            )

            threshold = st.slider(
                "Fraud Probability Threshold",
                min_value=0.01,
                max_value=0.99,
                value=0.50,
                step=0.01,
                format="%.2f"
            )

            st.caption(
                f"The model will flag a transaction as fraud when its fraud probability is at least {threshold:.0%}."
            )

            if st.button("Load Fraud Example", use_container_width=True):

                fraud_rows = uploaded_data[
                    uploaded_data["Class"] == 1
                ]

                if fraud_rows.empty:
                    st.error("No fraud records were found in the dataset.")

                else:

                    fraud_sample = fraud_rows.sample(n=1).iloc[0]

                    sample_values = {
                        feature: float(fraud_sample[feature])
                        for feature in feature_columns
                    }

                    st.session_state.fraud_example_values = sample_values
                    st.session_state.fraud_example_label = int(
                        fraud_sample["Class"]
                    )

                    st.session_state.fraud_example_loaded = True

                    for feature, value in sample_values.items():
                        st.session_state[f"live_{feature}"] = value

                    st.rerun()

            if st.session_state.fraud_example_loaded:
                st.success(
                    "A fraud-labelled transaction from your dataset has been loaded."
                )

                st.caption(
                    "This sample may have been included in model training. It is not an independent test-set evaluation."
                )

            with st.form("live_prediction_form"):

                transaction_values = {}
                input_columns = st.columns(3)

                for i, feature in enumerate(feature_columns):

                    input_key = f"live_{feature}"

                    if input_key not in st.session_state:
                        st.session_state[input_key] = float(
                            uploaded_data[feature].median()
                        )

                    with input_columns[i % 3]:
                        transaction_values[feature] = st.number_input(
                            feature,
                            format="%.6f",
                            key=input_key
                        )

                submitted = st.form_submit_button(
                    "Analyze Transaction",
                    use_container_width=True
                )

            if submitted:

                input_data = pd.DataFrame(
                    [transaction_values],
                    columns=feature_columns
                )

                try:

                    explanation = explain_prediction(
                        uploaded_model,
                        input_data,
                        threshold=threshold
                    )

                    prediction = explanation["prediction"]
                    probability = explanation["fraud_probability"]

                    st.markdown("### Prediction Results")

                    p1, p2 = st.columns(2)

                    with p1:
                        st.metric(
                            "Fraud Probability",
                            f"{probability * 100:.2f}%"
                        )

                    with p2:
                        st.metric(
                            "Model Prediction",
                            "Potential Fraud" if prediction == 1 else "Genuine"
                        )

                    st.metric(
                        "Selected Threshold",
                        f"{threshold * 100:.0f}%"
                    )

                    if prediction == 1:
                        st.error(
                            "The model has classified this transaction as potentially fraudulent."
                        )
                    else:
                        st.success(
                            "The model has classified this transaction as genuine."
                        )

                    st.progress(probability)

                    actual_label = None
                    example_values = st.session_state.fraud_example_values

                    if (
                        st.session_state.fraud_example_loaded
                        and example_values is not None
                    ):

                        same_sample = all(
                            abs(
                                transaction_values[feature]
                                - example_values[feature]
                            ) < 1e-9
                            for feature in feature_columns
                        )

                        if same_sample:
                            actual_label = st.session_state.fraud_example_label

                    if actual_label is not None:

                        st.markdown("### Actual Dataset Label")

                        if actual_label == 1:
                            st.write("Actual label: **Fraud**")
                        else:
                            st.write("Actual label: **Genuine**")

                        if actual_label == 1 and prediction == 0:
                            st.error(
                                "FALSE NEGATIVE: The dataset labels this transaction as fraud, but the model classified it as genuine."
                            )

                        elif actual_label == 0 and prediction == 1:
                            st.warning(
                                "FALSE POSITIVE: The dataset labels this transaction as genuine, but the model flagged it as fraud."
                            )

                        elif actual_label == prediction:
                            st.success(
                                "The prediction matches the transaction's dataset label."
                            )

                        st.caption(
                            "This comparison is against the supplied dataset label, not an independent banking verification."
                        )

                    st.markdown("### Why Does the Model Flag This Transaction?")

                    st.caption(
                        "These are the five features with the highest overall importance in the Random Forest model. They are not individual explanations of this prediction."
                    )

                    importance_data = explanation["top_features"]

                    st.markdown("#### Top 5 Important Features")

                    for _, row in importance_data.iterrows():
                        st.write(
                            f"**{row['Feature']}** — Value: {row['Value']:.4f} | Importance: {row['Importance']:.4f}"
                        )

                    st.caption(
                        "Feature importance shows overall model behaviour. It does not prove that a particular feature caused this prediction."
                    )

                    st.caption(
                        "Educational model prediction only. It is not a confirmed banking decision."
                    )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

elif page == "Transactions":

    st.markdown(
        '<div class="main-title">Transaction Records</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Explore the demonstration dataset and transactions analyzed during this session.</div>',
        unsafe_allow_html=True
    )

    record_type = st.radio(
        "Select Records",
        ["Demo Dataset", "Analyzed Transactions"],
        horizontal=True
    )

    if record_type == "Demo Dataset":

        display_data = data.copy()

        display_data["Status"] = display_data["Class"].map({
            0: "Genuine",
            1: "Fraud"
        })

        display_data = display_data.drop(columns=["Class"])

    else:
        display_data = (
            pd.DataFrame(analysis_history)
            if analysis_history else pd.DataFrame()
        )

    if not display_data.empty:

        if "Status" in display_data.columns:

            status_options = [
                "All"
            ] + list(display_data["Status"].unique())

            selected_status = st.selectbox(
                "Filter by Status",
                status_options
            )

            if selected_status != "All":
                display_data = display_data[
                    display_data["Status"] == selected_status
                ]

        st.dataframe(display_data, use_container_width=True)

        csv = display_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            data=csv,
            file_name="fraudshield_transactions.csv",
            mime="text/csv"
        )

    else:
        st.info("No analyzed transactions available yet.")

    if record_type == "Analyzed Transactions" and analysis_history:

        if st.button("Clear Analyzed Transaction History"):
            st.session_state.analysis_history = []
            st.rerun()
elif page == "Model Analytics":

    st.markdown(
        '<div class="main-title">Model Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Compare the performance of different fraud detection approaches.</div>',
        unsafe_allow_html=True
    )

    if not results:
        st.warning("No demonstration model results are available.")

    else:

        model_names = list(results.keys())

        selected_model = st.selectbox(
            "Select Model",
            model_names
        )

        model_result = results[selected_model]

        a, b, c, d = st.columns(4)

        metric_values = [
            (a, "ACCURACY", model_result["accuracy"]),
            (b, "PRECISION", model_result["precision"]),
            (c, "RECALL", model_result["recall"]),
            (d, "F1 SCORE", model_result["f1"])
        ]

        for col, label, value in metric_values:

            with col:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value * 100:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

        left, right = st.columns(2)

        with left:

            st.markdown("### Model Comparison")

            comparison_data = pd.DataFrame({
                "Model": model_names,
                "Accuracy": [
                    results[name]["accuracy"] * 100
                    for name in model_names
                ],
                "Precision": [
                    results[name]["precision"] * 100
                    for name in model_names
                ],
                "Recall": [
                    results[name]["recall"] * 100
                    for name in model_names
                ],
                "F1 Score": [
                    results[name]["f1"] * 100
                    for name in model_names
                ]
            })

            st.dataframe(
                comparison_data.round(2),
                use_container_width=True,
                hide_index=True
            )

        with right:

            st.markdown("### Confusion Matrix")

            fig, ax = plt.subplots(figsize=(5, 4))

            sns.heatmap(
                model_result["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Genuine", "Fraud"],
                yticklabels=["Genuine", "Fraud"],
                ax=ax
            )

            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")

            st.pyplot(fig)
            plt.close(fig)

        st.info(
            "The model comparison above uses the synthetic demonstration dataset. The uploaded Kaggle model is evaluated separately below."
        )

    st.write("")

    st.markdown("## Uploaded Dataset: Threshold Comparison")

    uploaded_results = st.session_state.uploaded_results

    if uploaded_results is None:

        st.info(
            "Upload a dataset and train the Random Forest model in Dataset Upload to see this comparison."
        )

    elif not isinstance(uploaded_results, dict):

        st.error("Saved model results are invalid. Please retrain the model.")

    elif (
        "y_test" not in uploaded_results
        or "probabilities" not in uploaded_results
    ):

        st.error(
            "Your saved results are outdated and do not contain test-set probabilities."
        )

        st.write(
            "Available saved result keys:",
            list(uploaded_results.keys())
        )

        st.warning(
            "Open Dataset Upload and click Train / Retrain Model on Uploaded Dataset."
        )
    else:
        y_true = uploaded_results["y_test"]
        probabilities = uploaded_results["probabilities"]
        if len(y_true) != len(probabilities):
            st.error(
                "Test labels and probabilities have different lengths. Please retrain the model."
            )
        elif len(probabilities) == 0:
            st.error(
                "No test-set probabilities were found. Please retrain the model."
            )
        else:
            thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
            threshold_rows = [
                calculate_threshold_metrics(
                    y_true,
                    probabilities,
                    threshold
                )
                for threshold in thresholds
            ]
            threshold_table = pd.DataFrame(threshold_rows)
            st.success(
                "Uploaded dataset test-set probabilities loaded successfully."
            )
            st.caption(
                "All five thresholds are evaluated on the same held-out test set. Only the classification threshold changes."
            )
            st.dataframe(
                threshold_table.round(2),
                use_container_width=True,
                hide_index=True
            )
            st.markdown("### Threshold Performance Chart")
            chart_data = threshold_table[
                ["Threshold", "Precision", "Recall", "F1 Score"]
            ].melt(
                id_vars="Threshold",
                var_name="Metric",
                value_name="Score"
            )
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(
                data=chart_data,
                x="Threshold",
                y="Score",
                hue="Metric",
                ax=ax
            )
            ax.set_xlabel("Fraud Probability Threshold")
            ax.set_ylabel("Score (%)")
            ax.set_ylim(0, 105)
            ax.legend(title="Evaluation Metric")
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("### Understanding the Results")
            st.write(
                "- **Precision:** Of all transactions flagged as fraud, how many were actually fraud?"
            )
            st.write(
                "- **Recall:** Of all actual fraud transactions, how many did the model detect?"
            )
            st.write(
                "- **F1 Score:** A combined measure of precision and recall."
            )
            st.write(
                "- **False Positives:** Genuine transactions incorrectly flagged as fraud."
            )

            st.write(
                "- **False Negatives:** Fraud transactions incorrectly classified as genuine."
            )

            st.info(
                "A lower threshold generally catches more fraud but can also create more false alarms. Choose a threshold based on the measured results and the cost of missed fraud versus false alarms."
            )

            st.caption(
                "These results are calculated from the uploaded dataset's held-out test set. They do not establish real-world banking performance."
            )
elif page == "Dataset Upload":
    st.markdown(
        '<div class="main-title">Dataset Upload</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Upload, inspect and train a fraud detection model using your transaction dataset.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Upload CSV Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="dataset_uploader"
    )

    st.caption(
        "Expected target column: Class. Use 0 for genuine and 1 for fraud."
    )

    if uploaded_file is not None:

        try:

            file_bytes = uploaded_file.getvalue()

            current_file_id = hashlib.sha256(
                file_bytes
            ).hexdigest()

            if current_file_id != st.session_state.uploaded_file_id:

                with st.spinner("Reading and validating dataset..."):

                    new_data = pd.read_csv(
                        io.BytesIO(file_bytes)
                    )

                if new_data.empty:
                    st.error("The uploaded CSV file is empty.")

                elif "Class" not in new_data.columns:
                    st.error("The dataset must contain a Class column.")

                elif new_data["Class"].isna().any():
                    st.error("The Class column contains missing values.")

                elif not new_data["Class"].isin([0, 1]).all():
                    st.error("The Class column must contain only 0 and 1.")

                elif set(new_data["Class"].unique()) != {0, 1}:
                    st.error(
                        "The dataset must contain both genuine and fraud records."
                    )

                elif new_data.drop(columns=["Class"]).isna().any().any():
                    st.error(
                        "Feature columns contain missing values."
                    )

                elif not all(
                    pd.api.types.is_numeric_dtype(new_data[col])
                    for col in new_data.columns
                    if col != "Class"
                ):
                    st.error(
                        "All transaction feature columns must be numerical."
                    )

                else:

                    st.session_state.uploaded_data = new_data
                    st.session_state.uploaded_model = None
                    st.session_state.uploaded_results = None
                    st.session_state.uploaded_results_outdated = False
                    st.session_state.uploaded_file_id = current_file_id

                    st.session_state.fraud_example_loaded = False
                    st.session_state.fraud_example_values = None
                    st.session_state.fraud_example_label = None

                    for key in list(st.session_state.keys()):
                        if key.startswith("live_"):
                            del st.session_state[key]

                    save_uploaded_state()

                    st.success(
                        "Dataset uploaded, validated and saved successfully."
                    )

            else:

                st.success(
                    "Dataset is already loaded. Your saved dataset is being used."
                )

        except Exception as error:

            st.error(
                f"Unable to read this CSV file: {error}"
            )
    uploaded_data = st.session_state.uploaded_data

    if uploaded_data is not None:

        total_rows = len(uploaded_data)

        fraud_rows = int(
            (uploaded_data["Class"] == 1).sum()
        )

        genuine_rows = int(
            (uploaded_data["Class"] == 0).sum()
        )

        fraud_rate = (
            fraud_rows / total_rows * 100
            if total_rows > 0 else 0
        )

        st.markdown("---")
        st.markdown("### Dataset Overview")

        u1, u2, u3, u4 = st.columns(4)

        upload_metrics = [
            (u1, "TOTAL RECORDS", f"{total_rows:,}"),
            (u2, "FRAUD RECORDS", f"{fraud_rows:,}"),
            (u3, "GENUINE RECORDS", f"{genuine_rows:,}"),
            (u4, "FRAUD RATE", f"{fraud_rate:.2f}%")
        ]

        for col, label, value in upload_metrics:

            with col:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                """, unsafe_allow_html=True)

        left, right = st.columns(2)

        with left:

            st.markdown("### Fraud Distribution")

            fig, ax = plt.subplots(figsize=(6, 4))

            sns.countplot(
                data=uploaded_data,
                x="Class",
                hue="Class",
                legend=False,
                palette=["#2878C8", "#E34D59"],
                ax=ax
            )

            ax.set_xticks([0, 1], ["Genuine", "Fraud"])
            ax.set_xlabel("")
            ax.set_ylabel("Number of transactions")

            st.pyplot(fig)
            plt.close(fig)

        with right:

            st.markdown("### Dataset Information")

            information = pd.DataFrame({
                "Property": [
                    "Number of rows",
                    "Number of columns",
                    "Missing values",
                    "Duplicate records"
                ],
                "Value": [
                    total_rows,
                    len(uploaded_data.columns),
                    int(uploaded_data.isna().sum().sum()),
                    int(uploaded_data.duplicated().sum())
                ]
            })
            st.table(information)
        st.markdown("### Dataset Preview")
        st.dataframe(
            uploaded_data.head(20),
            use_container_width=True
        )
        st.download_button(
            "Download Uploaded Dataset",
            data=uploaded_data.to_csv(index=False).encode("utf-8"),
            file_name="uploaded_transaction_dataset.csv",
            mime="text/csv"
        )
        st.markdown("---")
        st.markdown("### Train Fraud Detection Model")

        st.caption(
            "Train or retrain a Random Forest model using the uploaded dataset's numerical features."
        )

        feature_columns = [
            col for col in uploaded_data.columns
            if col != "Class"
        ]

        non_numeric = [
            col for col in feature_columns
            if not pd.api.types.is_numeric_dtype(uploaded_data[col])
        ]

        if non_numeric:

            st.error(
                "The uploaded model currently requires numerical feature columns: "
                + ", ".join(non_numeric)
            )

        elif uploaded_data[feature_columns].isna().any().any():

            st.error(
                "Feature columns contain missing values. Clean them before training."
            )

        elif uploaded_data["Class"].value_counts().min() < 2:

            st.error(
                "The dataset needs at least two examples of each class for a stratified train/test split."
            )

        else:

            current_results = st.session_state.uploaded_results

            if st.session_state.uploaded_results_outdated:

                st.warning(
                    "Your saved model contains older training results. Retraining is required for threshold comparison."
                )

            elif current_results is not None:

                if (
                    "y_test" in current_results
                    and "probabilities" in current_results
                ):
                    st.success(
                        "A trained model with test-set probabilities is saved."
                    )

                else:
                    st.warning(
                        "The saved results are incomplete. Please retrain the model."
                    )

            else:

                st.info(
                    "No trained uploaded-dataset model is currently saved."
                )

            if st.button(
                "Train / Retrain Model on Uploaded Dataset",
                use_container_width=True
            ):

                with st.spinner(
                    "Training Random Forest model. This may take several minutes..."
                ):

                    try:

                        trained_model, trained_results = train_uploaded_model(
                            st.session_state.uploaded_data
                        )

                        required_keys = [
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "confusion_matrix",
                            "report",
                            "y_test",
                            "probabilities"
                        ]

                        missing_keys = [
                            key for key in required_keys
                            if key not in trained_results
                        ]

                        if missing_keys:
                            raise ValueError(
                                "Training results are missing: "
                                + ", ".join(missing_keys)
                            )

                        if len(trained_results["y_test"]) != len(
                            trained_results["probabilities"]
                        ):
                            raise ValueError(
                                "Test labels and probabilities have different lengths."
                            )
                        st.session_state.uploaded_model = trained_model
                        st.session_state.uploaded_results = trained_results
                        st.session_state.uploaded_results_outdated = False
                        save_uploaded_state()
                        st.success(
                            "Model retrained and new results saved successfully!"
                        )
                        st.write(
                            "**Saved result keys:**",
                            list(trained_results.keys())
                        )

                        st.write(
                            "**Test labels available:**",
                            "y_test" in trained_results
                        )

                        st.write(
                            "**Probabilities available:**",
                            "probabilities" in trained_results
                        )

                    except Exception as error:

                        st.error(
                            f"Model training failed: {error}"
                        )

       
        uploaded_results = st.session_state.uploaded_results
        if isinstance(uploaded_results, dict):
            required_metric_keys = [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "confusion_matrix",
                "report"
            ]
            if all(
                key in uploaded_results
                for key in required_metric_keys
            ):
                st.markdown("---")
                st.markdown("### Uploaded Dataset Model Performance")
                m1, m2, m3, m4 = st.columns(4)
                model_metrics = [
                    (m1, "ACCURACY", uploaded_results["accuracy"]),
                    (m2, "PRECISION", uploaded_results["precision"]),
                    (m3, "RECALL", uploaded_results["recall"]),
                    (m4, "F1 SCORE", uploaded_results["f1"])
                ]
                for col, label, value in model_metrics:
                    with col:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{value * 100:.2f}%</div>
                            </div>
                        """, unsafe_allow_html=True)

                st.markdown("### Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(
                    uploaded_results["confusion_matrix"],
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Genuine", "Fraud"],
                    yticklabels=["Genuine", "Fraud"],
                    ax=ax
                )
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Actual Label")
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("### Classification Report")

                st.code(uploaded_results["report"])
                if (
                    "y_test" in uploaded_results
                    and "probabilities" in uploaded_results
                ):
                    st.success(
                        "Test-set labels and probabilities are saved. Threshold comparison is available in Model Analytics."
                    )
                else:
                    st.warning(
                        "This is an older evaluation. Retrain the model to enable threshold comparison."
                    )
                st.info(
                    "Evaluation uses a stratified train/test split. Results are from this Kaggle dataset and do not establish real-world banking performance."
                )
    else:
        st.info("Upload a CSV dataset to begin.")

st.markdown("---")

st.markdown(
    "<center style='color:#718096;font-size:12px;'>FraudShield AI | Credit Card Fraud Detection using Machine Learning and NLP</center>",
    unsafe_allow_html=True
)
