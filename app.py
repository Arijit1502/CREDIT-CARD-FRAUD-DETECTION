import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Project folder and saved model
PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_FOLDER / "fraud_model.joblib"

st.title("💳 Credit Card Fraud Detection")
st.write(
    "Upload transaction data to identify transactions that "
    "the machine learning model flags as potentially fraudulent."
)

st.warning(
    "This application provides predictions for educational purposes. "
    "It cannot guarantee that a transaction is fraudulent or legitimate."
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.info(
        "Check that fraud_model.joblib exists in the same folder as app.py."
    )
    st.stop()

st.success("Trained machine learning model loaded successfully!")

st.subheader("Upload Transaction Data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.write("### Preview of Uploaded Data")
        st.dataframe(data.head(10), use_container_width=True)

        # Remove the target column if the uploaded dataset includes it.
        input_data = data.drop(columns=["Class"], errors="ignore")

        # The trained model expects all original input features.
        expected_features = list(model.feature_names_in_)

        missing_columns = [
            column for column in expected_features
            if column not in input_data.columns
        ]

        if missing_columns:
            st.error(
                "The uploaded CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
            st.info(
                "For testing, upload the original creditcard.csv file "
                "or a CSV containing the same input columns."
            )
        else:
            # Use the exact feature order expected by the model.
            input_data = input_data[expected_features]

            if input_data.isnull().any().any():
                st.error(
                    "The uploaded data contains missing values. "
                    "Please clean the data before making predictions."
                )
            elif not all(
                pd.api.types.is_numeric_dtype(input_data[column])
                for column in input_data.columns
            ):
                st.error("All transaction feature columns must be numeric.")
            else:
                if st.button("Detect Fraud", type="primary"):
                    predictions = model.predict(input_data)

                    results = data.copy()

                    results["Fraud Prediction"] = [
                        "Potential Fraud" if prediction == 1
                        else "Likely Legitimate"
                        for prediction in predictions
                    ]

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_data)
                        results["Fraud Probability (%)"] = (
                            probabilities[:, 1] * 100
                        ).round(2)

                    st.subheader("Prediction Results")

                    fraud_count = int((predictions == 1).sum())
                    legitimate_count = int((predictions == 0).sum())

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Total Transactions", len(predictions))
                    col2.metric("Potential Fraud", fraud_count)
                    col3.metric("Likely Legitimate", legitimate_count)

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    csv_data = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Prediction Results",
                        data=csv_data,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )

    except Exception as error:
        st.error(f"An error occurred while processing the CSV: {error}")

else:
    st.info("Upload a CSV file to get started.")