import os
import pandas as pd
import joblib
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.title("Online Shopping Purchase Prediction")

st.write("Machine Learning Assignment 2")

# Saved model files
model_files = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "KNN": "models/knn.joblib",
    "Naive Bayes": "models/naive_bayes.joblib",
    "SVM": "models/svm.joblib"
}

# Select model
choice = st.selectbox(
    "Select Machine Learning Model",
    list(model_files.keys())
)

# Upload test dataset
file = st.file_uploader("Upload Test Data", type="csv")

if file is not None:

    data = pd.read_csv(file)

    if "Revenue" not in data.columns:
        st.error("Revenue column is missing")
        st.stop()

    X = data.drop("Revenue", axis=1)

    y = data["Revenue"].astype(str).str.lower().map({
        "true": 1, "false": 0, "1": 1, "0": 0
    })

    if y.isna().any():
        st.error("Invalid Revenue values")
        st.stop()

    y = y.astype(int)

    # Load model
    if not os.path.exists(model_files[choice]):
        st.error("Train and save the models first")
        st.stop()

    model = joblib.load(model_files[choice])

    # Prediction
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    # Display evaluation
    st.subheader("Evaluation Results")

    st.write("Accuracy:", round(accuracy_score(y, pred), 4))
    st.write("AUC:", round(roc_auc_score(y, prob), 4))
    st.write("Precision:", round(precision_score(y, pred, zero_division=0), 4))
    st.write("Recall:", round(recall_score(y, pred, zero_division=0), 4))
    st.write("F1 Score:", round(f1_score(y, pred, zero_division=0), 4))
    st.write("MCC:", round(matthews_corrcoef(y, pred), 4))

    # Confusion matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, pred, labels=[0, 1])

    cm_table = pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    st.dataframe(cm_table)

    # Classification report
    st.subheader("Classification Report")
    st.text(classification_report(y, pred, zero_division=0))

    # Comparison of all models
    st.subheader("Comparison of All Five Models")

    results = []

    for name, path in model_files.items():

        if not os.path.exists(path):
            continue

        m = joblib.load(path)
        p = m.predict(X)
        pr = m.predict_proba(X)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y, p),
            "AUC": roc_auc_score(y, pr),
            "Precision": precision_score(y, p, zero_division=0),
            "Recall": recall_score(y, p, zero_division=0),
            "F1": f1_score(y, p, zero_division=0),
            "MCC": matthews_corrcoef(y, p)
        })

    st.dataframe(pd.DataFrame(results).round(4))

    # Show example predictions
    st.subheader("Sample Predictions")

    output = X.head(10).copy()
    output["Actual Revenue"] = y.iloc[:10].values
    output["Predicted Revenue"] = pred[:10]

    st.dataframe(output)

else:
    st.info("Upload the test_data.csv file to begin")
