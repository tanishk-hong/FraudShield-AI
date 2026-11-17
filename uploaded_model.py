
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train_uploaded_model(df):
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")
    if df["Class"].isna().any():
        raise ValueError("Class column contains missing values.")
    if set(df["Class"].unique()) != {0, 1}:
        raise ValueError("Class must contain both 0 and 1.")

    X = df.drop("Class", axis=1)
    y = df["Class"].astype(int)

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise ValueError("All features must be numerical.")
    if X.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    fraud_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, fraud_index]

    predictions = (probabilities >= 0.50).astype(int)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "report": classification_report(y_test, predictions, zero_division=0),
        "threshold": 0.50,
        "y_test": y_test,
        "probabilities": probabilities
    }

    return model, results

def explain_prediction(model, transaction, threshold=0.50):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if list(transaction.columns) != list(model.feature_names_in_):
        raise ValueError("Transaction features do not match the trained model.")

    probabilities = model.predict_proba(transaction)[0]
    fraud_index = list(model.classes_).index(1)
    fraud_probability = float(probabilities[fraud_index])
    prediction = int(fraud_probability >= threshold)

    feature_importance = pd.DataFrame({
        "Feature": transaction.columns,
        "Value": transaction.iloc[0].values,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return {
        "fraud_probability": fraud_probability,
        "prediction": prediction,
        "threshold": threshold,
        "top_features": feature_importance.head(5)
    }
