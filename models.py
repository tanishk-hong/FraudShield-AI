
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_models(df):

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numerical_features = ["Amount", "Time"]
    categorical_features = ["Merchant"]

    ml_preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ])

    hybrid_preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features),
        ("text", TfidfVectorizer(), "Description")
    ])

    ml_model = Pipeline([
        ("preprocessor", ml_preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ))
    ])

    nlp_model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ))
    ])

    hybrid_model = Pipeline([
        ("preprocessor", hybrid_preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ))
    ])

    ml_model.fit(X_train, y_train)
    nlp_model.fit(X_train["Description"], y_train)
    hybrid_model.fit(X_train, y_train)

    models = {
        "Machine Learning": ml_model,
        "NLP": nlp_model,
        "Hybrid": hybrid_model
    }

    results = {}

    for name, model in models.items():

        if name == "NLP":
            predictions = model.predict(X_test["Description"])
        else:
            predictions = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            )
        }

    return models, results
