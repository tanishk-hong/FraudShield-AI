
import pandas as pd
import numpy as np


def generate_demo_data(n=5000):
    np.random.seed(42)

    fraud = np.random.choice([0, 1], n, p=[0.95, 0.05])

    amount = np.where(
        fraud == 1,
        np.random.uniform(5000, 50000, n),
        np.random.uniform(50, 5000, n)
    )

    hour = np.random.randint(0, 24, n)

    merchants = np.random.choice(
        ["Grocery", "Shopping", "Restaurant", "Travel", "Online"],
        n
    )

    normal_text = [
        "regular grocery purchase",
        "monthly shopping payment",
        "restaurant bill payment",
        "normal travel booking",
        "regular online purchase"
    ]

    fraud_text = [
        "unusual high value online transaction",
        "suspicious international payment",
        "unknown merchant transaction",
        "multiple unusual payment attempts",
        "unrecognized card purchase"
    ]

    descriptions = [
        np.random.choice(fraud_text) if f == 1
        else np.random.choice(normal_text)
        for f in fraud
    ]

    return pd.DataFrame({
        "Amount": amount,
        "Time": hour,
        "Merchant": merchants,
        "Description": descriptions,
        "Class": fraud
    })
