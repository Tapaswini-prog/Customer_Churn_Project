import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess():

    # 🔥 Simple & correct CSV loading
    data = pd.read_csv("data/raw/churn.csv")

    # 🔹 Clean column names
    data.columns = data.columns.str.strip()

    print("Columns:", data.columns)

    # 🔹 Drop customerID if exists
    if "customerID" in data.columns:
        data.drop("customerID", axis=1, inplace=True)

    # 🔹 Fix TotalCharges (important for churn dataset)
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # 🔹 Fill missing values
    data.fillna(data.mean(numeric_only=True), inplace=True)

    # 🔹 Encode categorical columns
    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # 🔹 Handle target column safely
    if "Churn" in data.columns:
        target_col = "Churn"
    elif "churn" in data.columns:
        target_col = "churn"
    elif "Exited" in data.columns:
        target_col = "Exited"
    else:
        raise Exception("❌ Target column not found in dataset")

    # 🔹 Split features & target
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # 🔹 Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✅ Preprocessing done")

    return X_train, X_test, y_train, y_test