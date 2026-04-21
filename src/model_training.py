from sklearn.ensemble import RandomForestClassifier
import pickle
from data_preprocessing import load_and_preprocess
from model_evaluation import evaluate_model

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess()

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
evaluate_model(model, X_test, y_test)

# Save model
pickle.dump(model, open("models/churn_model.pkl", "wb"))

print("✅ Model trained, evaluated and saved!")