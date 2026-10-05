import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ----------------------------------
# 1. Load Dataset
# ----------------------------------

df = pd.read_csv("dataset/cybersecurity_risk.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# ----------------------------------
# 2. Separate Features and Target
# ----------------------------------

X = df.drop("Risk_Level", axis=1)
y = df["Risk_Level"]

print("\nFeatures:")
print(X.columns.tolist())

# ----------------------------------
# 3. Encode Target
# ----------------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nRisk classes:")
print(encoder.classes_)

# ----------------------------------
# 4. Train-Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ----------------------------------
# 5. Random Forest Model
# ----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed!")

# ----------------------------------
# 6. Prediction
# ----------------------------------

y_pred = model.predict(X_test)

# ----------------------------------
# 7. Evaluation
# ----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------
# 8. Feature Importance
# ----------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# ----------------------------------
# 9. Save Model
# ----------------------------------

model_data = {
    "model": model,
    "encoder": encoder,
    "features": list(X.columns)
}

joblib.dump(
    model_data,
    "model/risk_model.pkl"
)

print("\nModel saved successfully!")
print("Location: model/risk_model.pkl")