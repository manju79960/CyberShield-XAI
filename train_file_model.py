# train_file_model.py
# Creates a prototype static-file cybersecurity model for CyberShield XAI

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"

DATASET_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATASET_DIR / "file_risk_dataset.csv"
MODEL_PATH = MODEL_DIR / "file_risk_model.pkl"

FEATURES = [
    "File_Size_KB",
    "Entropy",
    "Is_Executable",
    "Suspicious_String_Count",
    "URL_Count",
    "IP_Count",
    "Script_Indicator",
    "Macro_Indicator",
    "Archive_Indicator",
    "Extension_Risk"
]

def create_dataset(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n):
        size = np.clip(rng.lognormal(5.0, 1.1), 1, 10000)
        executable = int(rng.random() < 0.22)
        entropy = rng.uniform(3.0, 8.0)

        suspicious = np.clip(
            rng.poisson(2.0 + 7.0 * executable), 0, 40
        )
        url_count = np.clip(
            rng.poisson(1.0 + 3.0 * executable), 0, 25
        )
        ip_count = np.clip(
            rng.poisson(0.5 + 2.0 * executable), 0, 15
        )

        script = int(rng.random() < (0.15 + 0.55 * executable))
        macro = int(rng.random() < 0.12)
        archive = int(rng.random() < 0.20)
        extension_risk = int(executable or script)

        score = 0

        if entropy > 7.2:
            score += 2
        if executable:
            score += 3
        if suspicious >= 8:
            score += 3
        elif suspicious >= 4:
            score += 1
        if url_count >= 4:
            score += 1
        if ip_count >= 3:
            score += 1
        if script:
            score += 2
        if macro:
            score += 2
        if archive:
            score += 1
        if extension_risk:
            score += 2

        if rng.random() < 0.08:
            score += 1

        if score >= 8:
            risk = "High"
        elif score >= 4:
            risk = "Medium"
        else:
            risk = "Low"

        rows.append([
            size, entropy, executable, suspicious, url_count,
            ip_count, script, macro, archive, extension_risk, risk
        ])

    return pd.DataFrame(rows, columns=FEATURES + ["Risk_Level"])


def main():
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        print("Existing file-risk dataset found.")
    else:
        print("Creating file-risk dataset...")
        df = create_dataset()
        df.to_csv(DATASET_PATH, index=False)
        print("Dataset saved:", DATASET_PATH)

    X = df[FEATURES]
    y = df["Risk_Level"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n==============================")
    print("FILE RISK MODEL")
    print("==============================")
    print("Dataset shape:", df.shape)
    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("\nClasses:", list(encoder.classes_))
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=encoder.classes_
    ))

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance.to_string(index=False))

    package = {
        "model": model,
        "encoder": encoder,
        "features": FEATURES,
        "X_reference": X_train.reset_index(drop=True)
    }

    joblib.dump(package, MODEL_PATH)
    print("\nSaved:", MODEL_PATH)


if __name__ == "__main__":
    main()
