# app.py
# CyberShield XAI
# Manual cybersecurity assessment + static file analysis + SHAP + LIME

import re
import io
import zipfile
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

from lime.lime_tabular import LimeTabularExplainer


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="CyberShield XAI",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

FILE_MODEL_PATH = MODEL_DIR / "file_risk_model.pkl"
MANUAL_MODEL_PATH = MODEL_DIR / "risk_model.pkl"


# =========================================================
# FILE FEATURES
# =========================================================

FILE_FEATURES = [
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

EXECUTABLE_EXTENSIONS = {
    ".exe", ".dll", ".scr", ".msi", ".com", ".bat", ".cmd",
    ".ps1", ".vbs", ".js", ".jar", ".elf", ".bin", ".sh"
}

SCRIPT_EXTENSIONS = {
    ".py", ".js", ".vbs", ".ps1", ".bat", ".cmd", ".sh",
    ".php", ".jsp"
}

ARCHIVE_EXTENSIONS = {
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"
}

HIGH_RISK_EXTENSIONS = EXECUTABLE_EXTENSIONS

SUSPICIOUS_TERMS = [
    b"powershell", b"cmd.exe", b"wscript", b"cscript",
    b"rundll32", b"regsvr32", b"mshta", b"download",
    b"invoke-webrequest", b"invoke-expression", b"base64",
    b"payload", b"reverse shell", b"keylogger", b"ransom",
    b"encrypt", b"bitcoin", b"credential", b"password",
    b"stealer", b"trojan", b"malware", b"autorun"
]

URL_PATTERN = re.compile(rb"https?://[^\s\"'<>]+", re.I)
IP_PATTERN = re.compile(rb"\b(?:\d{1,3}\.){3}\d{1,3}\b")


# =========================================================
# FILE ANALYSIS FUNCTIONS
# =========================================================

def calculate_entropy(data):
    if not data:
        return 0.0

    values = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(values, minlength=256)
    probabilities = counts[counts > 0] / len(values)

    return float(-(probabilities * np.log2(probabilities)).sum())


def archive_has_suspicious_file(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as z:
            for name in z.namelist():
                name = name.lower()

                if any(
                    name.endswith(ext)
                    for ext in EXECUTABLE_EXTENSIONS | SCRIPT_EXTENSIONS
                ):
                    return 1
    except Exception:
        pass

    return 0


def extract_file_features(uploaded_file):
    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()

    original_data = uploaded_file.getvalue()

    # Analyze at most 5 MB for content-based features.
    data = original_data[:5 * 1024 * 1024]
    lower_data = data.lower()

    file_size_kb = len(original_data) / 1024
    entropy = calculate_entropy(data)

    is_executable = int(
        extension in EXECUTABLE_EXTENSIONS
        or data.startswith(b"MZ")
        or data.startswith(b"\x7fELF")
    )

    suspicious_string_count = sum(
        lower_data.count(term) for term in SUSPICIOUS_TERMS
    )

    url_count = len(URL_PATTERN.findall(data))
    ip_count = len(IP_PATTERN.findall(data))

    script_indicator = int(
        extension in SCRIPT_EXTENSIONS
        or b"<script" in lower_data
        or b"powershell" in lower_data
        or b"javascript" in lower_data
        or b"#!/bin/bash" in lower_data
        or b"#!/usr/bin/python" in lower_data
    )

    macro_indicator = int(
        b"vba" in lower_data
        or b"macros" in lower_data
        or b"autoopen" in lower_data
        or b"document_open" in lower_data
    )

    archive_indicator = int(
        extension in ARCHIVE_EXTENSIONS
        or data.startswith(b"PK\x03\x04")
        or data.startswith(b"Rar!")
    )

    if archive_indicator:
        suspicious_string_count += archive_has_suspicious_file(data)

    extension_risk = int(extension in HIGH_RISK_EXTENSIONS)

    return {
        "File_Size_KB": round(file_size_kb, 3),
        "Entropy": round(entropy, 4),
        "Is_Executable": is_executable,
        "Suspicious_String_Count": suspicious_string_count,
        "URL_Count": url_count,
        "IP_Count": ip_count,
        "Script_Indicator": script_indicator,
        "Macro_Indicator": macro_indicator,
        "Archive_Indicator": archive_indicator,
        "Extension_Risk": extension_risk
    }


def sha256(data):
    return hashlib.sha256(data).hexdigest()


# =========================================================
# LOAD FILE MODEL
# =========================================================

@st.cache_resource
def load_file_model():
    if not FILE_MODEL_PATH.exists():
        return None

    return joblib.load(FILE_MODEL_PATH)


file_package = load_file_model()


# =========================================================
# SHAP
# =========================================================

def get_shap_dataframe(model, input_df, features):
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(input_df)

    predicted_class = int(model.predict(input_df)[0])

    if isinstance(raw, list):
        values = np.asarray(raw[predicted_class])[0]

    else:
        arr = np.asarray(raw)

        if arr.ndim == 2:
            values = arr[0]

        elif arr.ndim == 3:
            if arr.shape[0] == 1:
                values = arr[0, :, predicted_class]
            elif arr.shape[1] == 1:
                values = arr[predicted_class, 0, :]
            else:
                raise ValueError(f"Unsupported SHAP shape: {arr.shape}")
        else:
            raise ValueError(f"Unsupported SHAP shape: {arr.shape}")

    return pd.DataFrame({
        "Feature": features,
        "SHAP": values,
        "Absolute_SHAP": np.abs(values)
    }).sort_values("Absolute_SHAP", ascending=False)


# =========================================================
# LIME
# =========================================================

def get_lime_dataframe(model, encoder, reference, input_df, features):
    explainer = LimeTabularExplainer(
        reference.values,
        feature_names=features,
        class_names=list(encoder.classes_),
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    def predict_proba(x):
        return model.predict_proba(np.asarray(x))

    predicted_class = int(model.predict(input_df)[0])

    explanation = explainer.explain_instance(
        input_df.iloc[0].values.astype(float),
        predict_proba,
        num_features=len(features),
        labels=(predicted_class,)
    )

    try:
        values = explanation.as_list(label=predicted_class)
    except Exception:
        values = explanation.as_list()

    return pd.DataFrame(
        [
            {
                "Feature": text,
                "LIME": float(weight),
                "Absolute_LIME": abs(float(weight))
            }
            for text, weight in values
        ]
    ).sort_values("Absolute_LIME", ascending=False)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🛡️ CyberShield")
    st.caption("Explainable AI Security Platform")

    st.divider()

    mode = st.radio(
        "📌 Analysis Mode",
        [
            "📁 File Risk Analysis",
            "🎛️ Manual Security Assessment"
        ]
    )

    st.divider()

    st.markdown("### 🔐 Safety")

    st.info(
        "Files are analyzed using static characteristics only. "
        "CyberShield never executes uploaded files."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    "# 🛡️ Cybersecurity Risk Intelligence"
)

st.write(
    "Explainable AI platform for predicting and understanding cybersecurity risks."
)

st.divider()


# =========================================================
# FILE RISK ANALYSIS
# =========================================================

if mode == "📁 File Risk Analysis":

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🤖 AI Model", "Random Forest")
    c2.metric("🔎 XAI", "SHAP + LIME")
    c3.metric("📊 Features", len(FILE_FEATURES))
    c4.metric("🔐 Risk Levels", 3)

    st.header("📁 Suspicious File Analysis")

    if file_package is None:
        st.error(
            "file_risk_model.pkl is missing. Run this first:\n\n"
            "python train_file_model.py"
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a file for static analysis",
        type=None
    )

    if uploaded_file is not None:

        raw_data = uploaded_file.getvalue()
        features_dict = extract_file_features(uploaded_file)

        input_df = pd.DataFrame(
            [[features_dict[f] for f in FILE_FEATURES]],
            columns=FILE_FEATURES
        )

        st.success(f"File received: {uploaded_file.name}")

        a, b, c = st.columns(3)

        a.metric(
            "File Size",
            f"{len(raw_data) / 1024:.2f} KB"
        )

        b.metric(
            "Entropy",
            f"{features_dict['Entropy']:.2f}"
        )

        c.metric(
            "SHA-256",
            sha256(raw_data)[:16] + "..."
        )

        st.subheader("🔬 Extracted Security Features")

        st.dataframe(
            input_df.T.rename(columns={0: "Value"}),
            width="stretch"
        )

        if st.button(
            "🔍 ANALYZE SECURITY RISK",
            type="primary",
            width="stretch"
        ):

            model = file_package["model"]
            encoder = file_package["encoder"]
            features = file_package["features"]
            reference = file_package["X_reference"]

            prediction_encoded = int(model.predict(input_df)[0])
            probabilities = model.predict_proba(input_df)[0]

            prediction = encoder.inverse_transform(
                [prediction_encoded]
            )[0]

            confidence = probabilities[prediction_encoded]

            # -----------------------------------------
            # Risk result
            # -----------------------------------------

            st.subheader("🎯 Risk Assessment Result")

            if str(prediction).lower() == "high":
                st.error(
                    f"🔴 HIGH SECURITY RISK — "
                    f"{confidence * 100:.1f}% confidence"
                )
            elif str(prediction).lower() == "medium":
                st.warning(
                    f"🟠 MEDIUM SECURITY RISK — "
                    f"{confidence * 100:.1f}% confidence"
                )
            else:
                st.success(
                    f"🟢 LOW SECURITY RISK — "
                    f"{confidence * 100:.1f}% confidence"
                )

            # -----------------------------------------
            # Probabilities
            # -----------------------------------------

            st.subheader("📊 Risk Probability")

            probability_df = pd.DataFrame({
                "Risk Level": encoder.classes_,
                "Probability (%)": probabilities * 100
            })

            st.bar_chart(
                probability_df.set_index("Risk Level")
            )

            # -----------------------------------------
            # SHAP
            # -----------------------------------------

            st.subheader("🔎 SHAP Explanation")

            try:
                shap_df = get_shap_dataframe(
                    model, input_df, features
                )

                st.dataframe(
                    shap_df[
                        ["Feature", "SHAP", "Absolute_SHAP"]
                    ],
                    width="stretch"
                )

                plot_df = shap_df.head(10).sort_values("SHAP")

                fig, ax = plt.subplots(figsize=(9, 5))

                ax.barh(
                    plot_df["Feature"],
                    plot_df["SHAP"]
                )

                ax.axvline(0)
                ax.set_xlabel("SHAP Contribution")
                ax.set_title(
                    f"SHAP Explanation — {prediction}"
                )

                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                shap_df = pd.DataFrame()
                st.error(f"SHAP error: {e}")

            # -----------------------------------------
            # LIME
            # -----------------------------------------

            st.subheader("🧠 LIME Explanation")

            try:
                lime_df = get_lime_dataframe(
                    model,
                    encoder,
                    reference,
                    input_df,
                    features
                )

                st.dataframe(
                    lime_df[
                        ["Feature", "LIME", "Absolute_LIME"]
                    ],
                    width="stretch"
                )

                plot_df = lime_df.head(10).sort_values("LIME")

                fig2, ax2 = plt.subplots(figsize=(9, 5))

                ax2.barh(
                    plot_df["Feature"],
                    plot_df["LIME"]
                )

                ax2.axvline(0)
                ax2.set_xlabel("LIME Weight")
                ax2.set_title(
                    f"LIME Explanation — {prediction}"
                )

                st.pyplot(fig2)
                plt.close(fig2)

            except Exception as e:
                lime_df = pd.DataFrame()
                st.error(f"LIME error: {e}")

            # -----------------------------------------
            # Comparison
            # -----------------------------------------

            st.subheader("⚖️ SHAP vs LIME Comparison")

            if not shap_df.empty and not lime_df.empty:

                rows = []

                for feature in features:

                    shap_match = shap_df[
                        shap_df["Feature"] == feature
                    ]

                    lime_match = lime_df[
                        lime_df["Feature"].str.contains(
                            feature,
                            regex=False
                        )
                    ]

                    shap_value = (
                        float(shap_match.iloc[0]["SHAP"])
                        if not shap_match.empty
                        else 0.0
                    )

                    lime_value = (
                        float(lime_match.iloc[0]["LIME"])
                        if not lime_match.empty
                        else 0.0
                    )

                    rows.append({
                        "Feature": feature,
                        "SHAP": round(shap_value, 4),
                        "LIME": round(lime_value, 4)
                    })

                comparison = pd.DataFrame(rows)

                st.dataframe(
                    comparison,
                    width="stretch"
                )

                shap_top = set(
                    shap_df.head(5)["Feature"]
                )

                lime_top = set()

                for text in lime_df.head(5)["Feature"]:
                    for feature in features:
                        if feature in text:
                            lime_top.add(feature)

                union = shap_top | lime_top
                intersection = shap_top & lime_top

                agreement = (
                    len(intersection) / len(union) * 100
                    if union else 0
                )

                x, y, z = st.columns(3)

                x.metric(
                    "SHAP Top Features",
                    len(shap_top)
                )

                y.metric(
                    "LIME Top Features",
                    len(lime_top)
                )

                z.metric(
                    "Agreement",
                    f"{agreement:.1f}%"
                )

                top_features = ", ".join(
                    shap_df.head(3)["Feature"].tolist()
                )

                st.info(
                    f"SHAP identified {top_features} as major "
                    "contributors to the prediction. LIME provides "
                    "a local explanation for this particular file. "
                    "The comparison helps determine whether both "
                    "XAI methods highlight similar factors."
                )

            # -----------------------------------------
            # Recommendation
            # -----------------------------------------

            st.subheader("🛡️ Security Recommendation")

            if str(prediction).lower() == "high":
                st.error(
                    "Do not execute or open this file on a normal "
                    "workstation. Isolate it and perform further "
                    "analysis in a controlled environment."
                )
            elif str(prediction).lower() == "medium":
                st.warning(
                    "Treat this file cautiously. Verify its source "
                    "and perform additional security checks."
                )
            else:
                st.success(
                    "No strong static risk indicators were detected "
                    "by this prototype model. Low risk does not "
                    "guarantee that a file is completely safe."
                )


# =========================================================
# MANUAL MODE
# =========================================================

else:

    st.header("🎛️ Manual Security Assessment")

    if not MANUAL_MODEL_PATH.exists():
        st.error(
            "Your existing model/risk_model.pkl was not found."
        )
        st.stop()

    package = joblib.load(MANUAL_MODEL_PATH)

    model = package["model"]
    encoder = package["encoder"]
    features = package["features"]

    st.sidebar.markdown("### 🔐 Security Parameters")

    values = {}

    for feature in features:

        if "unusual" in feature.lower():
            values[feature] = st.sidebar.selectbox(
                feature,
                ["Normal", "Suspicious"]
            )
        else:
            values[feature] = st.sidebar.number_input(
                feature,
                min_value=0.0,
                value=1.0
            )

    row = {}

    for feature in features:
        value = values[feature]

        if isinstance(value, str):
            row[feature] = int(
                value.lower() == "suspicious"
            )
        else:
            row[feature] = value

    input_df = pd.DataFrame([row], columns=features)

    st.subheader("📋 Current Security Parameters")

    st.dataframe(
        input_df,
        width="stretch"
    )

    if st.button(
        "🔍 ANALYZE SECURITY RISK",
        type="primary"
    ):

        prediction_encoded = int(
            model.predict(input_df)[0]
        )

        probabilities = model.predict_proba(input_df)[0]

        prediction = encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        confidence = probabilities[prediction_encoded]

        st.subheader("🎯 Risk Assessment Result")

        if "high" in str(prediction).lower():
            st.error("🔴 HIGH SECURITY RISK")
        elif "medium" in str(prediction).lower():
            st.warning("🟠 MEDIUM SECURITY RISK")
        else:
            st.success("🟢 LOW SECURITY RISK")

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.1f}%"
        )

        probability_df = pd.DataFrame({
            "Risk Level": encoder.classes_,
            "Probability (%)": probabilities * 100
        })

        st.subheader("📊 Risk Probability")

        st.bar_chart(
            probability_df.set_index("Risk Level")
        )

        st.info(
            "The manual assessment is your original Random Forest "
            "security-risk module."
        )
