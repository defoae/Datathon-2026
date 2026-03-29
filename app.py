from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("artifacts/co2_change_model.joblib")


@st.cache_resource
def load_bundle(path: Path):
    return joblib.load(path)


def main() -> None:
    st.title("Global CO2 Change Predictor")
    st.caption(
        "Predict CO2 emissions change from GDP sector/industry percent changes (country-agnostic trend model)."
    )

    if not MODEL_PATH.exists():
        st.error("Model not found. Run training first: python train_predictive_model.py")
        return

    bundle = load_bundle(MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    metrics = bundle["metrics"]

    st.subheader("Model metrics")
    st.write(
        {
            "rows_used": metrics["rows_used"],
            "features_used": metrics["features_used"],
            "mae_change": metrics["mae_change"],
            "r2": metrics["r2"],
        }
    )

    st.subheader("Scenario input")
    st.write("Enter GDP sector/industry changes in percent points (for example, +5 means +5%).")

    values_ratio: dict[str, float] = {}
    cols = st.columns(2)
    for idx, feature in enumerate(feature_names):
        clean_label = feature.replace("gdp_change_", "")
        percent_value = cols[idx % 2].number_input(
            f"{clean_label} (%)",
            min_value=-100.0,
            max_value=500.0,
            value=0.0,
            step=0.5,
            format="%.2f",
        )
        values_ratio[feature] = percent_value / 100.0

    baseline_co2 = st.number_input(
        "Baseline CO2 value (for absolute delta conversion)",
        min_value=0.0,
        value=1_000_000.0,
        step=10_000.0,
    )

    if st.button("Predict CO2 change"):
        X = pd.DataFrame([values_ratio], columns=feature_names)
        pred = float(model.predict(X)[0])
        st.success(f"Predicted CO2 change: {pred:.4%}")
        st.write(f"Predicted absolute CO2 delta: {pred * baseline_co2:,.2f}")


if __name__ == "__main__":
    main()
