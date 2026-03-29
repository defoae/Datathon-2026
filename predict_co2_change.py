from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CO2 change prediction from GDP sector changes.")
    parser.add_argument(
        "--model-path",
        default="artifacts/co2_change_model.joblib",
        help="Path to trained model bundle.",
    )
    parser.add_argument(
        "--input-json",
        help="Path to JSON with feature_name -> change ratio (for example 0.05 for +5%).",
    )
    parser.add_argument(
        "--baseline-co2",
        type=float,
        default=1_000_000.0,
        help="Baseline CO2 value to convert relative change into absolute delta.",
    )
    args = parser.parse_args()

    bundle = joblib.load(Path(args.model_path))
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {name: 0.0 for name in feature_names}

    row = {name: float(payload.get(name, 0.0)) for name in feature_names}
    X = pd.DataFrame([row], columns=feature_names)
    predicted_change = float(model.predict(X)[0])
    absolute_delta = predicted_change * args.baseline_co2

    print(f"Predicted CO2 change ratio: {predicted_change:.6f}")
    print(f"Predicted CO2 change percent: {predicted_change * 100:.2f}%")
    print(f"Predicted absolute CO2 delta: {absolute_delta:.2f}")


if __name__ == "__main__":
    main()
