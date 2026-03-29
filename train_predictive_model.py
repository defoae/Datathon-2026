from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_and_transform_emissions(csv_path: Path) -> pd.DataFrame:
    emissions = pd.read_csv(csv_path)
    emissions["building_construction_manufacturing"] = (
        emissions["Buildings"] + emissions["Manufacturing and construction"]
    )

    drop_cols = [
        "Fugitive emissions",
        "Other fuel combustion",
        "Buildings",
        "Manufacturing and construction",
    ]
    emissions = emissions.drop(columns=drop_cols)
    emissions = emissions.rename(columns={"Entity": "Country"})
    return emissions


def load_isic_gdp_features(gdp_xlsx: Path) -> pd.DataFrame:
    gdp_raw = pd.read_excel(
        gdp_xlsx, sheet_name="Download-GDPcurrent-USD-countri", header=2
    )
    year_cols = [c for c in gdp_raw.columns if isinstance(c, (int, float))]
    long_df = gdp_raw[["Country", "IndicatorName"] + year_cols].melt(
        id_vars=["Country", "IndicatorName"], var_name="Year", value_name="value"
    )
    long_df["Year"] = long_df["Year"].astype(int)

    isic = long_df[long_df["IndicatorName"].str.contains(r"\(ISIC", na=False)].copy()
    isic = isic.pivot_table(
        index=["Country", "Year"],
        columns="IndicatorName",
        values="value",
        aggfunc="first",
    ).reset_index()

    # User-requested cleanup:
    # 1) Remove broad "Other Activities" bucket.
    # 2) Replace aggregate C-E by a derived Mining+Utilities level
    #    using (C-E) - D so manufacturing is not double represented.
    drop_col = "Other Activities (ISIC J-P)"
    ce_col = "Mining, Manufacturing, Utilities (ISIC C-E)"
    d_col = "Manufacturing (ISIC D)"
    derived_col = "Mining and Utilities (derived from ISIC C-E minus ISIC D)"

    if drop_col in isic.columns:
        isic = isic.drop(columns=[drop_col])

    if ce_col in isic.columns and d_col in isic.columns:
        isic[derived_col] = isic[ce_col] - isic[d_col]
        isic = isic.drop(columns=[ce_col])

    return isic


def build_change_dataset(emissions_csv: Path, gdp_xlsx: Path) -> tuple[pd.DataFrame, pd.Series]:
    emissions = load_and_transform_emissions(emissions_csv)
    isic = load_isic_gdp_features(gdp_xlsx)
    merged = emissions.merge(isic, on=["Country", "Year"], how="inner")

    emission_feature_cols = [
        c for c in emissions.columns if c not in ["Country", "Code", "Year"]
    ]
    merged["co2_total_selected"] = merged[emission_feature_cols].sum(axis=1)

    isic_cols = [c for c in isic.columns if c not in ["Country", "Year"]]
    merged = merged.sort_values(["Country", "Year"])

    # Build country-level year-over-year changes, then pool all countries.
    change_cols: list[str] = []
    for col in isic_cols:
        new_col = f"gdp_change_{col}"
        merged[new_col] = merged.groupby("Country")[col].pct_change(fill_method=None)
        change_cols.append(new_col)

    merged["co2_change_target"] = merged.groupby("Country")["co2_total_selected"].pct_change(
        fill_method=None
    )

    model_df = merged[change_cols + ["co2_change_target"]].replace(
        [np.inf, -np.inf], np.nan
    )
    model_df = model_df.dropna(subset=["co2_change_target"])
    model_df = model_df.fillna(0.0)

    X = model_df[change_cols]
    y = model_df["co2_change_target"]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
    numeric_cols = list(X.columns)
    preprocessor = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), numeric_cols)]
    )
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "rows_used": int(len(X)),
        "features_used": int(len(X.columns)),
        "mae_change": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    return pipeline, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a country-agnostic model to predict CO2 emissions change from ISIC GDP changes."
    )
    parser.add_argument(
        "--emissions-csv",
        default="data/ghg-emissions-by-sector.csv",
        help="Path to sector emissions CSV.",
    )
    parser.add_argument(
        "--gdp-xlsx",
        default="data/Download-GDPcurrent-USD-countries.xlsx",
        help="Path to GDP Excel file.",
    )
    parser.add_argument(
        "--model-out",
        default="artifacts/co2_change_model.joblib",
        help="Where to save the trained model bundle.",
    )
    args = parser.parse_args()

    X, y = build_change_dataset(Path(args.emissions_csv), Path(args.gdp_xlsx))
    pipeline, metrics = train_model(X, y)

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model": pipeline,
        "feature_names": list(X.columns),
        "metrics": metrics,
    }
    joblib.dump(model_bundle, out_path)

    print("Training complete.")
    print(f"Rows used: {metrics['rows_used']}")
    print(f"Features used: {metrics['features_used']}")
    print(f"MAE (change ratio): {metrics['mae_change']:.6f}")
    print(f"R^2: {metrics['r2']:.4f}")
    print(f"Model bundle saved to: {out_path}")


if __name__ == "__main__":
    main()
