from pathlib import Path
import pandas as pd
import os

# ------------------------------------------------------------------
# CONFIG – adjust lists if you add / drop columns
# ------------------------------------------------------------------
RAW_CSV   = Path("data/raw/media prediction and its cost.csv")

OUT_DIR   = Path("data/processed/SVM")
TRAIN_CSV = OUT_DIR / "train_text.csv"
TEST_CSV  = OUT_DIR / "test_text.csv"

TEXT_COLS = [
    "food_category","food_department","food_family","promotion_name",
    "sales_country","marital_status","gender","education","member_card",
    "occupation","houseowner","avg. yearly_income","brand_name",
    "store_type","store_city","store_state","media_type"
]

NUMERIC_COLS = [
    "store_sales(in millions)","store_cost(in millions)",
    "unit_sales(in millions)","SRP","gross_weight","net_weight",
    "units_per_case","store_sqft","grocery_sqft","frozen_sqft","meat_sqft"
]

TARGET        = "cost"
N_CLASSES     = 3          # use 5 if you prefer five classes
TRAIN_RATIO   = 0.80
RANDOM_STATE  = 42

# ------------------------------------------------------------------
def main() -> None:
    # 1) read raw ---------------------------------------------------
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"[ERROR] Raw file not found: {RAW_CSV}")
    print(f"Reading raw data from {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df):,} rows")

    # 2) build free-form text blob ---------------------------------
    txt_missing = [c for c in TEXT_COLS if c not in df.columns]
    for col in txt_missing:
        print(f"[WARNING] Column '{col}' not found in data; skipping.")
    used_text_cols = [c for c in TEXT_COLS if c in df.columns]

    df["text_blob"] = (
        df[used_text_cols]
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )

    # 3) bucket target ---------------------------------------------
    df["cost_class"] = pd.qcut(
        df[TARGET], q=N_CLASSES,
        labels=list(range(N_CLASSES)),
        duplicates="drop"      # avoid “bin edges must be unique”
    ).astype(int)

    # 4) select final columns --------------------------------------
    final_cols = ["text_blob"] + [c for c in NUMERIC_COLS if c in df.columns] + ["cost_class"]
    df_final   = df[final_cols]

    # 5) train / test split ----------------------------------------
    split_idx  = int(len(df_final) * TRAIN_RATIO)
    train_df   = df_final.iloc[:split_idx].reset_index(drop=True)
    test_df    = df_final.iloc[split_idx:].reset_index(drop=True)

    # 6) write files ----------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_CSV.exists(): TRAIN_CSV.unlink()
    if TEST_CSV.exists():  TEST_CSV.unlink()

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV,  index=False)

    print(f"Train set saved to {TRAIN_CSV} ({len(train_df):,} rows)")
    print(f"Test  set saved to {TEST_CSV}  ({len(test_df):,} rows)")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()