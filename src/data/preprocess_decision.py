import pandas as pd
import os

def preprocess_media_cost(raw, out):
    """
    Reads the 'media_prediction_and_its_cost.csv' from data/raw,
    applies categorical mappings, quantile-based bins,
    and saves cleaned data to 'data/processed/cleaned_data.csv'.
    """
    # 1) Read the CSV
    print(f"Reading raw data from {raw}")
    df = pd.read_csv(raw)

    # 2) Map string/categorical columns
    categorical_columns_str = [
        'food_category', 'food_department', 'food_family', 'promotion_name',
        'sales_country', 'marital_status', 'gender', 'education', 'member_card',
        'occupation', 'avg_cars_at home(approx)', 'avg. yearly_income',
        'num_children_at_home', 'brand_name', 'recyclable_package', 'low_fat',
        'store_type', 'store_city', 'store_state', 'coffee_bar', 'video_store',
        'salad_bar', 'prepared_food', 'florist', 'media_type', 'houseowner'
    ]
    for col in categorical_columns_str:
        if col not in df.columns:
            print(f"[WARNING] Column '{col}' not found in data; skipping mapping.")
            continue

        unique_values = df[col].unique()
        mapping = {val: i for i, val in enumerate(unique_values)}
        print(f"Mapping for {col}: {mapping}")
        df[col] = df[col].map(mapping)

    # 3) Quantile-based bins for numeric columns
    numeric_columns = [
        'store_sales(in millions)', 'store_cost(in millions)', 'SRP',
        'gross_weight', 'units_per_case', 'store_sqft', 'grocery_sqft',
        'frozen_sqft', 'meat_sqft', 'net_weight'
    ]
    for col in numeric_columns:
        if col not in df.columns:
            print(f"[WARNING] Numeric column '{col}' not found in data; skipping binning.")
            continue

        # Calculate quantiles
        very_low = df[col].quantile(0.2)
        low      = df[col].quantile(0.4)
        medium   = df[col].quantile(0.6)
        high     = df[col].quantile(0.8)

        print(f"\nColumn: {col}")
        print(f"  Very Low (< {very_low:.2f}) → 0")
        print(f"  Low (< {low:.2f}) → 1")
        print(f"  Medium (< {medium:.2f}) → 2")
        print(f"  High (< {high:.2f}) → 3")
        print(f"  Very High (≥ {high:.2f}) → 4")

        df[col] = df[col].apply(lambda x: categorize_sales(x, very_low, low, medium, high))

    # 4) Save output
    os.makedirs(os.path.dirname(out), exist_ok=True)  # ensure 'data/processed' exists
    if os.path.exists(out):
        os.remove(out)
    df.to_csv(out, index=False)

    print(f"\nCleaned data saved to: {out}")

def categorize_sales(value, very_low, low, medium, high):
    """Helper to bin a numeric value into 0..4 based on quantiles."""
    if value < very_low:
        return 0  # Very Low
    elif value < low:
        return 1  # Low
    elif value < medium:
        return 2  # Medium
    elif value < high:
        return 3  # High
    else:
        return 4  # Very High
    
def split_decision_data(
    input_csv="data/processed/decisionData.csv",
    train_csv="data/processed/train_data_decision.csv",
    test_csv="data/processed/test_data_decision.csv",
    train_ratio=0.8
):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    # Calculate split index
    split_index = int(len(df) * train_ratio)

    # Create train/test DataFrames
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # Remove old files if they exist
    if os.path.exists(train_csv):
        os.remove(train_csv)
    if os.path.exists(test_csv):
        os.remove(test_csv)

    # Save train/test
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Train set saved to {train_csv} ({len(train_df)} rows)")
    print(f"Test  set saved to {test_csv}  ({len(test_df)} rows)")
    
rawPath = os.path.join("data", "raw", "media prediction and its cost.csv")
outputPath = os.path.join("data", "processed", "decisionData.csv")

preprocess_media_cost(rawPath, outputPath)
split_decision_data()