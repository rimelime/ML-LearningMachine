import pandas as pd
import os

rawPath = os.path.join("data", "raw", "media prediction and its cost.csv")
outputPath = os.path.join("data", "processed", "cleaned_data.csv")

def categorize_sales(value, very_low, low, medium, high):
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

def preprocessing(csv_path=rawPath,out_path=outputPath):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    df = pd.read_csv(csv_path)

    categorical_columns = ['food_category', 'food_department','food_family', 'promotion_name','sales_country',
    'marital_status','gender','education','member_card','occupation','avg_cars_at home(approx)','avg. yearly_income',
    'num_children_at_home','brand_name','recyclable_package','low_fat','store_type','store_city','store_state',
    'coffee_bar','video_store','salad_bar','prepared_food','florist','media_type','houseowner']
    for col in categorical_columns:
        unique_categories = df[col].unique()

        mapping = {}
        for idx, category in enumerate(unique_categories):
            mapping[category] = idx  # Assign a unique number to each category
        
        print(f"Generated mapping for {col}: {mapping}")
        df[col] = df[col].map(mapping)
    categorical_columns = ['store_sales(in millions)','store_cost(in millions)','SRP','gross_weight','units_per_case',
    'store_sqft','grocery_sqft','frozen_sqft','meat_sqft','net_weight']
    for col in categorical_columns:
        very_low = df[col].quantile(0.2)  # 20th percentile
        low = df[col].quantile(0.4)       # 40th percentile
        medium = df[col].quantile(0.6)    # 60th percentile (Median)
        high = df[col].quantile(0.8)      # 80th percentile
        # Define categorization function
        
        print(f"    - Very Low (< {very_low}) → 0")
        print(f"    - Low (< {low}) → 1")
        print(f"    - Medium (< {medium}) → 2")
        print(f"    - High (< {high}) → 3")
        print(f"    - Very High (≥ {high}) → 4")

        # Apply categorization and convert to numeric values
        df[col] = df[col].apply(lambda x: categorize_sales(x, very_low, low, medium, high))
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    df.to_csv(out_path, index=False)
    return out_path