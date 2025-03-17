import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from eval_NN import train_eval_save_NN

TRAIN_CSV = "data/processed/cleaned_data.csv"
MODEL_PATH = "models/trained/NN_model.pkl"

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # csv_path = "D:/year 3/hk2/Machine Learning/testcode/cleaned_data.csv"
    csv_path = TRAIN_CSV
    df = pd.read_csv(csv_path)

    # input features và target
    features = [
        'store_sales(in millions)', 'store_cost(in millions)', 'total_children',
        'avg_cars_at home(approx)', 'num_children_at_home', 'gross_weight',
        'net_weight', 'store_sqft', 'grocery_sqft', 'units_per_case', 'SRP'
    ]
    target = 'cost'

    data = df[features].values
    data_output = df[target].values

    # Standardize features & target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    data = scaler_X.fit_transform(data)
    data_output = scaler_y.fit_transform(data_output.reshape(-1, 1))

    # Convert to tensor and push to device
    data = torch.tensor(data, dtype=torch.float32).to(device)
    data_output = torch.tensor(data_output, dtype=torch.float32).to(device).view(-1, 1)

    # K-Fold 
    results = train_eval_save_NN(
        data=data,
        data_output=data_output,
        scaler_y=scaler_y,
        device=device,
        k_folds=5,
        num_epochs=50,
        batch_size=64,
    )

    print("\nCross-Validation Results:")
    for res in results:
        print(f"Fold {res['fold']}: MSE = {res['mse']:.5f}, R² = {res['r2']:.5f}")


    avg_mse = np.mean([r['mse'] for r in results])
    avg_r2 = np.mean([r['r2'] for r in results])
    print(f"\nAverage MSE: {avg_mse:.5f}")
    print(f"Average R²: {avg_r2:.5f}")
