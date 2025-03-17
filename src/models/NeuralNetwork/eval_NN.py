import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from model_NN import NeuralNetworkModel 
import joblib
import os

BASE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(BASE_DIR,"models", "trained", "NN.pkl")

def train_eval_save_NN(data, data_output, scaler_y, device, k_folds=5, num_epochs=50, batch_size=64,output_file=MODEL_PATH):
# attemp cross validation using kfold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    criterion = torch.nn.MSELoss()
    results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(data)):
        print(f"Fold {fold+1}/{k_folds}")

        x_train, y_train = data[train_ids], data_output[train_ids]
        x_val,   y_val   = data[val_ids],  data_output[val_ids]

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = NeuralNetworkModel(x_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = running_loss / len(train_loader)
                print(f"  Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.5f}")

        model.eval()
        with torch.no_grad():
            y_pred = model(x_val)
            y_pred = torch.tensor(
                scaler_y.inverse_transform(y_pred.cpu().numpy()), 
                dtype=torch.float32
            ).to(device)
            y_val_orig = torch.tensor(
                scaler_y.inverse_transform(y_val.cpu().numpy()), 
                dtype=torch.float32
            ).to(device)

            accuracy = y_pred.eq(y_val_orig.round()).sum() / y_val_orig.shape[0]
            print(f"  Accuracy: {accuracy:.5f}")

            mse = criterion(y_pred, y_val_orig)
            print(f"  Validation MSE: {mse.item():.5f}")

            ss_total = torch.sum((y_val_orig - torch.mean(y_val_orig)) ** 2)
            ss_residual = torch.sum((y_val_orig - y_pred) ** 2)
            r2 = 1 - (ss_residual / ss_total)
            print(f"  R² Score: {r2.item():.5f}")

            results.append({
                "fold": fold + 1,
                "mse": mse.item(),
                "r2": r2.item()
            })
    
    #save the model
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure folder
    joblib.dump(model, output_file)
    print(f"Model saved to {output_file}")

    #return the result to evaluate the model
    return results
