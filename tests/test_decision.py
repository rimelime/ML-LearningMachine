import pandas as pd
import numpy as np
import joblib
import os
from src.models.decision_tree import DecisionTreeNode, predict

def mean_absolute_error_manual(y_true, y_pred):
    total_error = sum(abs(t - p) for t, p in zip(y_true, y_pred))
    return total_error / len(y_true)

def r2_score_manual(y_true, y_pred):
    mean_y = np.mean(y_true)
    ss_total = sum((y - mean_y)**2 for y in y_true)
    ss_residual = sum((t - p)**2 for t, p in zip(y_true, y_pred))
    return 1 - (ss_residual / ss_total)

def test_saved_model(
    test_csv="data/processed/test_data_decision.csv",
    model_path="models/trained/decision_tree_model.pkl",
    prediction_num=20
):
    """
    Loads the saved decision tree model from 'model_path',
    tests it on 'test_csv', prints metrics and sample predictions.
    """
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test file not found: {test_csv}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1) Load the model
    decision_tree = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # 2) Load the test data
    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=['cost']).values
    y_test = df_test['cost'].values
    print(f"Loaded test data from {test_csv}, shape={X_test.shape}")

    # 3) Make predictions
    y_pred = np.array([predict(decision_tree, x) for x in X_test])

    # 4) Compute metrics
    mae = mean_absolute_error_manual(y_test, y_pred)
    r2 = r2_score_manual(y_test, y_pred)
    print(f"\nTest MAE: {mae:.2f}")
    print(f"Test R²: {r2:.2f}")

    # 5) Print sample predictions
    print("\nSample Predictions:")
    for i in range(min(prediction_num, len(y_test))):
        print(f"  Row {i}: Actual = {y_test[i]:.2f}, Predicted = {y_pred[i]:.2f}")

test_saved_model()