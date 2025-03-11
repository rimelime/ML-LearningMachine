import pandas as pd
import numpy as np
import os
import joblib

from src.models.decision_tree import DecisionTreeNode, predict

MAX_DEPTH = 5
TRAIN_CSV = "data/processed/train_data_decision.csv"
MODEL_PATH = "models/trained/decision_tree_model.pkl"



def gini_impurity(y):
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities**2)

def find_best_split(X, y):
    best_gini = 999
    best_feature = None
    best_threshold = None

    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_mask = X[:, feature_index] <= threshold
            right_mask = ~left_mask

            # Skip if one side is empty
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            gini_left = gini_impurity(y[left_mask])
            gini_right = gini_impurity(y[right_mask])

            weighted_gini = (
                (np.sum(left_mask) * gini_left) +
                (np.sum(right_mask) * gini_right)
            ) / len(y)

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold

def build_tree(X, y, depth=0, max_depth=5):
    # Stop if depth limit reached or data is pure
    if depth >= max_depth or len(set(y)) == 1:
        return DecisionTreeNode(prediction=np.mean(y))

    feature, threshold = find_best_split(X, y)
    if feature is None:
        return DecisionTreeNode(prediction=np.mean(y))

    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask

    left_subtree = build_tree(X[left_mask], y[left_mask], depth+1, max_depth)
    right_subtree = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)

    return DecisionTreeNode(feature, threshold, left_subtree, right_subtree)

def train_and_save_model(train_csv=TRAIN_CSV, model_path=MODEL_PATH, max_depth=MAX_DEPTH):
    """
    Loads the training data from 'train_csv',
    builds the decision tree, and saves the model to 'model_path'.
    """
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Could not find {train_csv}")

    # 1) Load train data
    df_train = pd.read_csv(train_csv)
    X_train = df_train.drop(columns=['cost']).values
    y_train = df_train['cost'].values

    print(f"Loaded train data from {train_csv} with shape {X_train.shape}")

    # 2) Build the tree
    decision_tree = build_tree(X_train, y_train, depth=0, max_depth=max_depth)
    print("Decision tree trained successfully!")

    # 3) Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure folder
    joblib.dump(decision_tree, model_path)
    print(f"Model saved to {model_path}")

train_and_save_model()