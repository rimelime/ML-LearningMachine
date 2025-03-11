import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import chi2_contingency

# Step 1: Load data
dataset_path = "D:/year 3/hk2/Machine Learning/testcode"  # Folder containing the dataset
dataset_files = os.listdir(dataset_path)
print("Files in dataset folder:", dataset_files)

# Find the first CSV file
csv_files = [f for f in dataset_files if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the dataset folder.")

csv_path = os.path.join(dataset_path, csv_files[0])
print("Using CSV file:", csv_path)

# Load data
data = pd.read_csv(csv_path)
print(data.head())

# Step 2: Data processing
# Remove missing values
data = data.dropna()

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Perform feature independence analysis using Chi-Square test
print("\nFeature Independence Analysis:")
independent_features = []
for column in data.columns:
    if column != 'cost':
        contingency_table = pd.crosstab(data[column], data['cost'])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        print(f"{column}: p-value = {p:.5f}")
        if p > 0.05:
            independent_features.append(column)

print("Potentially independent features:", independent_features)

# Split input and target variables
X = data.drop(columns=['cost'])  # Input variables
y = data['cost']  # Target variable

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Step 4: Train model with smoothing (Laplace Smoothing)
model = GaussianNB(var_smoothing=1e-9)
model.fit(X_train, y_train)

# Evaluate on training data
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print(f'Training MSE: {train_mse}')

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R^2 Score: {r2}')

# Visualize actual vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
plt.title("Actual vs Predicted Cost")
plt.axline((0, 0), slope=1, color="r", linestyle="--")
plt.show()
