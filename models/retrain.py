import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.ensemble import BalancedRandomForestClassifier
import pickle

# Load the data
data = pd.read_csv('datasets\combined_dataset_with_synthetic_data.csv')

X = data.drop(['readmitted_yes'], axis=1)
y = data['readmitted_yes']  # Target variable for classification

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the balanced data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the data for the Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the Neural Network model
# mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
# mlp.fit(X_train_scaled, y_train)

# Train a Balanced Random Forest model
balanced_rf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
balanced_rf.fit(X_train, y_train)

# Define the directory where you want to save the models
save_dir = 'models'

# Create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Define file paths
# mlp_model_path = os.path.join(save_dir, 'mlp_model.pkl')
rf_model_path = os.path.join(save_dir, 'balanced_rf_model.pkl')
scaler_path = os.path.join(save_dir, 'scaler.pkl')

# Save the models and scaler using pickle
# try:
#     with open(mlp_model_path, 'wb') as f:
#         pickle.dump(mlp, f)
#     print(f"MLP model saved successfully to {mlp_model_path}.")
# except Exception as e:
#     print(f"Error saving MLP model: {e}")

try:
    with open(rf_model_path, 'wb') as f:
        pickle.dump(balanced_rf, f)
    print(f"Balanced Random Forest model saved successfully to {rf_model_path}.")
except Exception as e:
    print(f"Error saving Balanced Random Forest model: {e}")

try:
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved successfully to {scaler_path}.")
except Exception as e:
    print(f"Error saving scaler: {e}")
