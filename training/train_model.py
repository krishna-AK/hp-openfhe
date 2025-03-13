import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os

def load_data(data_path):
    """Load and preprocess the California Housing dataset."""
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    
    return X, y

def train_model(X, y):
    """Train a linear regression model and evaluate its performance."""
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    return model, scaler, {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

def save_model_weights(model, scaler, metrics, feature_names, output_dir='../app/'):
    """Save model weights and parameters for FHE implementation."""
    # Create weights dictionary
    weights_dict = {
        'coefficients': model.coef_.tolist(),
        'intercept': float(model.intercept_),
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }
    
    # Save weights
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'model_weights.json'), 'w') as f:
        json.dump(weights_dict, f, indent=4)
    
    # Save metrics
    with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Model weights and metrics saved to {output_dir}")
    print("\nModel Performance:")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R2 Score: {metrics['r2']:.4f}")

def main():
    # Load data
    data_path = 'data/california_housing.csv'
    print("Loading data...")
    X, y = load_data(data_path)
    feature_names = X.columns.tolist()
    
    # Train model
    print("Training model...")
    model, scaler, metrics = train_model(X, y)
    
    # Save model weights and metrics
    print("Saving model weights and metrics...")
    save_model_weights(model, scaler, metrics, feature_names)

if __name__ == '__main__':
    main() 