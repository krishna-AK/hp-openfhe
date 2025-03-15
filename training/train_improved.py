import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json
import os
import warnings
import copy
warnings.filterwarnings('ignore')
from fhe_model import (
    FHECompatibleNN, 
    preprocess_features, 
    # quantize_weights,  # Commented out quantization
    convert_to_fhe
)

class CustomLoss(nn.Module):
    """Custom loss function that combines MSE with relative error"""
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        relative_loss = torch.mean(torch.abs((pred - target) / (target + 1e-8)))
        return self.alpha * mse_loss + (1 - self.alpha) * relative_loss

class HousingDataset(Dataset):
    """Custom dataset for housing data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(X, y, scaler=None):
    """Prepare data for FHE-compatible training"""
    # Convert to PyTorch tensors without scaling
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).view(-1, 1)
    
    # Preprocess features for FHE
    X = preprocess_features(X)
    
    return X, y, None  # No scaler needed

def train_model(X, y, X_val, y_val, input_dim):
    # Initialize model with appropriate hidden dimension
    model = FHECompatibleNN(input_dim=input_dim, hidden_dim=16)
    
    # Use a learning rate appropriate for the scale of house prices
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0001)
    
    # Use pure MSE loss for house price prediction
    criterion = nn.MSELoss()  # Pure MSE loss is better for regression
    
    # Use a more patient scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, verbose=True
    )
    
    # Training loop with more epochs
    n_epochs = 1000
    best_r2 = -float('inf')
    best_model = None
    early_stop_patience = 50
    early_stop_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimize
        loss.backward()
        # Use a moderate gradient clipping threshold
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            r2_val = r2_score_torch(y_val, val_outputs)
            
            # Save the best model based on R² score
            if r2_val > best_r2:
                best_r2 = r2_val
                best_model = copy.deepcopy(model)
            
            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, R² Score: {r2_val:.4f}')
            
            if epoch % 50 == 0:
                print("\nSample predictions:")
                for i in range(5):
                    print(f"Predicted: {val_outputs[i].item():.2f}, Actual: {y_val[i].item():.2f}")
                print()
            
            scheduler.step(val_loss)
    
    # Return the best model based on validation R² score
    return best_model if best_model is not None else model

def plot_training_history(train_losses, val_losses):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('../app/training_history.png')
    plt.close()

def save_model(model_data, output_dir='../app/'):
    """Save model and parameters"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        'mse': float(model_data['mse']),
        'mae': float(model_data['mae']),
        'r2': float(model_data['r2'])
    }
    
    with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save model state
    torch.save(model_data['model'].state_dict(), os.path.join(output_dir, 'model.pt'))
    
    # Save scaler parameters
    scaler_params = {
        'mean': model_data['scaler'].mean_.tolist(),
        'scale': model_data['scaler'].scale_.tolist()
    }
    
    with open(os.path.join(output_dir, 'scaler_params.json'), 'w') as f:
        json.dump(scaler_params, f, indent=4)
    
    # Plot training history
    plot_training_history(model_data['train_losses'], model_data['val_losses'])
    
    print(f"\nModel and parameters saved to {output_dir}")
    print("\nModel Performance:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")

def save_model_for_fhe(model):
    """Save model parameters in FHE-compatible format."""
    try:
        model_params = convert_to_fhe(model)
        
        # Save model parameters
        with open('../app/fhe_model_params.json', 'w') as f:
            json.dump(model_params, f, indent=4)
        print("Model parameters saved to ../app/fhe_model_params.json")
        
        # Print model architecture summary
        print("\nModel Architecture Summary:")
        print(f"Input dimension: {model.input_dim}")
        print(f"Hidden dimension: {model.hidden_dim}")
        print(f"Activation: Polynomial (degree {model.act1.degree})")
        print(f"Polynomial coefficients: {model.act1.coefficients.data.tolist()}")
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise

def load_data():
    """Load data directly from CSV files"""
    print("Loading data from CSV files...")
    try:
        X = pd.read_csv('data/X_train.csv')
        y = pd.read_csv('data/y_train.csv')
        return X.values, y.values.flatten()
    except FileNotFoundError as e:
        print(f"Error: Could not find data files in '../data' directory.")
        raise e

def r2_score_torch(y_true, y_pred):
    """Calculate R² score for PyTorch tensors"""
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()

def main():
    try:
        # Load dataset
        X, y = load_data()
        
        print(f"Dataset shape: {X.shape}")
        print("\nFirst few rows of the dataset:")
        print(pd.DataFrame(X).head())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Prepare data
        print("\nPreparing data...")
        X_train, y_train, _ = prepare_data(X_train, y_train)
        X_val, y_val, _ = prepare_data(X_val, y_val)
        X_test, y_test, _ = prepare_data(X_test, y_test)
        
        # Train the model
        print("\nTraining model...")
        model = train_model(X_train, y_train, X_val, y_val, input_dim=X_train.shape[1])
        
        # Evaluate model on test set
        print("\nEvaluating model on test data...")
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).numpy().flatten()
            mse = mean_squared_error(y_test.numpy(), y_pred)
            mae = mean_absolute_error(y_test.numpy(), y_pred)
            r2 = r2_score(y_test.numpy(), y_pred)
            print(f"\nFinal Test Performance Metrics:")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
        
        # Save model in FHE-compatible format
        print("\nSaving model in FHE-compatible format...")
        save_model_for_fhe(model)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 