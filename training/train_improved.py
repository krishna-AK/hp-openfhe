import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
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
warnings.filterwarnings('ignore')
from fhe_model import FHECompatibleNN, preprocess_features, quantize_weights, convert_to_fhe

class CustomLoss(nn.Module):
    """Custom loss function that combines MSE with relative error"""
    def __init__(self, alpha=0.7):
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
        self.y = torch.FloatTensor(y.values if hasattr(y, 'values') else y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_features(X):
    """Create FHE-friendly features"""
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(X, columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                                 'Population', 'AveOccup', 'Latitude', 'Longitude'])
    
    # Create FHE-friendly features
    features = pd.DataFrame()
    
    # Basic features (already FHE-friendly)
    features['MedInc'] = df['MedInc']
    features['HouseAge'] = df['HouseAge']
    features['AveRooms'] = df['AveRooms']
    features['AveBedrms'] = df['AveBedrms']
    features['Population'] = df['Population']
    features['AveOccup'] = df['AveOccup']
    features['Latitude'] = df['Latitude']
    features['Longitude'] = df['Longitude']
    
    # FHE-friendly derived features
    features['bedroom_ratio'] = df['AveBedrms'] / (df['AveRooms'] + 1e-6)
    features['occupancy_rate'] = df['AveOccup'] / (df['AveRooms'] + 1e-6)
    features['population_density'] = df['Population'] / (df['AveRooms'] + 1e-6)
    
    # Location-based features (FHE-friendly)
    features['lat_long_sum'] = df['Latitude'] + df['Longitude']
    features['lat_long_diff'] = df['Latitude'] - df['Longitude']
    
    # Age-based features (FHE-friendly)
    features['age_century'] = df['HouseAge'] / 100
    
    # Income-based features (FHE-friendly)
    features['income_squared'] = df['MedInc'] ** 2
    features['income_cubed'] = df['MedInc'] ** 3
    
    # Room-based features (FHE-friendly)
    features['rooms_squared'] = df['AveRooms'] ** 2
    features['rooms_cubed'] = df['AveRooms'] ** 3
    
    return features.values

def prepare_data(X, y):
    """Prepare data for FHE-compatible training"""
    # Create FHE-friendly features
    X = create_features(X)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # Preprocess features for FHE
    X = preprocess_features(X)
    
    return X, y, scaler

def train_model(X, y, X_val, y_val, input_dim):
    """Train the FHE-compatible model"""
    # Initialize model
    model = FHECompatibleNN(input_dim=input_dim)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
    
    # Training parameters
    batch_size = 256
    n_epochs = 400
    patience = 40
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            # Quantize weights after each update
            quantize_weights(model)
            
            total_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.unsqueeze(1))
            
            # Calculate R² score
            y_pred = val_outputs.numpy().flatten()
            r2 = 1 - np.sum((y_val.numpy() - y_pred) ** 2) / np.sum((y_val.numpy() - np.mean(y_val.numpy())) ** 2)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], '
                  f'Loss: {total_loss/n_batches:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'R² Score: {r2:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'r2_score': r2,
                'epoch': epoch
            }, '../app/model_fhe.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return model

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

def main():
    try:
        # Load dataset
        print("Loading California Housing dataset...")
        data = fetch_california_housing()
        X, y = data.data, data.target
        
        print(f"Dataset shape: {X.shape}")
        print("\nFirst few rows of the dataset:")
        print(pd.DataFrame(X, columns=data.feature_names).head())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Prepare data
        print("\nPreparing data...")
        X_train, y_train, scaler = prepare_data(X_train, y_train)
        X_val, y_val, _ = prepare_data(X_val, y_val)
        X_test, y_test, _ = prepare_data(X_test, y_test)
        
        # Train model
        print("\nTraining model...")
        model = train_model(X_train, y_train, X_val, y_val, input_dim=X_train.shape[1])
        
        # Evaluate model
        print("\nEvaluating model...")
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).numpy().flatten()
            
            # Calculate metrics
            mse = np.mean((y_test.numpy() - y_pred) ** 2)
            mae = np.mean(np.abs(y_test.numpy() - y_pred))
            r2 = 1 - np.sum((y_test.numpy() - y_pred) ** 2) / np.sum((y_test.numpy() - np.mean(y_test.numpy())) ** 2)
            
            print(f"\nFinal Performance Metrics:")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
        
        # Convert model for FHE deployment
        print("\nConverting model for FHE deployment...")
        fhe_model = convert_to_fhe(model)
        
        # Save FHE model parameters and scaler
        torch.save({
            'fhe_model': fhe_model,
            'scaler': scaler
        }, '../app/fhe_model_params.pth')
        print("\nModel and parameters saved to ../app/")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 