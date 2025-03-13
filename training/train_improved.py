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

class FHECompatibleNN(nn.Module):
    """Neural network designed to be FHE-compatible"""
    def __init__(self, input_dim):
        super().__init__()
        
        # Wider architecture with residual connections
        self.layer1a = nn.Linear(input_dim, 512)
        self.layer1b = nn.Linear(512, 512)
        
        self.layer2a = nn.Linear(512, 256)
        self.layer2b = nn.Linear(256, 256)
        
        self.layer3a = nn.Linear(256, 128)
        self.layer3b = nn.Linear(128, 128)
        
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, 1)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.15)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for FHE compatibility"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization with small bounds
                bound = np.sqrt(2.0 / (m.weight.size(0) + m.weight.size(1))) * 0.1
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First residual block
        identity1 = x
        x = self.layer1a(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.layer1b(x)
        x = torch.relu(x)
        if identity1.shape[1] != x.shape[1]:
            identity1 = torch.cat([identity1, torch.zeros(identity1.shape[0], x.shape[1] - identity1.shape[1])], dim=1)
        x = x + identity1  # Residual connection
        
        # Second residual block
        identity2 = x
        x = self.layer2a(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.layer2b(x)
        x = torch.relu(x)
        x = x + identity2[:, :256]  # Residual connection
        
        # Third residual block
        identity3 = x
        x = self.layer3a(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.layer3b(x)
        x = torch.relu(x)
        x = x + identity3[:, :128]  # Residual connection
        
        x = self.layer4(x)
        x = torch.relu(x)
        x = self.dropout4(x)
        
        x = self.layer5(x)
        return x

class HousingDataset(Dataset):
    """Custom dataset for housing data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values if hasattr(y, 'values') else y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_features(df):
    """Create FHE-friendly features"""
    df_new = df.copy()
    
    # Log transform for skewed features (FHE-friendly as we can precompute these)
    df_new['log_income'] = np.log1p(df_new['MedInc'])
    df_new['log_rooms'] = np.log1p(df_new['AveRooms'])
    df_new['log_bedrooms'] = np.log1p(df_new['AveBedrms'])
    df_new['log_population'] = np.log1p(df_new['Population'])
    df_new['log_occupancy'] = np.log1p(df_new['AveOccup'])
    
    # Simple ratios (FHE-friendly as they only use multiplication and division)
    df_new['rooms_per_person'] = df_new['AveRooms'] / df_new['Population']
    df_new['bedrooms_per_person'] = df_new['AveBedrms'] / df_new['Population']
    df_new['income_per_room'] = df_new['MedInc'] / df_new['AveRooms']
    df_new['occupancy_density'] = df_new['Population'] / df_new['AveOccup']
    df_new['bedroom_ratio'] = df_new['AveBedrms'] / df_new['AveRooms']
    
    # Location features (FHE-friendly as they use basic arithmetic)
    df_new['coastal_distance'] = abs(df_new['Longitude'] + 122)
    df_new['latitude_adjusted'] = df_new['Latitude'] - 36  # Normalize around central CA
    df_new['location_score'] = df_new['coastal_distance'] * df_new['latitude_adjusted']
    
    # Binned features (FHE-friendly as we can precompute these)
    df_new['age_decade'] = np.floor(df_new['HouseAge'] / 10)
    df_new['age_century'] = np.floor(df_new['HouseAge'] / 100)
    
    # Feature interactions (all FHE-friendly)
    df_new['income_age'] = df_new['log_income'] * df_new['age_decade']
    df_new['rooms_age'] = df_new['log_rooms'] * df_new['age_decade']
    df_new['location_value'] = df_new['coastal_distance'] * df_new['latitude_adjusted']
    df_new['density_income'] = df_new['log_income'] * df_new['occupancy_density']
    df_new['bedroom_density'] = df_new['bedrooms_per_person'] * df_new['occupancy_density']
    
    # Quadratic terms (FHE-friendly)
    df_new['income_squared'] = df_new['log_income'] ** 2
    df_new['rooms_squared'] = df_new['log_rooms'] ** 2
    df_new['coastal_squared'] = df_new['coastal_distance'] ** 2
    df_new['latitude_squared'] = df_new['latitude_adjusted'] ** 2
    
    # Cubic terms (FHE-friendly)
    df_new['income_cubed'] = df_new['log_income'] ** 3
    df_new['rooms_cubed'] = df_new['log_rooms'] ** 3
    
    # Drop original features that have been transformed
    df_new = df_new.drop(['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup'], axis=1)
    
    return df_new

def train_model(X, y, batch_size=256, epochs=400, learning_rate=0.0005):
    """Train the FHE-compatible neural network"""
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create datasets
    train_dataset = HousingDataset(X_train_scaled, y_train)
    val_dataset = HousingDataset(X_val_scaled, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = FHECompatibleNN(input_dim=X_train.shape[1])
    criterion = CustomLoss(alpha=0.6)  # Adjusted alpha for better balance
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 40
    patience_counter = 0
    
    print("\nTraining neural network...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs.squeeze(), batch_y).item()
                val_predictions.extend(outputs.squeeze().numpy())
                val_targets.extend(batch_y.numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Calculate R² score for validation set
        val_r2 = r2_score(val_targets, val_predictions)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model
    model.load_state_dict(best_model)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_val_scaled)).numpy()
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    results = {
        'model': model,
        'scaler': scaler,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    print(f"\nFinal R² Score: {r2:.4f}")
    print(f"Final MSE: {mse:.4f}")
    print(f"Final MAE: {mae:.4f}")
    
    return results

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
    # Load data
    print("Loading data...")
    housing = fetch_california_housing()
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )
    df['MedHouseVal'] = housing.target
    
    print("\nDataset shape:", df.shape)
    print("\nFeatures:", housing.feature_names)
    print("\nFirst few rows:")
    print(df.head())
    
    # Prepare data
    print("\nPreparing data and engineering features...")
    df = create_features(df)
    
    # Separate features and target
    X = df.drop(['MedHouseVal'], axis=1)
    y = df['MedHouseVal']
    
    # Train model
    print("\nTraining model...")
    model_data = train_model(X, y)
    
    # Save model and metrics
    print("\nSaving model and metrics...")
    save_model(model_data)

if __name__ == '__main__':
    main() 