import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    """Create advanced features from the dataset."""
    print("\nAvailable columns:")
    for col in df.columns:
        print(f"- {col}")
    
    # Create copy of dataframe
    df_new = df.copy()
    
    # Ratio features
    df_new['rooms_per_household'] = df_new['AveRooms']  # Already per household
    df_new['bedrooms_per_room'] = df_new['AveBedrms'] / df_new['AveRooms']
    df_new['population_per_household'] = df_new['Population'] / df_new['AveOccup']
    
    # Location-based features
    df_new['location_cluster'] = (df_new['Latitude'].round(1).astype(str) + '_' + 
                                df_new['Longitude'].round(1).astype(str))
    
    # Create location cluster encoding
    location_stats = df_new.groupby('location_cluster')['MedHouseVal'].agg(['mean', 'std']).reset_index()
    location_stats.columns = ['location_cluster', 'loc_mean_price', 'loc_std_price']
    df_new = df_new.merge(location_stats, on='location_cluster', how='left')
    
    # Distance from major cities (San Francisco and Los Angeles)
    sf_coords = (37.7749, -122.4194)
    la_coords = (34.0522, -118.2437)
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    df_new['distance_to_sf'] = haversine_distance(
        df_new['Latitude'], df_new['Longitude'], sf_coords[0], sf_coords[1]
    )
    df_new['distance_to_la'] = haversine_distance(
        df_new['Latitude'], df_new['Longitude'], la_coords[0], la_coords[1]
    )
    
    # Binned features
    df_new['age_bin'] = pd.qcut(df_new['HouseAge'], q=10, labels=False)
    df_new['income_bin'] = pd.qcut(df_new['MedInc'], q=10, labels=False)
    
    # Additional features
    df_new['rooms_per_bedroom'] = df_new['AveRooms'] / df_new['AveBedrms']
    df_new['population_density'] = df_new['Population'] / (df_new['AveRooms'] * df_new['AveOccup'])
    
    # Drop intermediate columns
    df_new = df_new.drop(['location_cluster'], axis=1)
    
    return df_new

def prepare_data(df):
    """Prepare data for modeling."""
    # Create features
    df = create_features(df)
    
    # Separate features and target
    X = df.drop(['MedHouseVal'], axis=1)
    y = df['MedHouseVal']
    
    # Create polynomial features for important numerical columns
    important_features = ['MedInc', 'rooms_per_household', 'population_per_household']
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[important_features])
    poly_feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]
    
    # Add polynomial features to the dataset
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    X = pd.concat([X, poly_df], axis=1)
    
    return X, y

def train_model(X, y):
    """Train an XGBoost model with cross-validation."""
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    print("\nTraining XGBoost model...")
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Perform cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='r2')
    
    results = {
        'model': model,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'cv_scores': cv_scores.mean()
    }
    
    print(f"XGBoost R² Score: {r2:.4f}")
    print(f"XGBoost CV R² Score: {cv_scores.mean():.4f}")
    
    return results, scaler, X.columns.tolist()

def save_model(model_data, feature_names, output_dir='../app/'):
    """Save model and parameters."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        'mse': float(model_data['mse']),
        'mae': float(model_data['mae']),
        'r2': float(model_data['r2']),
        'cv_r2': float(model_data['cv_scores'])
    }
    
    with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=4)
    
    # Save model
    joblib.dump(model_data['model'], os.path.join(output_dir, 'model.joblib'))
    
    print(f"Model and parameters saved to {output_dir}")
    print("\nModel Performance:")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Cross-validation R² Score: {metrics['cv_r2']:.4f}")

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
    X, y = prepare_data(df)
    
    # Train model
    print("\nTraining models...")
    best_model, scaler, feature_names = train_model(X, y)
    
    # Save model and metrics
    print("\nSaving model and metrics...")
    save_model(best_model, feature_names)

if __name__ == '__main__':
    main() 