from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def download_california_housing():
    # Download the dataset
    california = fetch_california_housing()
    
    # Create a DataFrame
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['median_house_value'] = california.target * 100000  # Convert to actual prices
    
    # Create ocean proximity dummy variables (all zeros since this data doesn't have ocean proximity)
    df['ocean_proximity_<1H OCEAN'] = 0
    df['ocean_proximity_INLAND'] = 0
    df['ocean_proximity_ISLAND'] = 0
    df['ocean_proximity_NEAR BAY'] = 0
    df['ocean_proximity_NEAR OCEAN'] = 0
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/california_housing.csv', index=False)
    print("Dataset downloaded and saved to data/california_housing.csv")
    print(f"Dataset shape: {df.shape}")
    print("\nFeatures:")
    for col in df.columns:
        print(f"- {col}")

if __name__ == '__main__':
    download_california_housing() 