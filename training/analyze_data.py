import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_analyze_data(data_path):
    """Load and perform basic analysis of the dataset."""
    # Load data
    df = pd.read_csv(data_path)
    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print("\nFeature Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    # Analyze feature distributions
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.select_dtypes(include=[np.number]).columns, 1):
        plt.subplot(3, 4, i)
        sns.histplot(df[column], kde=True)
        plt.title(f'{column} Distribution')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/feature_distributions.png')
    plt.close()
    
    # Correlation analysis
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig('data/correlation_matrix.png')
    plt.close()
    
    # Analyze outliers
    print("\n=== Outlier Analysis ===")
    for column in df.select_dtypes(include=[np.number]).columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)][column]
        print(f"\n{column}:")
        print(f"Number of outliers: {len(outliers)}")
        print(f"Percentage of outliers: {(len(outliers) / len(df)) * 100:.2f}%")
    
    # Analyze target variable
    print("\n=== Target Variable Analysis ===")
    print("Median House Value Statistics:")
    print(df['median_house_value'].describe())
    
    # Calculate skewness
    print("\n=== Feature Skewness ===")
    skewness = df.skew()
    print(skewness)
    
    return df

def suggest_improvements(df):
    """Analyze the data and suggest improvements."""
    suggestions = []
    
    # Check for highly correlated features
    correlation_matrix = df.corr()
    high_correlation = np.where(np.abs(correlation_matrix) > 0.7)
    high_correlation = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y])
                       for x, y in zip(*high_correlation) if x != y and x < y]
    
    if high_correlation:
        suggestions.append("\n=== Feature Correlation Suggestions ===")
        suggestions.append("Highly correlated feature pairs:")
        for feat1, feat2, corr in high_correlation:
            suggestions.append(f"- {feat1} & {feat2}: {corr:.3f}")
    
    # Check for skewed features
    skewed_features = df.select_dtypes(include=[np.number]).apply(lambda x: stats.skew(x)).abs()
    highly_skewed = skewed_features[skewed_features > 1]
    
    if not highly_skewed.empty:
        suggestions.append("\n=== Feature Transformation Suggestions ===")
        suggestions.append("Consider log transformation for highly skewed features:")
        for feature, skew in highly_skewed.items():
            suggestions.append(f"- {feature}: skewness = {skew:.3f}")
    
    # Suggest feature engineering
    suggestions.append("\n=== Feature Engineering Suggestions ===")
    suggestions.append("Consider creating the following features:")
    suggestions.append("- rooms_per_household = total_rooms / households")
    suggestions.append("- bedrooms_per_room = total_bedrooms / total_rooms")
    suggestions.append("- population_per_household = population / households")
    suggestions.append("- Add polynomial features for latitude and longitude")
    suggestions.append("- Create interaction terms between related features")
    
    return "\n".join(suggestions)

if __name__ == "__main__":
    data_path = 'data/california_housing.csv'
    print("=== Starting Data Analysis ===")
    df = load_and_analyze_data(data_path)
    
    print("\n=== Improvement Suggestions ===")
    suggestions = suggest_improvements(df)
    print(suggestions)
    
    # Save suggestions to file
    with open('data/analysis_suggestions.txt', 'w') as f:
        f.write(suggestions) 