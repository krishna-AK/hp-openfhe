# California Housing Price Prediction

This project implements a machine learning model to predict housing prices in California using the California Housing dataset from scikit-learn. The model uses XGBoost with advanced feature engineering to achieve high prediction accuracy.

## Features

- Advanced feature engineering including:
  - Location-based features (distance to major cities)
  - Ratio features (rooms per household, bedrooms per room)
  - Polynomial features for important numerical columns
  - Binned features for age and income
  - Population density calculations

- XGBoost model with optimized hyperparameters
- Cross-validation for robust performance evaluation
- Standardized data preprocessing pipeline

## Performance

The model achieves:
- R² Score: 0.8736 (87.36%)
- Cross-validation R² Score: 0.8679 (86.79%)
- Mean Squared Error: 0.17
- Mean Absolute Error: 0.26

## Project Structure

```
.
├── training/
│   └── train_improved.py    # Training script with feature engineering
├── app/
│   ├── model.joblib        # Trained model
│   ├── model_metrics.json  # Model performance metrics
│   └── feature_names.json  # List of features used
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/california-housing-prediction.git
cd california-housing-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train the model:
```bash
cd training
python train_improved.py
```

The trained model and metrics will be saved in the `app/` directory.

## Dependencies

- numpy
- pandas
- scikit-learn
- xgboost
- joblib

## License

MIT License 