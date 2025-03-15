# FHE-Compatible Housing Price Prediction Model

This project implements a neural network model for predicting California housing prices that is compatible with Fully Homomorphic Encryption (FHE). The model achieves high accuracy while maintaining the ability to be encrypted for secure inference.

## Features

- **High Accuracy**: Achieves 82.86% R² score on the California Housing dataset
- **FHE Compatibility**: All operations are designed to be compatible with FHE encryption
- **Advanced Architecture**: Uses residual connections and careful weight initialization
- **Robust Training**: Implements early stopping, learning rate scheduling, and gradient clipping
- **Feature Engineering**: Creates FHE-friendly features using basic arithmetic operations

## Model Architecture

The neural network is designed with FHE compatibility in mind:

- **Input Layer**: Takes engineered features
- **Hidden Layers**: 
  - 512 → 512 (with residual connection)
  - 256 → 256 (with residual connection)
  - 128 → 128 (with residual connection)
  - 64 → 1
- **Activations**: ReLU (to be replaced with polynomial approximation in FHE)
- **Regularization**: Progressive dropout (0.3 → 0.25 → 0.2 → 0.15)

## Feature Engineering

The model uses several types of FHE-friendly features:

1. **Log Transformations**:
   - Log income
   - Log rooms
   - Log bedrooms
   - Log population
   - Log occupancy

2. **Simple Ratios**:
   - Rooms per person
   - Bedrooms per person
   - Income per room
   - Occupancy density
   - Bedroom ratio

3. **Location Features**:
   - Coastal distance
   - Adjusted latitude
   - Location score

4. **Binned Features**:
   - Age by decade
   - Age by century

5. **Feature Interactions**:
   - Income-age interaction
   - Rooms-age interaction
   - Location value
   - Density-income interaction
   - Bedroom density

6. **Polynomial Terms**:
   - Quadratic terms for income, rooms, and location
   - Cubic terms for income and rooms

## Training Process

The model is trained with the following optimizations:

- **Batch Size**: 256
- **Learning Rate**: 0.0005
- **Optimizer**: AdamW with weight decay (0.005)
- **Loss Function**: Custom loss combining MSE and relative error
- **Learning Rate Scheduling**: ReduceLROnPlateau with patience=20
- **Early Stopping**: Patience=40
- **Gradient Clipping**: max_norm=0.5

## FHE Compatibility

All operations in the model are designed to be FHE-compatible:

1. **Feature Engineering**:
   - Uses only basic arithmetic operations
   - Precomputable transformations
   - No complex mathematical functions

2. **Network Operations**:
   - Linear layers (matrix multiplication and addition)
   - ReLU activations (to be replaced with polynomial approximation)
   - Residual connections (addition only)

3. **Loss Function**:
   - Based on basic arithmetic operations
   - No complex mathematical functions

## Performance Metrics

- **R² Score**: 0.8286 (82.86%)
- **MSE**: 0.2246
- **MAE**: 0.2987

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.2
torch>=2.0.0
joblib>=1.1.0
matplotlib>=3.4.0
```

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python training/train_improved.py
```

3. The model and parameters will be saved in the `app/` directory:
   - `model.pt`: PyTorch model state
   - `model_metrics.json`: Performance metrics
   - `scaler_params.json`: Feature scaling parameters
   - `training_history.png`: Training visualization

## Future Work

1. Implement FHE encryption for secure inference
2. Add model quantization for better FHE performance
3. Optimize polynomial approximations for ReLU
4. Add support for batch inference
5. Implement model compression techniques

## License

MIT License 