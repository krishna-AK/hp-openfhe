import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolynomialActivation(nn.Module):
    """FHE-friendly activation using low-degree polynomials"""
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        # Initialize coefficients with small values
        self.coefficients = nn.Parameter(torch.ones(degree + 1) * 0.01)
    
    def forward(self, x):
        # Clip input to prevent numerical instability
        x = torch.clamp(x, -10, 10)
        result = 0
        for i in range(self.degree + 1):
            result = result + self.coefficients[i] * (x ** i)
        return result

def quantize_tensor(tensor, bits=8):
    """Quantize a single tensor"""
    with torch.no_grad():
        scale = (2**bits - 1) / (tensor.max() - tensor.min() + 1e-8)
        return torch.round(tensor * scale) / scale

def quantize_weights(model, bits=8):
    """Quantize weights to reduce computational depth"""
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                # Scale to [0, 2^bits-1]
                scale = (2**bits - 1) / (param.max() - param.min() + 1e-8)
                param.data = torch.round(param.data * scale) / scale

def preprocess_features(X):
    """Preprocess features to minimize FHE computation"""
    # Scale to [-0.5, 0.5] to reduce multiplication depth
    X = X / 2 - 0.5
    
    # Clip values to prevent numerical instability
    X = torch.clamp(X, -0.5, 0.5)
    
    # Pre-compute feature products that will be needed
    X_squared = X ** 2
    X_cubed = X ** 3
    
    return torch.cat([X, X_squared, X_cubed], dim=1)

class FHECompatibleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Simplified architecture for FHE
        self.layer1 = nn.Linear(input_dim, 256)
        self.act1 = PolynomialActivation(degree=2)
        
        self.layer2 = nn.Linear(256, 128)
        self.act2 = PolynomialActivation(degree=2)
        
        self.layer3 = nn.Linear(128, 64)
        self.act3 = PolynomialActivation(degree=2)
        
        self.layer4 = nn.Linear(64, 1)
        
        # Initialize with small weights
        self._init_weights()
        # Quantize weights
        quantize_weights(self)
    
    def _init_weights(self):
        """Initialize weights with small values"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization with small bounds
                bound = np.sqrt(2.0 / (m.weight.size(0) + m.weight.size(1))) * 0.1
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Clip input to prevent numerical instability
        x = torch.clamp(x, -10, 10)
        
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.act3(x)
        x = self.layer4(x)
        return x

def convert_to_fhe(model):
    """Convert trained model for FHE inference"""
    # Extract and quantize weights
    weights = {}
    for name, param in model.named_parameters():
        if isinstance(param, nn.Parameter):
            weights[name] = quantize_tensor(param.data).tolist()
    
    # Convert activation functions to polynomial approximations
    poly_coeffs = {}
    for name, module in model.named_modules():
        if isinstance(module, PolynomialActivation):
            poly_coeffs[name] = quantize_tensor(module.coefficients.data).tolist()
    
    return {
        'weights': weights,
        'poly_coeffs': poly_coeffs,
        'architecture': {
            'input_dim': model.layer1.in_features,
            'hidden_dims': [256, 128, 64],
            'output_dim': 1
        }
    } 