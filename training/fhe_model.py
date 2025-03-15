import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolynomialActivation(nn.Module):
    """FHE-friendly activation using low-degree polynomials"""
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree
        # Initialize coefficients for better scaling
        # For degree 3: [constant, linear, quadratic, cubic]
        self.coefficients = nn.Parameter(torch.tensor([0.0, 1.0, 0.01, 0.001]))
    
    def forward(self, x):
        # Clip input for stability
        x = torch.clamp(x, -10, 10)
        
        # Use Horner's method for polynomial evaluation
        # For a cubic: a0 + x*(a1 + x*(a2 + x*a3))
        result = self.coefficients[-1]  # Start with highest degree coefficient
        for i in range(self.degree - 1, -1, -1):
            result = self.coefficients[i] + x * result
        return result

def quantize_tensor(tensor, bits=10):
    """Quantize a single tensor"""
    # with torch.no_grad():
    #     abs_max = torch.abs(tensor).max() + 1e-8
    #     scale = (2**(bits-1) - 1) / abs_max
    #     return torch.round(tensor * scale) / scale
    return tensor  # Return tensor as is without quantization

def quantize_weights(model, bits=10):
    """Quantize weights to reduce computational depth"""
    # with torch.no_grad():
    #     for param in model.parameters():
    #         if param.requires_grad:
    #             abs_max = torch.abs(param.data).max() + 1e-8
    #             scale = (2**(bits-1) - 1) / abs_max
    #             param.data = torch.round(param.data * scale) / scale
    pass  # Do nothing for now

def preprocess_features(X):
    """Preprocess features for FHE computation"""
    # Keep original values - no scaling
    return X

class FHECompatibleNN(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=16):
        super(FHECompatibleNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # First layer
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self._init_weights(self.layer1, scale=1.0)
        
        # Second layer (output layer)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self._init_weights(self.layer2, scale=1.0)  # Moderate scale
        self.layer2.bias.data.fill_(0.0)  # Initialize bias to zero (to be learned)
        
        # Polynomial activation with degree 3
        self.act1 = PolynomialActivation(degree=3)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        return x
    
    def _init_weights(self, layer, scale=1.0):
        # Fix weight scaling by using in-place operation
        nn.init.xavier_uniform_(layer.weight.data)
        layer.weight.data.mul_(scale)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

def convert_to_fhe(model):
    """Convert trained model for FHE inference"""
    fhe_model = {
        'weights': {
            'layer1.weight': model.layer1.weight.data.tolist(),  # [hidden_dim, input_dim]
            'layer1.bias': model.layer1.bias.data.tolist(),      # [hidden_dim]
            'layer2.weight': model.layer2.weight.data.tolist(),  # [1, hidden_dim]
            'layer2.bias': model.layer2.bias.data.tolist()       # [1]
        },
        'poly_coeffs': {
            'act1': model.act1.coefficients.data.tolist()       # [degree + 1]
        }
    }
    return {'fhe_model': fhe_model} 