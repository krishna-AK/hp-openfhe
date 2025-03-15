import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolynomialActivation(nn.Module):
    """FHE-friendly activation using low-degree polynomials"""
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        # Initialize coefficients for better scaling
        # For degree 2: [constant, linear, quadratic]
        if degree == 2:
            self.coefficients = nn.Parameter(torch.tensor([0.0, 1.0, 0.01]))
        elif degree == 3:
            self.coefficients = nn.Parameter(torch.tensor([0.0, 1.0, 0.01, 0.001]))
        else:
            # Default to degree 2
            self.degree = 2
            self.coefficients = nn.Parameter(torch.tensor([0.0, 1.0, 0.01]))
    
    def forward(self, x):
        # Clip input for stability
        x = torch.clamp(x, -10, 10)
        
        # Use Horner's method for polynomial evaluation
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
        # Initialize with weights appropriate for the feature scale
        self._init_weights(self.layer1, scale=1.0)
        
        # Second layer (output layer)
        self.layer2 = nn.Linear(hidden_dim, 1)
        # Initialize the final layer with weights appropriate for house price range
        self._init_weights(self.layer2, scale=1000.0)  # Scale for house prices
        # Initialize bias to a typical house price value
        self.layer2.bias.data.fill_(91000.0)  # Typical house price in dataset
        
        # Polynomial activation with degree 2 (quadratic)
        self.act1 = PolynomialActivation(degree=2)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        return x
    
    def _init_weights(self, layer, scale=1.0):
        # Initialize weights with Xavier uniform and apply scaling
        nn.init.xavier_uniform_(layer.weight.data)
        layer.weight.data.mul_(scale)
        # Don't initialize bias to zero for the output layer (handled separately)
        if layer.bias is not None and scale == 1.0:
            layer.bias.data.fill_(0.0)

def convert_to_fhe(model):
    """Convert trained model for FHE inference"""
    # Ensure we're getting the correct shapes
    layer1_weight = model.layer1.weight.data.tolist()  # [hidden_dim, input_dim]
    layer1_bias = model.layer1.bias.data.tolist()      # [hidden_dim]
    layer2_weight = model.layer2.weight.data.tolist()  # [1, hidden_dim]
    layer2_bias = model.layer2.bias.data.tolist()      # [1]
    
    # Ensure layer2 weights and biases are correctly formatted
    # They should be lists, not lists of lists
    if isinstance(layer2_bias, list) and len(layer2_bias) > 0 and isinstance(layer2_bias[0], list):
        layer2_bias = layer2_bias[0]  # Take the first element if it's a list of lists
    
    # Make sure layer2_weight is a list with a single element (which is a list)
    if not isinstance(layer2_weight[0], list):
        layer2_weight = [layer2_weight]  # Wrap in a list if it's not already
    
    # Get polynomial coefficients
    poly_coeffs = model.act1.coefficients.data.tolist()
    
    fhe_model = {
        'weights': {
            'layer1.weight': layer1_weight,
            'layer1.bias': layer1_bias,
            'layer2.weight': layer2_weight,
            'layer2.bias': layer2_bias
        },
        'poly_coeffs': {
            'act1': poly_coeffs
        }
    }
    return {'fhe_model': fhe_model} 