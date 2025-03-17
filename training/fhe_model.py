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
        if degree == 1:
            self.coefficients = nn.Parameter(torch.tensor([0.0, 1.0]))
        elif degree == 2:
            self.coefficients = nn.Parameter(torch.tensor([0.0, 1.0, 0.01]))
        elif degree == 3:
            self.coefficients = nn.Parameter(torch.tensor([0.0, 1.0, 0.01, 0.001]))
        else:
            # Default to degree 2
            self.degree = 2
            self.coefficients = nn.Parameter(torch.tensor([0.0, 1.0, 0.01]))
    
    def forward(self, x):
        # Clip input for stability
        # x = torch.clamp(x, -1, 1)
        
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
    def __init__(self, input_dim=13, hidden_dims=[], poly_degree=1):
        super(FHECompatibleNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims) + 1  # Including output layer
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        
        # Handle case with no hidden layers (direct input to output)
        if len(hidden_dims) == 0:
            # Direct connection from input to output
            self.layers.append(nn.Linear(input_dim, 1))
            self._init_weights(self.layers[0], scale=1000.0)  # Scale for house prices
            # Initialize bias to a typical house price value
            self.layers[0].bias.data.fill_(91000.0)  # Typical house price in dataset
        else:
            # Input layer to first hidden layer
            self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
            self._init_weights(self.layers[0], scale=1.0)
            
            # Hidden layers
            for i in range(1, len(hidden_dims)):
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                self._init_weights(self.layers[i], scale=1.0)
            
            # Output layer
            self.layers.append(nn.Linear(hidden_dims[-1], 1))
            self._init_weights(self.layers[-1], scale=1000.0)  # Scale for house prices
            # Initialize bias to a typical house price value
            self.layers[-1].bias.data.fill_(91000.0)  # Typical house price in dataset
        
        # Polynomial activations for each hidden layer
        self.activations = nn.ModuleList()
        for _ in range(len(hidden_dims)):
            self.activations.append(PolynomialActivation(degree=poly_degree))
    
    def forward(self, x):
        # Handle case with no hidden layers
        if len(self.hidden_dims) == 0:
            # Direct connection from input to output (no activation)
            return self.layers[0](x)
        
        # Pass through all hidden layers with activations
        for i in range(len(self.hidden_dims)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
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
    # Extract weights and biases from all layers
    weights = {}
    poly_coeffs = {}
    
    # Process all layers
    for i, layer in enumerate(model.layers):
        layer_name = f"layer{i+1}"
        weights[f"{layer_name}.weight"] = layer.weight.data.tolist()
        weights[f"{layer_name}.bias"] = layer.bias.data.tolist()
    
    # Process all activations
    for i, activation in enumerate(model.activations):
        poly_coeffs[f"act{i+1}"] = activation.coefficients.data.tolist()
    
    # Ensure output layer weights and biases are correctly formatted
    output_layer_name = f"layer{len(model.layers)}"
    output_weights = weights[f"{output_layer_name}.weight"]
    output_bias = weights[f"{output_layer_name}.bias"]
    
    # Make sure output layer weights are a list with a single element (which is a list)
    if not isinstance(output_weights[0], list):
        weights[f"{output_layer_name}.weight"] = [output_weights]
    
    # Make sure output layer bias is a list, not a list of lists
    if isinstance(output_bias, list) and len(output_bias) > 0 and isinstance(output_bias[0], list):
        weights[f"{output_layer_name}.bias"] = output_bias[0]
    
    fhe_model = {
        'weights': weights,
        'poly_coeffs': poly_coeffs,
        'architecture': {
            'input_dim': model.input_dim,
            'hidden_dims': model.hidden_dims,
            'num_layers': model.num_layers
        }
    }
    return {'fhe_model': fhe_model} 