import argparse
import sys

from openfhe import *
import numpy as np
import json

class FHEInference:
    def __init__(self, context, pub_key):
        self.context = context
        self.pub_key = pub_key
        self.layer1_weights = None
        self.layer1_bias = None
        self.layer2_weights = None
        self.layer2_bias = None
        self.act1_coeffs = None
        self.act2_coeffs = None

    def load_model(self, model_path):
        """
        Load model parameters for California Housing dataset prediction
        Features:
        - longitude, latitude
        - housing_median_age
        - total_rooms, total_bedrooms
        - population, households
        - median_income
        - ocean_proximity (one-hot encoded, 5 columns)
        """
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Extract weights and coefficients
        weights_data = model_data["fhe_model"]["weights"]
        self.layer1_weights = np.array(weights_data["layer1.weight"])
        self.layer1_bias = np.array(weights_data["layer1.bias"])
        self.layer2_weights = np.array(weights_data["layer2.weight"])
        self.layer2_bias = np.array(weights_data["layer2.bias"])
        
        poly_data = model_data["fhe_model"]["poly_coeffs"]
        self.act1_coeffs = np.array(poly_data["act1"])
        # Removed act2 coefficients as they are not used

    def polynomial_activation(self, x, coeffs):
        """Evaluate polynomial using Horner's method: a0 + x(a1 + x(a2))"""
        # Start with highest degree coefficient
        result = self.context.MakeCKKSPackedPlaintext([coeffs[-1]])
        
        # Apply Horner's method
        for i in range(len(coeffs)-2, -1, -1):
            # Multiply by x
            result = self.context.EvalMult(x, result)
            # Add next coefficient
            coeff_plain = self.context.MakeCKKSPackedPlaintext([coeffs[i]])
            result = self.context.EvalAdd(result, coeff_plain)
        
        return result

    def sum_elements(self, x, length):
        """Sum elements using only rotation index 1"""
        result = x
        current_sum = x
        for _ in range(length - 1):
            rotated = self.context.EvalRotate(current_sum, 1)
            result = self.context.EvalAdd(result, rotated)
            current_sum = rotated
        return result

    def inference(self, encrypted_input):
        """
        Perform FHE inference on encrypted California Housing features
        Returns: Encrypted predicted house price in the first slot
        """
        # First layer: Linear transformation + activation
        layer1_outputs = []
        for i in range(len(self.layer1_weights)):
            # Linear transformation
            weight_plain = self.context.MakeCKKSPackedPlaintext(self.layer1_weights[i])
            temp = self.context.EvalMult(encrypted_input, weight_plain)
            
            # Sum all elements using rotation index 1
            temp = self.sum_elements(temp, len(self.layer1_weights[i]))
            
            # Add bias
            bias_plain = self.context.MakeCKKSPackedPlaintext([self.layer1_bias[i]])
            temp = self.context.EvalAdd(temp, bias_plain)
            
            # Polynomial activation
            temp = self.polynomial_activation(temp, self.act1_coeffs)
            layer1_outputs.append(temp)
        
        # Second layer: Linear transformation (single output)
        final_output = None
        
        # Process each neuron in the hidden layer
        for i, output in enumerate(layer1_outputs):
            # Get the weight for this neuron
            weight = self.layer2_weights[0][i]
            weight_plain = self.context.MakeCKKSPackedPlaintext([weight])
            
            # Multiply the neuron output by its weight
            weighted_output = self.context.EvalMult(output, weight_plain)
            
            # Add to the final output
            if final_output is None:
                final_output = weighted_output
            else:
                final_output = self.context.EvalAdd(final_output, weighted_output)
        
        # Add bias
        bias_plain = self.context.MakeCKKSPackedPlaintext([self.layer2_bias[0]])
        final_output = self.context.EvalAdd(final_output, bias_plain)
        
        return final_output

class CKKSParser:
    def __init__(self):
        self.context = CryptoContext()
        self.public_key = None
        self.input = None

    def load(self, args):
        self.init_context(args.cc)
        self.init_public_key(args.key_pub)
        self.init_eval_mult_key(args.key_mult)
        self.init_rotation_key(args.key_rot)
        self.init_ciphertext(args.sample)

    def init_context(self, context_path):
        self.context, ok = DeserializeCryptoContext(context_path, BINARY)
        if not ok:
            raise Exception('load crypto context')

    def init_public_key(self, public_key_path):
        self.public_key, ok = DeserializePublicKey(public_key_path, BINARY)
        if not ok:
            raise Exception('load public key')

    def init_eval_mult_key(self, eval_key_path):
        if not self.context.DeserializeEvalMultKey(eval_key_path, BINARY):
            raise Exception('load mult key')

    def init_rotation_key(self, rotation_key_path):
        if not self.context.DeserializeEvalAutomorphismKey(rotation_key_path, BINARY):
            raise Exception('load rotation key')
        
    def init_ciphertext(self, ciphertext_path):
        self.input, ok = DeserializeCiphertext(ciphertext_path, BINARY)
        if not ok:
            raise Exception('load ciphertext')

def solve(input, context, pub_key):
    """
    Solve the California Housing price prediction challenge
    Input: Encrypted vector of 13 features (8 numerical + 5 one-hot encoded)
    Output: Encrypted predicted house price in the first slot
    """
    # Initialize FHE inference
    fhe = FHEInference(context, pub_key)
    
    # Load model weights
    fhe.load_model("fhe_model_params.json")
    
    # Perform inference and return prediction
    return fhe.inference(input)

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--key_pub')
        parser.add_argument('--key_mult')
        parser.add_argument('--key_rot')
        parser.add_argument('--cc')
        parser.add_argument('--sample')
        parser.add_argument('--output')
        args = parser.parse_args()

        a = CKKSParser()
        a.load(args)
        
        a.context.Enable(PKESchemeFeature.PKE)
        a.context.Enable(PKESchemeFeature.KEYSWITCH)
        a.context.Enable(PKESchemeFeature.LEVELEDSHE)
        a.context.Enable(PKESchemeFeature.ADVANCEDSHE)

        answer = solve(a.input, a.context, a.public_key)

        if not SerializeToFile(args.output, answer, BINARY):
            raise Exception('output serialization failed')

    except Exception as err:
        print(f'execution error: {err}')
        sys.exit(1) 