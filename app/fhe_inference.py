import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor

class FHEInference:
    def __init__(self, context, pub_key):
        self.context = context
        self.pub_key = pub_key
        self.layer1_weights = None
        self.layer1_bias = None
        self.layer2_weights = None
        self.layer2_bias = None
        self.act1_coeffs = None
        self.poly_degree = None

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

        # Determine polynomial degree from coefficients length
        # For degree 1: [a0, a1] (2 coefficients)
        # For degree 2: [a0, a1, a2] (3 coefficients)
        # For degree 3: [a0, a1, a2, a3] (4 coefficients)
        self.poly_degree = len(self.act1_coeffs) - 1
        print(f"Detected polynomial activation of degree {self.poly_degree}")

    def polynomial_activation(self, x, coeffs):
        """
        Evaluate polynomial using direct computation for better FHE compatibility.
        For degree 1: a0 + a1*x
        For degree 2: a0 + a1*x + a2*x^2
        For degree 3: a0 + a1*x + a2*x^2 + a3*x^3
        """
        # Encrypt the constant term so that result is a ciphertext.
        a0_plain = self.context.MakeCKKSPackedPlaintext([float(coeffs[0])])
        result = self.context.Encrypt(self.pub_key, a0_plain)

        # Add linear term
        if len(coeffs) > 1:
            a1 = self.context.MakeCKKSPackedPlaintext([float(coeffs[1])])
            linear_term = self.context.EvalMult(x, a1)
            result = self.context.EvalAdd(result, linear_term)

        # Add quadratic term if degree >= 2
        if len(coeffs) > 2:
            a2 = self.context.MakeCKKSPackedPlaintext([float(coeffs[2])])
            x_squared = self.context.EvalMult(x, x)
            quadratic_term = self.context.EvalMult(x_squared, a2)
            result = self.context.EvalAdd(result, quadratic_term)

        # Add cubic term if degree >= 3
        if len(coeffs) > 3:
            a3 = self.context.MakeCKKSPackedPlaintext([float(coeffs[3])])
            # x_squared is computed above when degree>=2
            x_cubed = self.context.EvalMult(x_squared, x)
            cubic_term = self.context.EvalMult(x_cubed, a3)
            result = self.context.EvalAdd(result, cubic_term)

        return result

    def sum_elements(self, x, length):
        """
        Sum elements using binary tree reduction.
        Instead of rotating by 1 repeatedly, this method performs O(log n) rotations.
        """
        result = x
        shift = 1
        while shift < length:
            rotated = self.context.EvalRotate(result, shift)
            result = self.context.EvalAdd(result, rotated)
            shift *= 2
        return result

    def compute_neuron(self, i, encrypted_input):
        """
        Compute the output for one neuron in the hidden layer.
        """
        # Linear transformation:
        weight = self.layer1_weights[i]
        weight_plain = self.context.MakeCKKSPackedPlaintext(weight.tolist())
        temp = self.context.EvalMult(encrypted_input, weight_plain)

        # Summing elements (using the optimized binary reduction)
        temp = self.sum_elements(temp, len(weight))

        # Add bias:
        bias_plain = self.context.MakeCKKSPackedPlaintext([float(self.layer1_bias[i])])
        temp = self.context.EvalAdd(temp, bias_plain)

        # Apply polynomial activation:
        temp = self.polynomial_activation(temp, self.act1_coeffs)
        return temp

    def inference(self, encrypted_input):
        """
        Perform FHE inference on encrypted California Housing features.
        Returns: Encrypted predicted house price in the first slot.
        """
        # First layer: Linear transformation + activation via parallel processing
        num_neurons = len(self.layer1_weights)
        layer1_outputs = [None] * num_neurons

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.compute_neuron, i, encrypted_input): i
                       for i in range(num_neurons)}
            for future in futures:
                i = futures[future]
                layer1_outputs[i] = future.result()

        # Second layer: Linear transformation (single output)
        final_output = None

        for i, output in enumerate(layer1_outputs):
            weight = self.layer2_weights[0][i]
            weight_plain = self.context.MakeCKKSPackedPlaintext([float(weight)])
            weighted_output = self.context.EvalMult(output, weight_plain)
            if final_output is None:
                final_output = weighted_output
            else:
                final_output = self.context.EvalAdd(final_output, weighted_output)

        # Add bias for the second layer
        bias_plain = self.context.MakeCKKSPackedPlaintext([float(self.layer2_bias[0])])
        final_output = self.context.EvalAdd(final_output, bias_plain)

        return final_output