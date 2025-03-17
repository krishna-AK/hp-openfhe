# app.py
import argparse
import sys
from openfhe import *
import json

class FHEInference:
    def __init__(self, context, pub_key):
        self.context = context
        self.pub_key = pub_key
        self.weights = {}
        self.poly_coeffs = {}
        self.architecture = None
        self.poly_degree = None
        # Precomputed plaintext coefficients for activation functions (no encryption)
        self.precomputed_coeffs = {}
        # Precomputed plaintext representations for weights and biases
        self.precomputed_plaintext_weights = {}
        self.precomputed_plaintext_biases = {}
        # Removed ThreadPoolExecutor instance to run computations sequentially

    def load_model(self, model_path):
        """
        Load model parameters for California Housing dataset prediction.
        Expected features include numerical values and one-hot encoded categorical columns.
        """
        with open(model_path, 'r') as f:
            model_data = json.load(f)

        self.weights = model_data["fhe_model"]["weights"]
        self.poly_coeffs = model_data["fhe_model"]["poly_coeffs"]

        # Use provided architecture or infer from weight keys.
        if "architecture" in model_data["fhe_model"]:
            self.architecture = model_data["fhe_model"]["architecture"]
        else:
            self.architecture = self._infer_architecture()

        # Determine the polynomial degree from the coefficients if there are any activations
        if self.poly_coeffs:
            first_act = list(self.poly_coeffs.keys())[0]
            self.poly_degree = len(self.poly_coeffs[first_act]) - 1
            print(f"Detected polynomial activation of degree {self.poly_degree}")
        else:
            self.poly_degree = 0
            print("No polynomial activations detected (linear model)")

        print(f"Model architecture: {self.architecture}")

        # Precompute plaintext activation coefficients (avoid encryption for performance)
        for act_key, coeff_list in self.poly_coeffs.items():
            plain_coeffs = []
            for coeff in coeff_list:
                pt = self.context.MakeCKKSPackedPlaintext([float(coeff)])
                plain_coeffs.append(pt)
            self.precomputed_coeffs[act_key] = plain_coeffs

        # Precompute plaintext representations for all weights and biases.
        self._precompute_plaintext_weights_biases()

    def _infer_architecture(self):
        """Infer model architecture from weight keys."""
        layer_names = {key.split('.')[0] for key in self.weights.keys()}
        num_layers = len(layer_names)
        input_dim = len(self.weights["layer1.weight"][0])

        # Check if we have a model with no hidden layers (just input to output)
        if num_layers == 1:
            return {"input_dim": input_dim, "hidden_dims": [], "num_layers": 1}

        # Otherwise, determine hidden dimensions
        hidden_dims = []
        for i in range(1, num_layers):
            layer_name = f"layer{i}"
            if f"{layer_name}.bias" in self.weights:
                hidden_dims.append(len(self.weights[f"{layer_name}.bias"]))
        return {"input_dim": input_dim, "hidden_dims": hidden_dims, "num_layers": num_layers}

    def _precompute_plaintext_weights_biases(self):
        """
        Precomputes plaintext representations for each layer's weights and biases.
        For the first layer, the entire weight vector is packed.
        For subsequent layers, individual weight elements are converted.
        """
        for key in self.weights:
            if key.endswith(".weight"):
                layer = key.split('.')[0]
                weights_matrix = self.weights[key]
                self.precomputed_plaintext_weights[key] = []

                # Special case for models with no hidden layers
                if layer == "layer1" and len(self.architecture["hidden_dims"]) == 0:
                    # For direct input-to-output, precompute plaintext for each weight element
                    for neuron_weights in weights_matrix:
                        neuron_plain_list = []
                        for w in neuron_weights:
                            pt = self.context.MakeCKKSPackedPlaintext([float(w)])
                            neuron_plain_list.append(pt)
                        self.precomputed_plaintext_weights[key].append(neuron_plain_list)
                elif layer == "layer1":
                    # Pack entire weight vector per neuron for first hidden layer
                    for neuron_weights in weights_matrix:
                        pt = self.context.MakeCKKSPackedPlaintext(neuron_weights)
                        self.precomputed_plaintext_weights[key].append(pt)
                else:
                    # For later layers, precompute plaintext for each weight element
                    for neuron_weights in weights_matrix:
                        neuron_plain_list = []
                        for w in neuron_weights:
                            pt = self.context.MakeCKKSPackedPlaintext([float(w)])
                            neuron_plain_list.append(pt)
                        self.precomputed_plaintext_weights[key].append(neuron_plain_list)
            elif key.endswith(".bias"):
                # Precompute plaintext biases
                bias_list = self.weights[key]
                self.precomputed_plaintext_biases[key] = []
                for b in bias_list:
                    pt = self.context.MakeCKKSPackedPlaintext([float(b)])
                    self.precomputed_plaintext_biases[key].append(pt)

    def polynomial_activation(self, x, act_key):
        """
        Evaluate polynomial activation using precomputed plaintext coefficients.
        For degree 1 activation: f(x) = a0 + a1 * x.
        For higher degrees, additional terms are added.
        The operations are adjusted so that the first operand of EvalAdd is always a ciphertext.
        """
        coeffs_pt = self.precomputed_coeffs[act_key]
        # For degree >= 1, start with the product of input and the linear coefficient.
        if self.poly_degree >= 1:
            result = self.context.EvalMult(x, coeffs_pt[1])
            # Add the constant term (plaintext) to the ciphertext.
            result = self.context.EvalAdd(result, coeffs_pt[0])
        else:
            # If no activation is specified, simply return x.
            result = x

        if self.poly_degree >= 2:
            x_squared = self.context.EvalMult(x, x)
            quadratic_term = self.context.EvalMult(x_squared, coeffs_pt[2])
            result = self.context.EvalAdd(result, quadratic_term)
        if self.poly_degree >= 3:
            # Compute x^3 using the previously computed x_squared.
            x_cubed = self.context.EvalMult(x_squared, x)
            cubic_term = self.context.EvalMult(x_cubed, coeffs_pt[3])
            result = self.context.EvalAdd(result, cubic_term)
        return result

    def sum_elements(self, x, length):
        return self.context.EvalSum(x, length)

    def compute_neuron(self, layer_idx, neuron_idx, inputs):
        """
        Compute one neuron's output for a given layer.
        For the first hidden layer, inputs are a packed ciphertext.
        For subsequent layers, inputs are lists of ciphertexts.
        """
        layer_name = f"layer{layer_idx}"
        weight_key = f"{layer_name}.weight"
        bias_key = f"{layer_name}.bias"
        if isinstance(inputs, list):
            # For layers beyond the first, use precomputed per-element plaintext weights.
            neuron_plain_weights = self.precomputed_plaintext_weights[weight_key][neuron_idx]
            temp = self.context.EvalMult(inputs[0], neuron_plain_weights[0])
            for j in range(1, len(neuron_plain_weights)):
                weighted = self.context.EvalMult(inputs[j], neuron_plain_weights[j])
                temp = self.context.EvalAdd(temp, weighted)
        else:
            # For the first hidden layer, use the precomputed packed plaintext weight.
            neuron_plain_weight = self.precomputed_plaintext_weights[weight_key][neuron_idx]
            num_slots = len(self.weights[weight_key][neuron_idx])
            temp = self.context.EvalMult(inputs, neuron_plain_weight)
            temp = self.sum_elements(temp, num_slots)

        # Add bias using the precomputed plaintext.
        neuron_bias_pt = self.precomputed_plaintext_biases[bias_key][neuron_idx]
        temp = self.context.EvalAdd(temp, neuron_bias_pt)

        # Apply activation if this is not the final output layer.
        if layer_idx < self.architecture["num_layers"]:
            act_key = f"act{layer_idx}"
            if act_key in self.poly_coeffs:
                temp = self.polynomial_activation(temp, act_key)
        return temp

    def inference(self, encrypted_input):
        """
        Perform FHE inference using sequential neuron computations.
        Returns the final encrypted output.
        """
        # Handle the case of a model with no hidden layers (direct input to output)
        if len(self.architecture["hidden_dims"]) == 0:
            return self._linear_model_inference(encrypted_input)

        layer_outputs = [None] * self.architecture["num_layers"]

        # Process the first hidden layer (packed input).
        first_layer_size = self.architecture["hidden_dims"][0]
        layer_outputs[0] = [None] * first_layer_size
        for i in range(first_layer_size):
            layer_outputs[0][i] = self.compute_neuron(1, i, encrypted_input)

        # Process subsequent hidden layers (inputs are lists of ciphertexts).
        for layer_idx in range(1, self.architecture["num_layers"] - 1):
            current_layer_size = self.architecture["hidden_dims"][layer_idx]
            layer_outputs[layer_idx] = [None] * current_layer_size
            inputs_for_layer = layer_outputs[layer_idx - 1]
            for i in range(current_layer_size):
                layer_outputs[layer_idx][i] = self.compute_neuron(layer_idx + 1, i, inputs_for_layer)

        # Process the output layer (assumed single neuron for regression).
        output_layer_idx = self.architecture["num_layers"]
        final_output = None
        last_hidden_outputs = layer_outputs[output_layer_idx - 2]
        weight_key = f"layer{output_layer_idx}.weight"
        bias_key = f"layer{output_layer_idx}.bias"
        for i, output in enumerate(last_hidden_outputs):
            neuron_weight_pt = self.precomputed_plaintext_weights[weight_key][0][i]
            weighted_output = self.context.EvalMult(output, neuron_weight_pt)
            if final_output is None:
                final_output = weighted_output
            else:
                final_output = self.context.EvalAdd(final_output, weighted_output)
        final_output = self.context.EvalAdd(final_output, self.precomputed_plaintext_biases[bias_key][0])
        return final_output

    def _linear_model_inference(self, encrypted_input):
        """
        Perform inference for a model with no hidden layers (direct input to output).
        This is a simple linear model: y = Wx + b
        """
        weight_key = "layer1.weight"
        bias_key = "layer1.bias"
        neuron_weights = self.precomputed_plaintext_weights[weight_key][0]
        bias = self.precomputed_plaintext_biases[bias_key][0]
        final_output = None
        for i, weight in enumerate(neuron_weights):
            feature = self.context.EvalRotate(encrypted_input, -i)
            weighted_feature = self.context.EvalMult(feature, weight)
            if final_output is None:
                final_output = weighted_feature
            else:
                final_output = self.context.EvalAdd(final_output, weighted_feature)
        final_output = self.context.EvalAdd(final_output, bias)
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
    # Initialize FHE inference
    fhe = FHEInference(context, pub_key)
    # Load model weights (plaintext encoding only)
    fhe.load_model("fhe_model_params.json")
    # Perform inference and return prediction
    result = fhe.inference(input)
    return result

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