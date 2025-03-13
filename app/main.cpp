#include "fhe_inference.hpp"
#include <openfhe.h>
#include <iostream>
#include <string>

using namespace lbcrypto;

Ciphertext<DCRTPoly> solve(const Ciphertext<DCRTPoly>& input, const CryptoContext<DCRTPoly>& context, const PublicKey<DCRTPoly>& pub_key) {
    // Initialize FHE inference
    FHEInference fhe;
    fhe.setupFHE(3);  // Set multiplicative depth
    
    // Load model weights
    fhe.loadModel("fhe_model_params.pth");
    
    // Perform inference
    return fhe.inference(input);
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        if (argc != 7) {
            std::cerr << "Usage: " << argv[0] << " --key_pub <key_pub> --key_mult <key_mult> --key_rot <key_rot> --cc <cc> --sample <sample> --output <output>" << std::endl;
            return 1;
        }

        // Load crypto context
        CryptoContext<DCRTPoly> context;
        if (!DeserializeCryptoContext(argv[4], context, BINARY)) {
            throw std::runtime_error("Failed to load crypto context");
        }

        // Load public key
        PublicKey<DCRTPoly> pub_key;
        if (!DeserializePublicKey(argv[1], pub_key, BINARY)) {
            throw std::runtime_error("Failed to load public key");
        }

        // Load evaluation keys
        if (!context->DeserializeEvalMultKey(argv[2], BINARY)) {
            throw std::runtime_error("Failed to load evaluation multiplication key");
        }
        if (!context->DeserializeEvalAutomorphismKey(argv[3], BINARY)) {
            throw std::runtime_error("Failed to load evaluation rotation key");
        }

        // Load input ciphertext
        Ciphertext<DCRTPoly> input;
        if (!DeserializeCiphertext(argv[5], input, BINARY)) {
            throw std::runtime_error("Failed to load input ciphertext");
        }

        // Enable required features
        context->Enable(PKE);
        context->Enable(KEYSWITCH);
        context->Enable(LEVELEDSHE);
        context->Enable(ADVANCEDSHE);

        // Perform inference
        Ciphertext<DCRTPoly> result = solve(input, context, pub_key);

        // Save result
        if (!SerializeToFile(argv[6], result, BINARY)) {
            throw std::runtime_error("Failed to save result");
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 