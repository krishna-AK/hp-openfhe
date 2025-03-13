#pragma once

#include <openfhe.h>
#include <vector>
#include <string>
#include <memory>

using namespace lbcrypto;

class FHEInference {
public:
    FHEInference();
    
    // Initialize FHE context and keys
    void setupFHE(uint32_t multiplicative_depth);
    
    // Load model weights from file
    void loadModel(const std::string& model_path);
    
    // Perform inference on encrypted input
    Ciphertext<DCRTPoly> inference(const Ciphertext<DCRTPoly>& encrypted_input);
    
    // Helper functions
    Ciphertext<DCRTPoly> polynomialActivation(const Ciphertext<DCRTPoly>& x, const std::vector<double>& coeffs);
    Plaintext vectorToPlaintext(const std::vector<double>& vec);
    std::vector<double> plaintextToVector(const Plaintext& pt);

private:
    // FHE context and keys
    CryptoContext<DCRTPoly> context;
    KeyPair<DCRTPoly> keyPair;
    PublicKey<DCRTPoly> publicKey;
    
    // Model parameters
    std::vector<std::vector<double>> weights;  // Layer weights
    std::vector<std::vector<double>> poly_coeffs;  // Polynomial activation coefficients
    std::vector<double> scaler_mean;
    std::vector<double> scaler_scale;
}; 