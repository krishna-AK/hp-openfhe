#include "fhe_inference.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

FHEInference::FHEInference() {}

void FHEInference::setupFHE(uint32_t multiplicative_depth) {
    // Set up FHE parameters
    CCParams<CryptoContextBFVRNS> parameters;
    parameters.SetPlaintextModulus(65537);
    parameters.SetMultiplicativeDepth(multiplicative_depth);
    
    // Create crypto context
    context = GenCryptoContext(parameters);
    
    // Enable features
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    
    // Generate keys
    keyPair = context->KeyGen();
    publicKey = keyPair.publicKey;
}

void FHEInference::loadModel(const std::string& model_path) {
    // Load model weights from file
    std::ifstream file(model_path);
    json model_data;
    file >> model_data;
    
    // Extract weights and coefficients
    weights = model_data["fhe_model"]["weights"].get<std::vector<std::vector<double>>>();
    poly_coeffs = model_data["fhe_model"]["poly_coeffs"].get<std::vector<std::vector<double>>>();
    scaler_mean = model_data["scaler"]["mean"].get<std::vector<double>>();
    scaler_scale = model_data["scaler"]["scale"].get<std::vector<double>>();
}

Ciphertext<DCRTPoly> FHEInference::inference(const Ciphertext<DCRTPoly>& encrypted_input) {
    // Start with encrypted input
    Ciphertext<DCRTPoly> result = encrypted_input;
    
    // Apply model layers
    for (size_t i = 0; i < weights.size(); ++i) {
        // Linear layer
        Plaintext weight_plain = vectorToPlaintext(weights[i]);
        result = context->EvalMult(result, weight_plain);
        
        // Polynomial activation (except for last layer)
        if (i < weights.size() - 1 && !poly_coeffs.empty()) {
            result = polynomialActivation(result, poly_coeffs[i]);
        }
    }
    
    return result;
}

Ciphertext<DCRTPoly> FHEInference::polynomialActivation(const Ciphertext<DCRTPoly>& x, const std::vector<double>& coeffs) {
    // Evaluate polynomial using Horner's method
    Ciphertext<DCRTPoly> result = context->EvalMult(x, vectorToPlaintext({coeffs[0]}));
    
    for (size_t i = 1; i < coeffs.size(); ++i) {
        result = context->EvalMult(result, x);
        result = context->EvalAdd(result, vectorToPlaintext({coeffs[i]}));
    }
    
    return result;
}

Plaintext FHEInference::vectorToPlaintext(const std::vector<double>& vec) {
    return context->MakeCKKSPackedPlaintext(vec);
}

std::vector<double> FHEInference::plaintextToVector(const Plaintext& pt) {
    return pt->GetRealPackedValue();
} 