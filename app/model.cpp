#include "model.hpp"
#include <fstream>
#include <stdexcept>

void HousePricePredictor::Initialize(const std::string& weights_path, const std::string& cc_path) {
    LoadModelWeights(weights_path);
    LoadCryptoContext(cc_path);
}

void HousePricePredictor::LoadKeys(const std::string& pub_key_path, const std::string& mult_key_path, const std::string& rot_key_path) {
    if (!cc) {
        throw std::runtime_error("Crypto context not initialized");
    }
    
    // Load public key
    std::ifstream pub_key_file(pub_key_path, std::ios::binary);
    if (!pub_key_file.is_open()) {
        throw std::runtime_error("Could not open public key file");
    }
    Serial::Deserialize(public_key, pub_key_file, SerType::BINARY);
    pub_key_file.close();
    
    // Load multiplication key
    std::ifstream mult_key_file(mult_key_path, std::ios::binary);
    if (!mult_key_file.is_open()) {
        throw std::runtime_error("Could not open multiplication key file");
    }
    Serial::Deserialize(eval_mult_key, mult_key_file, SerType::BINARY);
    mult_key_file.close();
    
    // Load rotation key
    std::ifstream rot_key_file(rot_key_path, std::ios::binary);
    if (!rot_key_file.is_open()) {
        throw std::runtime_error("Could not open rotation key file");
    }
    
    EvalKey<DCRTPoly> rot_key;
    Serial::Deserialize(rot_key, rot_key_file, SerType::BINARY);
    eval_rot_keys.push_back(rot_key);
    rot_key_file.close();
}

void HousePricePredictor::LoadModelWeights(const std::string& weights_path) {
    std::ifstream file(weights_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open weights file");
    }
    
    json weights_json;
    file >> weights_json;
    
    coefficients = weights_json["coefficients"].get<std::vector<double>>();
    intercept = weights_json["intercept"].get<double>();
    feature_names = weights_json["feature_names"].get<std::vector<std::string>>();
    scaler_mean = weights_json["scaler_mean"].get<std::vector<double>>();
    scaler_scale = weights_json["scaler_scale"].get<std::vector<double>>();
}

void HousePricePredictor::LoadCryptoContext(const std::string& cc_path) {
    std::ifstream file(cc_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open crypto context file");
    }
    
    Serial::Deserialize(cc, file, SerType::BINARY);
    file.close();
}

Plaintext HousePricePredictor::CreatePlaintextVector(const std::vector<double>& vec) {
    return cc->MakeCKKSPackedPlaintext(vec);
}

Ciphertext<DCRTPoly> HousePricePredictor::Predict(const Ciphertext<DCRTPoly>& encrypted_input) {
    // Create plaintext vectors for scaling
    auto scale_plaintext = CreatePlaintextVector(scaler_scale);
    auto mean_plaintext = CreatePlaintextVector(scaler_mean);
    auto coef_plaintext = CreatePlaintextVector(coefficients);
    
    // Scale the input: (x - mean) / scale
    auto scaled_input = cc->EvalSub(encrypted_input, mean_plaintext);
    scaled_input = cc->EvalDiv(scaled_input, scale_plaintext);
    
    // Compute dot product with coefficients
    auto result = cc->EvalMult(scaled_input, coef_plaintext);
    
    // Add intercept
    auto intercept_plaintext = cc->MakeCKKSPackedPlaintext({intercept});
    result = cc->EvalAdd(result, intercept_plaintext);
    
    return result;
}

void HousePricePredictor::SaveResult(const Ciphertext<DCRTPoly>& result, const std::string& output_path) {
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file");
    }
    
    Serial::Serialize(result, file, SerType::BINARY);
    file.close();
} 