#pragma once

#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

using namespace lbcrypto;
using json = nlohmann::json;

class HousePricePredictor {
public:
    HousePricePredictor() = default;
    
    // Initialize the model with weights and crypto context
    void Initialize(const std::string& weights_path, const std::string& cc_path);
    
    // Load keys
    void LoadKeys(const std::string& pub_key_path, const std::string& mult_key_path, const std::string& rot_key_path);
    
    // Predict house price using encrypted input
    Ciphertext<DCRTPoly> Predict(const Ciphertext<DCRTPoly>& encrypted_input);
    
    // Save result
    void SaveResult(const Ciphertext<DCRTPoly>& result, const std::string& output_path);

private:
    // Model parameters
    std::vector<double> coefficients;
    double intercept;
    std::vector<double> scaler_mean;
    std::vector<double> scaler_scale;
    std::vector<std::string> feature_names;
    
    // Cryptographic context and keys
    CryptoContext<DCRTPoly> cc;
    PublicKey<DCRTPoly> public_key;
    EvalKey<DCRTPoly> eval_mult_key;
    std::vector<EvalKey<DCRTPoly>> eval_rot_keys;
    
    // Helper functions
    void LoadModelWeights(const std::string& weights_path);
    void LoadCryptoContext(const std::string& cc_path);
    Plaintext CreatePlaintextVector(const std::vector<double>& vec);
}; 