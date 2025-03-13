#include "model.hpp"
#include <iostream>
#include <fstream>
#include <cxxopts.hpp>

// Function to parse command line arguments
cxxopts::ParseResult parse_arguments(int argc, char* argv[]) {
    try {
        cxxopts::Options options("house_price_prediction", "House price prediction using FHE");
        
        options.add_options()
            ("s,sample", "Path to input ciphertext", cxxopts::value<std::string>())
            ("c,cc", "Path to crypto context", cxxopts::value<std::string>())
            ("p,key_pub", "Path to public key", cxxopts::value<std::string>())
            ("m,key_mult", "Path to multiplication key", cxxopts::value<std::string>())
            ("r,key_rot", "Path to rotation key", cxxopts::value<std::string>())
            ("o,output", "Path for output result", cxxopts::value<std::string>())
            ("h,help", "Print help");
            
        auto result = options.parse(argc, argv);
        
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        
        return result;
    }
    catch (const cxxopts::OptionException& e) {
        std::cout << "Error parsing options: " << e.what() << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    auto args = parse_arguments(argc, argv);
    
    try {
        // Create predictor instance
        HousePricePredictor predictor;
        
        // Initialize with model weights and crypto context
        std::cout << "Initializing predictor..." << std::endl;
        predictor.Initialize("model_weights.json", args["cc"].as<std::string>());
        
        // Load keys
        std::cout << "Loading keys..." << std::endl;
        predictor.LoadKeys(
            args["key_pub"].as<std::string>(),
            args["key_mult"].as<std::string>(),
            args["key_rot"].as<std::string>()
        );
        
        // Load input ciphertext
        std::cout << "Loading input ciphertext..." << std::endl;
        std::ifstream input_file(args["sample"].as<std::string>(), std::ios::binary);
        if (!input_file.is_open()) {
            throw std::runtime_error("Could not open input file");
        }
        
        Ciphertext<DCRTPoly> encrypted_input;
        Serial::Deserialize(encrypted_input, input_file, SerType::BINARY);
        input_file.close();
        
        // Make prediction
        std::cout << "Making prediction..." << std::endl;
        auto result = predictor.Predict(encrypted_input);
        
        // Save result
        std::cout << "Saving result..." << std::endl;
        predictor.SaveResult(result, args["output"].as<std::string>());
        
        std::cout << "Prediction completed successfully!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 