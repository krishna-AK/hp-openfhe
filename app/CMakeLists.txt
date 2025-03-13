cmake_minimum_required(VERSION 3.10)
project(fhe_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenFHE
find_package(OpenFHE REQUIRED)

# Find nlohmann_json
find_package(nlohmann_json REQUIRED)

# Add executable
add_executable(app main.cpp)

# Add include directories
target_include_directories(app PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(app PRIVATE
    OpenFHE::openfhe
    nlohmann_json::nlohmann_json
) 