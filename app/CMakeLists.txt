cmake_minimum_required(VERSION 3.13)
project(house_price_prediction_fhe)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenFHE package
find_package(OpenFHE REQUIRED)

# Add include directories
include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link directories
link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

# Add cxxopts
include(FetchContent)
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.1.1
)
FetchContent_MakeAvailable(cxxopts)

# Add nlohmann-json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Add source files
set(SOURCES
    main.cpp
    model.cpp
)

# Add the executable
add_executable(app ${SOURCES})

# Link against OpenFHE libraries and dependencies
target_link_libraries(app
    PRIVATE
    OpenFHE::OPENFHEcore
    OpenFHE::OPENFHEpke
    OpenFHE::OPENFHEbinfhe
    OpenFHE::OPENFHEpke_static
    OpenFHE::OPENFHEcore_static
    OpenFHE::OPENFHEbinfhe_static
    cxxopts::cxxopts
    nlohmann_json::nlohmann_json
) 