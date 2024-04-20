/**
 * @brief  BufferSize the maximal number of actors that allowed to exist in a space block. 
 * Differentce species of actors share this same buffer size for simplilcity. 
 * 
 */
#ifndef CONSTANTS_MACROS_CUH
#define CONSTANTS_MACROS_CUH

// Use the macro to define the type
#ifdef USE_DOUBLE_PRECISION
    using PrecisionType = double;
#else
    using PrecisionType = float;
#endif

#include <cuda_runtime.h>
#include <iostream>


void checkCudaError(cudaError_t result, const char *function, const int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result)
                  << " in " << function << " at line " << line << std::endl;
        // Optionally, exit or throw an exception
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

// Macro to simplify usage of the check function
#define CUDA_CHECK(call) checkCudaError((call), __FUNCTION__, __LINE__)

#endif
