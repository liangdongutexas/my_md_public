#ifndef TENSOR_UNIFIED_CU
#define TENSOR_UNIFIED_CU

#include "tensor-unified.cuh"
#include <cuda_runtime_api.h>
#include <fstream>
#include <new>

template<typename T, size_t Dim>
template<typename... Args>
TensorUnifiedProxy<T, Dim>:: TensorUnifiedProxy(Args... ds) : dims{static_cast<size_t>(ds)...} {
        // Ensure the number of arguments matches the tensor rank
        if (sizeof...(ds)!=Dim){
            throw std::invalid_argument("The number of integers does not match the rank of the tensor");
        }
        
        cudaMallocManaged(&tensor_unified_ptr, sizeof(TensorUnified<T,Dim>));
        // Allocate memory for the tensor
        allocateUnified(*tensor_unified_ptr); 
        initialized=true;
        tensor_unified_ptr->getDims();
    };


template<typename T, size_t Dim>
void TensorUnifiedProxy<T, Dim>:: resize(const std::array<size_t, Dim>& ds){
    if (initialized){
        freeUnified<Dim>(*tensor_unified_ptr);
    }
    for (int i=0; i<Dim; i++){
        dims[i] = ds[i];
    }

    cudaMallocManaged(&tensor_unified_ptr, sizeof(TensorUnified<T,Dim>));
    // Allocate memory for the tensor
    allocateUnified(*tensor_unified_ptr); 
    initialized=true;
    tensor_unified_ptr->getDims();
};


template<typename T, size_t Dim>
template<typename... Args>
void TensorUnifiedProxy<T, Dim>:: resize(Args... ds){
    if (sizeof...(ds)!=Dim){
        throw std::invalid_argument("The number of integers does not match the rank of the tensor");
    }
    if (initialized){
        freeUnified<Dim>(*tensor_unified_ptr);
    }
    dims = {static_cast<size_t>(ds)...};
    cudaMallocManaged(&tensor_unified_ptr, sizeof(TensorUnified<T,Dim>));
    // Allocate memory for the tensor
    allocateUnified(*tensor_unified_ptr); 
    initialized=true;
    tensor_unified_ptr->getDims();
};




template<typename T, size_t Dim>
template<size_t d>
void TensorUnifiedProxy<T, Dim>::allocateUnifiedImpl(TensorUnified<T, d>& tensor_unified, std::integral_constant<size_t, d>) {
    cudaError_t err = cudaMallocManaged(&tensor_unified.data, sizeof(TensorUnified<T, d-1>) * dims[Dim-d]);
    tensor_unified.curr_dim=dims[Dim-d];
    tensor_unified.initialized=true;
    if (err != cudaSuccess) {
        // Handle the error
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    for (int i = 0; i < dims[Dim-d]; i++) {
        allocateUnified<d-1>(tensor_unified.data[i]);
    }
}

template<typename T, size_t Dim>
void TensorUnifiedProxy<T, Dim>::allocateUnifiedImpl(TensorUnified<T, 1>& tensor_unified, std::integral_constant<size_t, 1>) {
    cudaError_t err = cudaMallocManaged(&tensor_unified.data, sizeof(T) * dims[Dim-1]);
    tensor_unified.curr_dim=dims[Dim-1];
    tensor_unified.initialized=true;

    //calls the constructor for all the objects allocated in the last indices
    for (int i=0; i<dims[Dim-1]; i++){
        new(tensor_unified.data+i) T();
    }

    if (err != cudaSuccess) {
        // Handle the error
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}


template<typename T, size_t Dim>
template<size_t d>
void TensorUnifiedProxy<T, Dim>::freeUnifiedImpl(TensorUnified<T, d>& tensor_unified, std::integral_constant<size_t, d>){
    for (int i = 0; i < dims[Dim-d]; i++) {
            freeUnified<d-1>(tensor_unified.data[i]);
        }
    cudaFree(tensor_unified.data);
}



template<typename T, size_t Dim>
void TensorUnifiedProxy<T, Dim>::freeUnifiedImpl(TensorUnified<T, 1>& tensor_unified, std::integral_constant<size_t, 1>){
    //calls the destructor for all the objects allocated in the last indices
    for (int i=0; i<dims[Dim-1]; i++){
        (tensor_unified.data+i)->~T();
    }
    cudaFree(tensor_unified.data);
};



template<typename T, size_t Dim>
template<size_t d>
void TensorUnifiedProxy<T, Dim>::saveImpl(std::ofstream& file, TensorUnified<T, d>& tensor_unified, std::integral_constant<size_t, d>){
    for (int i=0; i<dims[Dim-d]; i++){
        saveRecur<d-1>(file, tensor_unified[i]);
    }
};


template<typename T, size_t Dim>
void TensorUnifiedProxy<T, Dim>:: saveImpl(std::ofstream& file, TensorUnified<T, 1>& tensor_unified, std::integral_constant<size_t, 1>){
    for (int i=0; i<dims[Dim-1]; i++){
        //depends on how the operator << is overloaded in the type T, it gives an error is not defined
        file<<tensor_unified[i]<<" ";
    }
};



template<typename T, size_t Dim>
template<size_t d>
void TensorUnifiedProxy<T, Dim>:: loadImpl(std::ifstream& file, TensorUnified<T, d>& tensor_unified, std::integral_constant<size_t, d>){
    for (int i=0; i<dims[Dim-d]; i++){
        loadRecur<d-1>(file, tensor_unified[i]);
    }
};

template<typename T, size_t Dim>
void TensorUnifiedProxy<T, Dim>:: loadImpl(std::ifstream& file, TensorUnified<T, 1>& tensor_unified, std::integral_constant<size_t, 1>){
    for (int i=0; i<dims[Dim-1]; i++){
        //depends on how the operator << is overloaded in the type T, it gives an error is not defined
        file>>tensor_unified[i];
    }
};

template<typename T, size_t Dim>
template<typename To, size_t d>
void TensorUnifiedProxy<T, Dim>:: transverseImpl(Modifier<To, T>& m, TensorUnified<T, d>& tensor_unified, std::integral_constant<size_t, d>){
    for (int i=0; i<dims[Dim-d]; i++){
        transverseDataRecur<To, d-1>(m, tensor_unified[i]);
    }
};

template<typename T, size_t Dim>
template<typename To>
void TensorUnifiedProxy<T, Dim>:: transverseImpl(Modifier<To, T>& m, TensorUnified<T, 1>& tensor_unified, std::integral_constant<size_t, 1>){
    for (int i=0; i<dims[Dim-1]; i++){
        //depends on how the operator << is overloaded in the type T, it gives an error is not defined
        m.action(tensor_unified[i]);
    }
};




#endif