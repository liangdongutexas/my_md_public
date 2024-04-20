#ifndef TENSOR_UNIFIED_CUH
#define TENSOR_UNIFIED_CUH
#include <iostream>
#include <fstream>
#include <type_traits>
#include <stdexcept>
#include <cassert>
#include <array>


template<typename Tf, typename Ti>
struct Modifier {
    Tf result;
    virtual void action(Ti&)=0;
};


/* a tensor datatype of unified memory which can be passed into kernel is needed*/
/* the data type T has to be some plain data type that itself does not contain pointers*/
template<typename T, size_t Dim>
class TensorUnified {
public:
    bool initialized=false;
    int curr_dim = -1;
    size_t dims[Dim];
    TensorUnified<T, Dim-1>* data;

    __device__ __host__ void getDims(){
        assert(initialized);

        dims[0]=curr_dim;
        for (int i=0; i<curr_dim; i++){
            (*this)[i].getDims();
        }

        for (int i=1; i<Dim; i++){
            dims[i]=(*this)[0].dims[i-1];
        }
    };

    __device__ __host__ T& operator()(size_t index[Dim]){
        assert(initialized);
        if (index[0]<curr_dim){
            return data[index[0]](index+1);
        }
    };


    __device__ __host__ TensorUnified<T, Dim-1>& operator[](size_t i) {
    #ifndef __CUDA_ARCH__
        if(i >= curr_dim){
            throw std::out_of_range("Index is out of range.");
        }
        else if (!initialized){
            throw std::runtime_error("The tensor has not been initialized thus cannot be dereferenced.");
        }
        else{
            return data[i];
        }
    #else
        // Error handling in device code
        assert(i < curr_dim && "Index is out of range.");
        assert(initialized && "The tensor has not been initialized thus cannot be dereferenced.");
        return data[i];
    #endif
        }
};


template<typename T>
class TensorUnified<T, 1> {
public:
    bool initialized=false;
    int curr_dim = -1;
    size_t dims[1];
    T* data;

    __device__ __host__ void getDims(){
        assert(initialized);
        dims[0]=curr_dim;
    };

    __device__ __host__ T& operator() (size_t index[1]){
        assert(initialized);
        if (index[0]<curr_dim){
            return (*this)[index[0]];
        }
    };

    __device__ __host__ T& operator[] (size_t i) {
    #ifndef __CUDA_ARCH__
        if(i >= curr_dim){
            throw std::out_of_range("Index is out of range.");
        }
        else if (!initialized){
            throw std::runtime_error("The tensor has not been initialized thus cannot be dereferenced.");
        }
        else{
            return data[i];
        }
    #else
        // Error handling in device code
        assert(i < curr_dim && "Index is out of range.");
        assert(initialized && "The tensor has not been initialized thus cannot be dereferenced.");
        return data[i];
    #endif
    }
};



//remember rule of three and rule of five in defining the class since it dynamically allocate unified memories
template<typename T, size_t Dim>
class TensorUnifiedProxy {
public:
    bool initialized=false;
    std::array<size_t, Dim> dims={}; //hold the dimension values in managed memory
    TensorUnified<T, Dim>* tensor_unified_ptr=nullptr;


    //delete default copy constructor and copy assignment operator due to tensor_unified_ptr
    TensorUnifiedProxy(const TensorUnifiedProxy&)=delete;
    TensorUnifiedProxy& operator=(const TensorUnifiedProxy&)=delete;

    TensorUnifiedProxy()=default;

    template<typename... ActorTypes>
    TensorUnifiedProxy(ActorTypes... ds);

    ~TensorUnifiedProxy() {
        // Free the tensor memory
        freeUnified<Dim>(*tensor_unified_ptr);
        cudaFree(tensor_unified_ptr);
    };


    void resize(const std::array<size_t, Dim>& ds);

    template<typename... ActorTypes>
    void resize(ActorTypes... ds);


    __device__ __host__ auto& operator[](size_t i){
        #ifndef __CUDA_ARCH__
            if(i >= dims[0]){
                throw std::out_of_range("Index is out of range.");
                return (*tensor_unified_ptr)[0];
            }
            else if (!initialized){
                throw std::runtime_error("The tensor has not been initialized thus cannot be dereferenced.");
                return (*tensor_unified_ptr)[0];
            }
            else{
                return (*tensor_unified_ptr)[i];
            }
        #else
            // Error handling in device code
            assert(i < dims[0] && "Index is out of range.");
            assert(initialized && "The tensor has not been initialized thus cannot be dereferenced.");
            return (*tensor_unified_ptr)[i];
        #endif
    };

    T& operator()(size_t index[Dim]){
        return (*tensor_unified_ptr)(index);
    }


    void save(const std::string& filepath){
        if (initialized){
            std::ofstream file(filepath);
            for (int i=0; i<Dim; i++){
                file<<dims[i]<<" ";
            }
            saveRecur<Dim>(file, *tensor_unified_ptr);
            file.close();
        }
    };

    void load(const std::string& filepath){
        
        std::array<size_t, Dim> file_dims;
        std::ifstream file(filepath);
        for (int i=0; i<Dim; i++){
            file>>file_dims[i];
        }
        if (initialized){
            //check if the dimension of the allocated memory is the same as the data to be loaded
            bool dim_equal = std::equal(dims.begin(), dims.end(), file_dims.begin());
            if (!dim_equal){
                resize(file_dims);
            }
            loadRecur<Dim>(file, *tensor_unified_ptr);
        }
        else {
            resize(file_dims);
            loadRecur<Dim>(file, *tensor_unified_ptr);
        }
        file.close();
    };


    template<typename To>
    void transverseData(Modifier<To, T>& m){
        transverseDataRecur<To, Dim>(m, *tensor_unified_ptr);
    };



private:
    template<size_t d>
    void allocateUnified(TensorUnified<T, d>& tensor_unified){
    allocateUnifiedImpl(tensor_unified, std::integral_constant<size_t, d>{});
    };

    template<size_t d>
    void freeUnified(TensorUnified<T, d>& tensor_unified){
        if (initialized) {
            freeUnifiedImpl(tensor_unified, std::integral_constant<size_t,d>{});
        }
    };

    template<size_t d>
    void saveRecur(std::ofstream& file, TensorUnified<T, d>& tensor_unified){
        saveImpl(file, tensor_unified, std::integral_constant<size_t, d>{});
    };

    template<size_t d>
    void loadRecur(std::ifstream& file, TensorUnified<T, d>& tensor_unified){
        loadImpl(file, tensor_unified, std::integral_constant<size_t, d>{});
    };

    template <typename To, size_t d>
    void transverseDataRecur(Modifier<To, T>& m, TensorUnified<T, d> tensor_unified){
        transverseImpl(m, tensor_unified, std::integral_constant<size_t, d>{});
    };

    void allocateUnifiedImpl(TensorUnified<T, 1>& tensor_unified, std::integral_constant<size_t, 1>);

    void freeUnifiedImpl(TensorUnified<T, 1>& tensor_unified, std::integral_constant<size_t, 1>);

    void saveImpl(std::ofstream& file, TensorUnified<T, 1>& tensor_unified, std::integral_constant<size_t, 1>);

    void loadImpl(std::ifstream& file, TensorUnified<T, 1>& tensor_unified, std::integral_constant<size_t, 1>);

    template<typename To>
    void transverseImpl(Modifier<To, T>& m, TensorUnified<T, 1>& tensor_unified, std::integral_constant<size_t, 1>);

    template<size_t d>
    void saveImpl(std::ofstream& file, TensorUnified<T, d>& tensor_unified, std::integral_constant<size_t, d>); 

    template<size_t d>
    void loadImpl(std::ifstream& file, TensorUnified<T, d>& tensor_unified, std::integral_constant<size_t, d>);

    template<size_t d>
    void allocateUnifiedImpl(TensorUnified<T, d>& tensor_unified, std::integral_constant<size_t, d>);

    template<size_t d>
    void freeUnifiedImpl(TensorUnified<T, d>& tensor_unified, std::integral_constant<size_t, d>);

    template<typename To, size_t d>
    void transverseImpl(Modifier<To, T>& m, TensorUnified<T, d>& tensor_unified, std::integral_constant<size_t, d>);
};


#include "tensor-unified.cu"


#endif