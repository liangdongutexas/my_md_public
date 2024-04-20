#include "gtest/gtest.h"
#include "../tensor-unified.cuh"

template<typename T>
__global__ void incrementKernel(TensorUnified<T, 1>* tensor, size_t size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        (*tensor)[idx]++;
    }
}

class TensorUnifiedTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        // Code here will be called immediately after the constructor (right before each test).
    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right before the destructor).
    }
};

// Test for TensorUnifiedProxy constructor
TEST_F(TensorUnifiedTest, InitializationTest) {
    // Example: Initialize a TensorUnifiedProxy object with dimension 3x4
    TensorUnifiedProxy<int, 2> tensor(3, 4);
    // Assuming there's a method to get the dimensions
    ASSERT_EQ(tensor.dims[0], 3);
    ASSERT_EQ(tensor.dims[1], 4);
}

// Test for resizing the tensor
TEST_F(TensorUnifiedTest, ResizeTest) {
    // Initialize a TensorUnifiedProxy object with dimension 3x4
    TensorUnifiedProxy<int, 2> tensor(3, 4);
    // Resize to 5x6
    tensor.resize(5, 6);
    ASSERT_EQ(tensor.dims[0], 5);
    ASSERT_EQ(tensor.dims[1], 6);
}

// Test for saving and loading tensor data
TEST_F(TensorUnifiedTest, SaveLoadTest) {
    // Initialize a TensorUnifiedProxy object and set some values
    TensorUnifiedProxy<int, 2> tensor(3, 4);
    // Assuming an operator to set values, example:
    tensor[1][2] = 42;
    
    // Save to a file
    tensor.save("temp.dat");

    // Load into a new tensor and verify
    TensorUnifiedProxy<int, 2> newTensor;
    newTensor.load("temp.dat");

    ASSERT_EQ(newTensor[1][2], 42);
}

// Test for transverseData
TEST_F(TensorUnifiedTest, TransverseDataTest) {
    // Initialize a TensorUnifiedProxy object and set some values
    TensorUnifiedProxy<int, 2> tensor(3, 4);
    // Assuming an operator to set values, example:
    tensor[1][2] = 42;
    
    // Create a modifier to modify data (e.g., increment each value by 1)
    class IncrementModifier : public Modifier<int, int> {
        void action(int &val) override {
            ++val;
        }
    } modifier;
    
    tensor.transverseData(modifier);
    
    // Check if the value at (1, 2) is incremented
    ASSERT_EQ(tensor[1][2], 43);
}

TEST_F(TensorUnifiedTest, CudaKernelTest) {
	const size_t size = 100; // adjust as necessary
    const size_t threadsPer = 256; // adjust as necessary
    const size_t numB = (size + threadsPer - 1) / threadsPer; // grid size to cover the whole data

    // Create a TensorUnifiedProxy object
    TensorUnifiedProxy<float, 1> tensorProxy(size);

    // Initialize it with some values, e.g., fill with zeros.
    for (size_t i = 0; i < size; ++i) {
        tensorProxy[i] = 0.0f;
    }

    // Launch the kernel to increment all values by one.
    incrementKernel<<<numB, threadsPer>>>(tensorProxy.tensor_unified_ptr, size);

    // Make sure to synchronize before trying to access the results on the host
    cudaDeviceSynchronize();

    // Check the results
    for (size_t i = 0; i < size; ++i) {
        // All values should now be 1.0
        ASSERT_EQ(tensorProxy[i], 1.0f);
    }
};

TEST_F(TensorUnifiedTest, CheckDefaultIni) {
	struct MyType {
    int data;
    MyType (): data(13){};
	};

	TensorUnifiedProxy<MyType, 3> my_tensor(2,2,2);
    
    printf("check if the default constructor is called\n");
    // Print the values from the tensor
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                ASSERT_EQ(my_tensor[i][j][k].data, 13);
            }
        }
    }
};


TEST_F(TensorUnifiedTest, CheckArraySelection) {
	struct MyType {
    int data;
    MyType (): data(13){};
	};

	TensorUnifiedProxy<MyType, 3> my_tensor(2,2,2);
    
    printf("check if the selection by indexes denoted by array works\n");
    size_t index[3]={1,1,1};
    my_tensor(index).data=20;
    ASSERT_EQ(my_tensor(index).data, 20);
};

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

