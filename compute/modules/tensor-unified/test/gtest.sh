nvcc -ccbin /usr/bin/gcc-9 gtest.cu ../tensor-unified.cu -o test-tensor -lgtest -lgtest_main -lstdc++ -lm
./test-tensor
rm test-tensor temp.dat