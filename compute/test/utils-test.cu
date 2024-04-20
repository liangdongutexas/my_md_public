#include <iostream>
#include <cstddef>
#include "../src/utils.cu"

// ... Your JacobiMethod class definition goes here ...

int main() {
    const size_t n = 3;
    double A[n][n] = {
        {2.0, -3.0, 0.0},
        {-3.0, 2.0, -1.0},
        {0.0, -1.0, 2.0}
    };

    double EVal[n][n];
    double EVect[n][n];

    JacobiMethod<n>::solve(A, EVal, EVect);

    // Print Eigenvalues
    std::cout << "Eigenvalues:\n";
    for (size_t i = 0; i < n; i++) {
        std::cout << EVal[i][i] << " ";
    }
    std::cout << "\n";

    // Print Eigenvectors
    std::cout << "Eigenvectors:\n";
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            std::cout << EVect[i][j] << " ";
        }
        std::cout << "\n";
    }

    double metric[3][3]={};
    for (int i=0; i<3; i++){
        metric[i][i]=1;
    }

    for (int i=0; i<3; i++){
        double norm=device_norm<double, 3>(EVect[i], metric);
        printf("the norm for the %dth row is %f \n", i, norm);
    }

    
    for (int j=0; j<3; j++){
        double diff=0;
        for (int i=0; i<3; i++){
            diff+=A[j][i]*EVect[i][0];
        }
        std::cout<<diff-EVal[0][0]*EVect[j][0]<<std::endl;
    }

    return 0;
}
