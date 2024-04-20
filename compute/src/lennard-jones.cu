#ifndef LENNARD_JONES_CUH
#define LENNARD_JONES_CUH

#include "interaction.cuh"
#include "../modules/lattice/data-type-lattice.cuh"
#include "utils.cu"

class LennardJones: public Interaction<Atom<2>, Atom<2>, LennardJones, 2> {
private:
    //parameters of the model
    const double epsilon;
    const double sigma;
public:
    //other parameters
    const double step;
    const double length_scale;

    LennardJones(): length_scale(0.5), epsilon(1.0), sigma(1.0), step(1.0){};

    /*device member function to be called by cuda kernel. displacement is the two dimensional relative displacement between two atoms*/
    __device__ __host__ inline double myGetLengthScale(){return length_scale;};

    __device__ __host__  double myCalPoten(double displacement[2], double metric[2][2]){
        double r=device_norm<double, 2>(displacement, metric);
        if (r==0){
            return 100000;
        }
        else{
            return 4 * epsilon * (pow(sigma / r, 12) - pow(sigma / r, 6));
        }
    };          

    __device__ __host__  void myCalGrad(double displacement[2], double metric[2][2], double gradient[2]){
        double r=device_norm<double, 2>(displacement, metric);
        double coeff = 24 * epsilon * (-2 * pow(sigma / r, 12) + pow(sigma / r, 6)) ;

        for (int i=0;i<2;i++){
            for (int j=0;j<2;j++)
            gradient[i]+=coeff * (metric[i][j]*displacement[j])/ pow(r,2);
        }
    };
    
    __device__ __host__  void myCalHessian(double displacement[2], double metric[2][2], double gradient[2]);

    //ignore 1/r^8 in gradient and hessian since they will cancel each other
    __device__ __host__  void myCalPseudoGrad(double displacement[2], double metric[2][2], double gradient[2], double metric_gradient[2][2]){
        double r=device_norm<double, 2>(displacement, metric);
        double coeff = 24 * epsilon * (-2 * pow(sigma, 12)/pow(r,6) + pow(sigma, 6));

        for (int i=0;i<2;i++){
            for (int j=0;j<2;j++)
            gradient[i]+=coeff * (metric[i][j]*displacement[j]);
        }


        for (int i=0;i<2;i++){
            for (int j=0;j<2;j++){
                metric_gradient[i][j]+=coeff *displacement[i]*displacement[j];
            }
        }
    };  
    
    //ignore 1/r^8 in gradient and hessian since they will cancel each other
    __device__ __host__  void myCalPseudoHessian(double displacement[2], double metric[2][2], double hessian[2][2], double metric_hessian[2][2][2][2]){
        double r=device_norm<double, 2>(displacement, metric);
        double coeff1 = 24 * epsilon * (-2 * pow(sigma, 12)*pow(r,-6) + pow(sigma, 6));
        double coeff2 = 96 * (epsilon / pow(r, 2)) * (7 * pow(sigma, 12)*pow(r,-6) -2 * pow(sigma, 6));
        
        //covariant displacement vector with indices lowered by metric
        double l_displacement[2]={};
        for (int i=0;i<2;i++){
            for (int j=0;j<2;j++){
                l_displacement[i]+=metric[i][j]*displacement[j];
            }
        }

        for (int i=0;i<2;i++){
            for (int j=0;j<2;j++){
                hessian[i][j]+=coeff1*metric[i][j] + coeff2 * l_displacement[i]*l_displacement[j];
            }
        }

        for (int i=0;i<2;i++){
            for (int j=0;j<2;j++){
                for (int m=0;m<2;m++){
                    for (int n=0;n<2;n++){
                        metric_hessian[i][j][m][n]+=coeff2 *displacement[i]*displacement[j]*displacement[m]*displacement[n];
                    }
                }
            }
        } 
    };  
};

#endif