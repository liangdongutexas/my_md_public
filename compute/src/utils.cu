#ifndef UTILS_CUH
#define UTILS_CUH

#include <cmath>
#include <iostream>

//translate the indices of flattened device data and the indices of corresponding Eigen::Tensor
template<size_t Dim>
__device__ __host__ size_t flattenIndex(size_t x[Dim], size_t dims[Dim]){
    size_t rawindex=0;
	size_t dim_curr=1;

	for (int i=0; i<Dim; i++){
		rawindex+=x[i]*dim_curr;
		dim_curr*=dims[i];
	}

	return rawindex;
};


template<size_t Dim>
__device__ __host__ void tensorizeIndex(size_t  rawIndex, size_t  x[Dim], size_t  dim[Dim]){
    for (int i = 0; i < Dim; ++i) {
    x[i] = rawIndex % dim[i];
    rawIndex /= dim[i];
	}
};




template<typename T, int d>
__device__ __host__ T device_norm(const T a[d], const T metric[d][d]){
	T result=0; 
	for (int i=0; i<d; i++){
		for (int j=0; j<d; j++){
			result+=a[i]*a[j]*metric[i][j];
		}
		
	}
    return sqrt(result);
};



template<typename T, int d>
__device__ __host__ T device_cosine(T a[d], T b[d]){
	T result;
	for (int i=0; i<d; i++){
		result+=a[i]*b[i];
	}
    return (result)/(device_norm<T,d>(a)*device_norm<T,d>(b));
}

template<size_t n>
class JacobiMethod {
public:
    // Function to perform the Jacobi Method
    __host__ __device__ static void solve(const double A[n][n], double EVal[n][n], double EVect[n][n]) {
        //initialize eigen vectors
        for (int i=0; i<n; i++){
            for (int j=0; j<n; j++){
                EVect[i][j]=(i == j) ? 1 : 0;
            }
        }
        //initialize eigen values
        for (int i=0; i<n; i++){
            for (int j=0; j<n; j++){
                EVal[i][j]=A[i][j];
            }
        }


        //JacobiMethod iteration
        for (int iter = 0; iter < 100; ++iter) {
            int p, q;
            findLargestOffDiag(p, q, EVal);
            if (fabs(EVal[p][q]) < 1e-8) {
                break;
                printf("number of iteration is %d", iter);
            }
            rotate(p, q, EVal, EVect);
        }
    }

private:
    // Function to find the largest off-diagonal element
    __host__ __device__ static void findLargestOffDiag(int &p, int &q, double A[n][n]) {
        double max = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (std::abs(A[i][j]) > max) {
                    max = std::abs(A[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
    }

    // Function to perform the rotation
    __host__ __device__ static void rotate(int p, int q, double A[n][n], double V[n][n]) {
        double a_pp = A[p][p], a_qq = A[q][q], a_pq = A[p][q];
        double theta = 0.5 * std::atan2(2.0 * a_pq, a_qq - a_pp);
        double c = std::cos(theta);
        double s = std::sin(theta);

        // Update A matrix
        A[p][p] = c * c * a_pp - 2.0 * s * c * a_pq + s * s * a_qq;
        A[q][q] = s * s * a_pp + 2.0 * s * c * a_pq + c * c * a_qq;
        A[p][q] = 0.0;
        A[q][p] = 0.0;
        for (int i = 0; i < n; ++i) {
            if (i != p && i != q) {
                double a_ip = A[i][p], a_iq = A[i][q];
                A[i][p] = c * a_ip - s * a_iq;
                A[i][q] = s * a_ip + c * a_iq;
                A[p][i] = A[i][p];
                A[q][i] = A[i][q];
            }
        }

        // Update eigenvectors (V matrix)
        for (int i = 0; i < n; ++i) {
            double v_ip = V[i][p], v_iq = V[i][q];
            V[i][p] = c * v_ip - s * v_iq;
            V[i][q] = s * v_ip + c * v_iq;
        }
    }
};



/*
template<typename T, int d>
__device__  void flatten_data(T* data_flat, T (*data_tensor)[d], int M){
	for (int i=0; i<M; i++){
		for (int j=0; j<d; j++){
			int x[2]={i,j};
			int dim[2]={M,N};
			int rawindex=flatten_index(x, dim, 2);

			data_flat[rawindex]=data_tensor[i][j];
		}
	}
};



template<typename T, int d>
__device__  void tensorize_data(T* data_flat, T (*data_tensor)[d], int M){
	for (int i=0; i<M; i++){
		for (int j=0; j<d; j++){
			int x[2]={i,j};
			int dim[2]={M,N};
			int rawindex=flatten_index(x, dim, 2);

			data_tensor[i][j]=data_flat[rawindex];
		}
	}
};
*/


/*
__device__ void pinvHessian(double hessian[4],double p_inv_hess[4]){
    double a=hessian[0];
    double b=hessian[1];
    double d=hessian[3];

	double lambda1 = (a + d + sqrt(pow(a - d, 2) + 4*pow(b,2))) / 2;
    double lambda2 = (a + d - sqrt(pow(a - d, 2) + 4*pow(b,2))) / 2;

	

	double n1_sqr=pow(b,2)+pow(lambda1-a,2);
    double n2_sqr=pow(b,2)+pow(lambda2-a,2);

	// for diagonal hessian matrix
	if (b==0){
		//set lower bound for a and d
		if (a==0){
			a=1;
		};

		if (d==0){
			d=1;
		}
		p_inv_hess[0]=  1/abs(a);
		p_inv_hess[1]=  0;
		p_inv_hess[2]=  0;
		p_inv_hess[3]=  1/abs(d);
	}

	// for hessian matrix with off diagonal 
	else{
		//set a lower bound for the eigenvalue
		if (lambda1==0){
			lambda1=1;
		};

		if (lambda2==0){
			lambda2=1;
		}

		p_inv_hess[0]=  pow(b,2)/(n1_sqr*lambda1)+pow(b,2)/(n2_sqr*lambda2);
		p_inv_hess[1]=  (b*(lambda1-a))/(n1_sqr*lambda1)+(b*(lambda2-a))/(n2_sqr*lambda2);
		p_inv_hess[2]=  p_inv_hess[1];
		p_inv_hess[3]=  pow(lambda1-a,2)/(n1_sqr*lambda1)+pow(lambda2-a,2)/(n2_sqr*lambda2);
	}
 
};
*/


/*

void draw_energy_change(std::vector<double>& poten_reduc){
	//plot the potential reduction
    // Determine the size of the image
    // Determine the size of the image and scale factor
    double poten_max=0.0;
    double poten_min=0.0;

    for (int i=0; i<poten_reduc.size(); i++){
        poten_max=max(poten_max,poten_reduc[i]);
        poten_min=min(poten_min,poten_reduc[i]);
    } 
    int width  = poten_reduc.size()+20; 
    int height = poten_reduc.size()+20;
    double x_scale = 1;
    double y_scale = static_cast<double>(height)/ (poten_max-poten_min);

    // Create a white image with 3 channels, 8-bit depth
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw the graph
    for (size_t i = 1; i < poten_reduc.size(); ++i) {   
        cv::Point point(static_cast<int>(i*x_scale+10), static_cast<int>((poten_reduc[i]-poten_min)*y_scale+10));

        // Plot black circle
        cv::circle(image, point, 2, cv::Scalar(0, 0, 0), -1);  
    }
    // Save the image
    cv::imwrite("relaxation_process/potential_reduction.png", image);
};


__device__ __host__ void LocalLattice::find_max(double metric[2][2]){

	    //pointer to the adjacent_atom with the largest norm
    AdjacentAtom* max=nullptr;
    double max_norm=DBL_MAX;

	
	max=&up;
	max_norm=device_norm<double,2>(up.relative_displacement, metric);

	for (AdjacentAtom i: {up, down, left, right}){
		if (device_norm<double,2>(i.relative_displacement, metric)>max_norm){
			max=&i;
			max_norm=device_norm<double,2>(i.relative_displacement, metric);
		};
	};
};
*/

#endif