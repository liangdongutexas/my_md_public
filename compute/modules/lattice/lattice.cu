#ifndef TWOD_LATTICE_CU
#define TWOD_LATTICE_CU

#include "lattice.cuh"
#include <cuda_runtime_api.h>
#include <experimental/filesystem>

template<size_t Dim>
void Lattice<D>::findShapeLocation(){
    struct Finder: Modifier<ShapeLocation, Atom>{
        ShapeLocation result;
        void action(Atom& a) override{
            for (int i=0; i<Dim; i++){
                result.center_position[i]+=a.posi[i];
                result.bounds[i][0]=min(result.bounds[i][0], a.posi[j]);
                result.bounds[i][1]=max(result.bounds[i][1], a.posi[j]);
            }
        }
    }

    if (atoms.initialized) {
        size_t num_atoms=1;
        for (int i=0; i<Dim; i++){
            num_atoms*=atoms.dims[i];
        }
        
        Finder f;
        atoms.transverseData<ShapeLocation>(f);
        shape_location=f.result;
        shape_location.center_position/=num_atoms;
    }
};

template<size_t Dim>
void Lattice<D>:: equationsOfMotion(bool minimization=false){
    if (minimization){
        size_t num_atoms=atoms.curr_dim;

        size_t blockSize = 512;
        size_t numBlocks = (num_atoms + blockSize - 1) / blockSize;

        equationsOfMotionKernel<<<blockSize, numBlocks>>>();
    }
    else {
        size_t num_atoms=1;
        for (int i=0; i<D; i++){
            num_atoms*=atoms.dims[i];
        }

        size_t blockSize = 512;
        size_t numBlocks = (num_atoms + blockSize - 1) / blockSize;


        
    }
    else {
        throw std::runtime_error("option not recognized, please choose between 'minimization' and 'EOM'");
    }
};

template<size_t Dim>
void Lattice<D>::find_minimum(int num_inter, int l_x, int l_y, bool metric_vary, bool atom_posi_vary, bool defect_seperate){
    //even/odd integer means boundary_size/defect core is not fixed/fixed during minimization
    int d_fix_counter=1;
    int d = 2;
    int size=2000;


    int blockSize = 512;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    //record the potential deducton during the minimization process
    std::vector<double> poten_reduc;

    // Create the directory for the relaxation process images
    std::experimental::filesystem::create_directories("relaxation_process");
    // Set up the video writer
    int scale   = 50;
    int width   = scale*(d+2);
    int height  = scale*(M+2);
    cv::VideoWriter video("video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 20, cv::Size(height, width));



    //intermediate data for calculation
    double* device_total_poten;                                  //store total potential before each minimization
    cudaMallocManaged(&device_total_poten, sizeof(double));
    double* device_lattice_grad;                                 //gradient of all atoms
    double* device_lattice_hessian;                              //hessian of all atoms
    cudaMallocManaged(&device_lattice_grad, d*init_data.M*init_data.N*sizeof(double));
    cudaMallocManaged(&device_lattice_hessian, d*d*init_data.M*init_data.N*sizeof(double));
    double* device_metric_grad;                                  //gradient of the metric
    double* device_metric_hessian;                               //hessian matrix of the metric
    cudaMallocManaged(&device_metric_grad, d*d*sizeof(double)); 
    cudaMallocManaged(&device_metric_hessian, pow(d,4)*sizeof(double));
    cudaError_t err;


    //label the defect before minimization
    if (defect_seperate){
        find_defect(uni_data);
    };

    for (int i=0;i < num_inter;i++) {
        // set to all gradient and hessian of the metric and total potential to zero before calculation
        cudaMemset(init_data.device_metric_grad, 0, pow(d,2)*sizeof(double));
        
        cudaMemset(init_data.device_metric_hessian, 0, pow(d,4)*sizeof(double));
        
        cudaMemset(init_data.device_total_poten, 0, sizeof(double));
        
        //draw the lattice configuration before each minimization
        draw_config(i);

        // Read the image and write it to the video
        cv::Mat img = cv::imread("relaxation_process/plot" + std::to_string(i) + ".png");
        video.write(img);
        // Delete the image
        std::experimental::filesystem::remove("relaxation_process/plot" + std::to_string(i) + ".png");

        //calculate all gradient and hessian matrix
        grad_hess_kernel<Interaction><<<numBlocks, blockSize>>>(uni_data, interaction, device_total_poten, device_lattice_grad, device_lattice_hessian, 
                                                      device_metric_grad, device_metric_hessian, N, M, l_x, l_y);
        cudaDeviceSynchronize(); 
        
        //update metric
        if (metric_vary){
            metric_mini(device_metric_grad, device_metric_hessian);
        }

        //update atom positions
        if (atom_posi_vary){
            //update atom positions
            if (defect_seperate){
                if ((b_fix_counter/10)%2==0){
                    relax_defect(uni_data);
                }
                else{
                    fix_defect(uni_data);
                }

                d_fix_counter++;
            }

            minimizer_kernel<<<numBlocks, blockSize>>>(uni_data, interaction, device_lattice_grad, device_lattice_hessian);
            cudaDeviceSynchronize();        
        }
        
        //add calculated potential to the end of the std::vector
        poten_reduc.push_back(*device_total_poten/(uni_data.M*uni_data.N));

        //copy information from uni_data to host_data
        transferDeviceToHost();
    }

    //free unified memory
    cudaFree(device_total_poten);
    cudaFree(device_lattice_grad);
    cudaFree(device_lattice_hessian);
    cudaFree(device_metric_grad); 
    cudaFree(device_metric_hessian);
    cudaFree(interaction);
    
    // Close the video writer
    video.release();
    //draw a plot of total energy change
    draw_energy_change(poten_reduc);
    
}

template<size_t Dim>
void Lattice<Interaction, D>::equations_of_motion(){
    // Placeholder for the equations_of_motion method
}





template<size_t Dim>
void Lattice<D>::metric_mini(double* device_metric,double* device_metric_grad, double* device_metric_hessian){
    // get dimension of the twod lattice
    int M = lattice_config.dimension(0);
    int d = lattice_config.dimension(1);
    int d = lattice_config.dimension(2);

    Eigen::MatrixXd metric_hessian = Eigen::MatrixXd::Random(d*d,d*d);
    //translate the data into matrix form
    for (int i=0; i<d*d; i++){
        for (int j=0; j<d*d; j++){
            int index=4*i+j;
            metric_hessian(i,j)=device_metric_hessian[index]/(M*N);
        }
    }

    for (int i=0; i<d*d; i++){
        device_metric_grad[i]=device_metric_grad[i]/(M*N);
    }

    // find the positive inverse hessian matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(metric_hessian);
    Eigen::MatrixXd Q = es.eigenvectors();
    Eigen::MatrixXd D = es.eigenvalues().asDiagonal();

    for (int i = 0; i < D.rows(); i++) {
        D(i, i) = 1 / abs(D(i, i));
    }

    Eigen::MatrixXd metric_inv_hessian = Q * D * Q.inverse();

    for (int i=0;i<d;i++){
        for (int j=0;j<d;j++){
            for (int k=0;k<d*d;k++){
                device_metric[2*i+j]-=metric_inv_hessian(2*i+j,k)*device_metric_grad[k];
            }    
        }
    }
}



template<class Interaction, size_t Dim>
void TwodLattice<Interaction, Dim>::find_dislocation(UnifiedData& uni_data, int m, int n){
    int blockSize = 512;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    LocalLattice* device_local_lattices;                                  //store total potential before each minimization
    cudaMallocManaged(&device_local_lattices, sizeof(LocalLattice));

    find_nearest_atoms<<<numBlocks, blockSize>>>(uni_data, device_local_lattices, m, n);

    
}


template<class Interaction, size_t D>
void TwodLattice<Interaction>:: update_a () {
	 // get dimension of the twod lattice
    int M = host_data.lattice_config.dimension(0);
    int d = host_data.lattice_config.dimension(1);
    int d = host_data.lattice_config.dimension(2);

	host_data.lattice_vectors[]
	
    int size_x=3; int size_y=3;
    int start=5;

    Eigen::Tensor <double, 3> l_lattice(size_x,size_y,2);


	for (int i=0; i<size_x; ++i){
		for (int j=0; j<size_y; ++j){
			for (int k=0; k<d; ++k){
				if (start+j<N-1 && start+i<M-1 && start+i+1<M-1 && start+j+1<N-1){

                    l_lattice(i,j,k)=lattice_config(start+i,start+j,k);

					a_1[k]+=(lattice_config(start+i+1,start+j,k)-lattice_config(start+i,start+j,k))/(size_x*size_y);
					a_2[k]+=(lattice_config(start+i,start+j+1,k)-lattice_config(start+i,start+j,k))/(size_x*size_y);
				}		
			}	
		}
	}
  
    /*printf("local lattice configuration used to calculate lattice vector: \n");
    for (int i = 0; i < size_x; ++i) {
        printf("\n");
        for (int j = 0; j < size_y; ++j){
            for (int k = 0; k < d; ++k) {
                printf("%f, ", l_lattice(i,j,k));
            }
        }
    } */

}



template<class Interaction, size_t D>
void TwodLattice<Interaction, D>::draw_config(int label) {
    int M = lattice_config.dimension(0);
    int N = lattice_config.dimension(1);

    // Determine the size of the image and scale factor
    int scale   = 50;                                   //how many pixes between two lattice points 
    int width   = scale*d;
    int height  = scale*M;
    

    // Create a white image with 3 channels, 8-bit depth
    cv::Mat image(height+scale*2, width+scale*2, CV_8UC3, cv::Scalar(255, 255, 255));

    // need to scale the position of latice points to fit in the image 
    // suppose the atoms roughly follow the initial relative position, 
    // then the four atoms at the corners of the twod lattice sets the scale

    double x_center=(lattice_config(0,0,0)+ lattice_config(0,d-1,0)+lattice_config(M-1,0,0)+lattice_config(M-1,d-1,0))/4;
    double y_center=(lattice_config(0,0,1)+ lattice_config(0,d-1,1)+lattice_config(M-1,0,1)+lattice_config(M-1,d-1,1))/4;

    double x_scale =width/(2*std::max({std::abs(lattice_config(0,0,0)-x_center),std::abs(lattice_config(0,d-1,0)-x_center),\
                    std::abs(lattice_config(M-1,0,0)-x_center),std::abs(lattice_config(M-1,d-1,0)-x_center)}));

    double y_scale =height/(2*std::max({std::abs(lattice_config(0,0,1)-y_center),std::abs(lattice_config(0,d-1,1)-y_center),\
                            std::abs(lattice_config(M-1,0,1)-y_center),std::abs(lattice_config(M-1,d-1,1)-y_center)}));


    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double x = (lattice_config(i, j, 0)-x_center) * x_scale;
            double y = (lattice_config(i, j, 1)-y_center) * y_scale;

            // Shift the origin to the center of the image
            cv::Point point(static_cast<int>(x+(width+2*scale)/2), static_cast<int>(y+(height+2*scale)/2));
    
            // Plot black circle
            cv::circle(image, point, 10, cv::Scalar(0, 0, 0), -1);  
        }
    }

    // Create the directory for the relaxation process images
    std::experimental::filesystem::create_directories("relaxation_process");

    // Save the image
    cv::imwrite("relaxation_process/plot" + std::to_string(label) + ".png", image);
};


#endif