#ifndef LATTICE_KERNELS_CUH
#define LATTICE_KERNELS_CUH

#include "lattice.cuh"
#include "../../src/utils.cu"
#include <eigen3/Eigen/Dense>

template<typename Interaction, size_t Dim>
__global__ void equationsOfMotionKernel(LatticeDevice<Interaction, Dim>* lattice_device){
    //get the tensor dimensions of many atoms
    lattice_device->atoms_device->getDims();
    size_t dims[D];
    for (int i=0; i<D; i++){
        dims[i]=lattice_device->atoms_device.dims[i];
    }


    //get tensor indices of the cuda thread
    size_t index_thread_flat = blockIdx.x * blockDim.x + threadIdx.x;
    size_t index_thread_tensor[D];
    tensorizeIndex(index_thread_flat, index_thread_tensor, dims);

    //check tensor index are within range of dims
    for (int i=0; i<D; i++){
        assert(index_thread_tensor[i]<dims[i]);
    }

    //get the minimization steps suggested by the interaction
    double step=lattice_device->interaction_device[0].step;


    double p_inv_hess[D][D];	
    p_inv_hessian((*lattice_device)(index_thread_tensor).hessian, p_inv_hess);

    /*lattice_config[2*i]-=0.01*lattice_grad[2*i];
    lattice_config[2*i+1]-=0.01*lattice_grad[2*i+1];*/

    //update the position of atoms by the gradient multiply inverse hessian

    double shift[2];
    shift[0]=p_inv_hess[0]*lattice_grad[2*i]+p_inv_hess[1]*lattice_grad[2*i+1];
    shift[1]=p_inv_hess[1]*lattice_grad[2*i]+p_inv_hess[3]*lattice_grad[2*i+1];

    //normalize the shift according to the length scale set by sigma
    double shift_mag=sqrt(pow(shift[0],2)+pow(shift[1],2));

    if (shift_mag>0.2*step){
        shift[0]=shift[0]/shift_mag*0.2*step;
        shift[1]=shift[1]/shift_mag*0.2*step;
    }
    
    if (uni_data.device_atoms_property[index].fixed_mini==OFF){
        uni_data.lattice_config[2*i]-=0.5*shift[0];
        uni_data.lattice_config[2*i+1]-=0.5*shift[1];
    }
}


//find position of dislocation by drawing loops
__global__ void find_dislo(int* device_atom_label, double* near_disp, int* near_label, int L_x, int L_y, int l_x, int l_y, ){

    int i_gpu = blockIdx.x * blockDim.x + threadIdx.x;
    //if the starting atom is dead

    // translate the flattened index to its coodinate in the twod lattice
    int i_x=i_gpu%L_x;
    int i_y=i_gpu/L_x;


    if (i_gpu<L_x*L_y){
        //if the starting atom is dead
        if (device_atom_label[i_gpu*3+2]==10){
            return;
        }

        //if it has already been identified
        if (device_atom_label[i_gpu*3+2]==1){
            return;
        }
    }
    

    //how the loop are formed: 0: move along a_1 direction, 1: -a_1; 2: a_2; 3: -a_2
    int path[]={0,2,2,1,1,3,3,0};
    //displacement accumulated along the path
    double dis_accumu[2]={0,0};


    //indices of the current atoms in the loop
    int curr_label_x=i_gpu%L_x;
    int curr_label_y=i_gpu/L_x;

    for (int i=0;i<8;i++){
        if (i_gpu<L_x*L_y){   
            dis_accumu[0]+=near_disp[8*i_gpu+2*path[i]];
            dis_accumu[1]+=near_disp[8*i_gpu+2*path[i]+1];

            curr_label_x=near_label[8*i_gpu+2*path[i]];
            curr_label_y=near_label[8*i_gpu+2*path[i]+1];

            i_gpu=curr_label_x+L_x*curr_label_y;
        }
        else{
            return;
        }    
    }


    if (dis_accumu[0]!=0 || dis_accumu[1]!=0){
        i_gpu = blockIdx.x * blockDim.x + threadIdx.x;
        device_atom_label[i_gpu*3+2]=1;       
    }   
}


void twod_lattice::set_label:: find_dislocation(){
    // get dimension of the twod lattice
    int M = owner->lattice_config.dimension(0);
    int N = owner->lattice_config.dimension(1);
    int d = owner->lattice_config.dimension(2);

    // set geometry of cuda thread
    int size=M*N;
    int blockSize = 512;
    int numBlocks = (size + blockSize - 1) / blockSize;
    

    double* device_lattice_config;                               //store updated lattice configuration
    int*    device_atom_label;
    double* near_disp; 
    int* near_label;
    double* device_a_1;
    double* device_a_2;

    cudaMallocManaged(&device_lattice_config, d*M*N*sizeof(double));
    cudaMallocManaged(&device_atom_label, 3*M*N*sizeof(int));
    cudaMallocManaged(&device_lattice_config, 8*M*N*sizeof(double));
    cudaMallocManaged(&device_atom_label, 8*M*N*sizeof(int));
    cudaMallocManaged(&device_a_1, d*sizeof(double));
    cudaMallocManaged(&device_a_2, d*sizeof(double));

    //copy lattice configuration to unified memory
    for (int x = 0; x < M; ++x) {
        for (int y = 0; y < N; ++y) {
            for (int z = 0; z < d; ++z) {
                device_lattice_config[x * N * d + y * d + z] = owner->lattice_config(x, y, z);
            }
        }
    }


    //copy atom label to unified memory
    for (int x = 0; x < M; ++x) {
        for (int y = 0; y < N; ++y) {
            for (int z = 0; z < 3; ++z){
                device_atom_label[x * N *3 + y*3 +z] = owner->atom_label(x,y,z);
            }
        }
    }

    owner->update_a();

    //copy lattice constant to unified memory
    for (int i = 0; i < d; ++i) {
        device_a_1[i]=owner->a_1[i];
        device_a_2[i]=owner->a_2[i];
    }


    int l_x=4;
    int l_y=4;

    relative_relation<<<numBlocks, blockSize>>>(device_lattice_config, device_atom_label, near_disp, near_label, 
								  N, M, l_x, l_y, device_a_1, device_a_2);

    find_dislo<<<numBlocks, blockSize>>>(device_atom_label, near_disp, near_label, N, M);
};




void twod_lattice::set_label:: fix_dislo(int* device_atom_label){
    double* device_near_disp;
    int* device_near_label;

    cudaMallocManaged(&device_near_disp, M*N*2*d*d*sizeof(double));
    cudaMallocManaged(&device_near_label, M*N*2*d*d*sizeof(int));

    nearest_atoms<<<numBlocks, blockSize>>>(d_data.device_lattice_config, d_data.device_atom_label, device_near_disp, device_near_label, 
                                            N, M, l_x, l_y, d_data.device_lattice_vectors, d_data.device_period);
    cudaDeviceSynchronize();

    label_dislocation(device_lattice_config, device_atom_label, N, M, l_x, l_y, x_peri_length, y_peri_length);
};



void twod_lattice::set_label:: relax_dislo(int* device_atom_label){

};

#endif