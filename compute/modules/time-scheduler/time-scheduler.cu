#include "../lattice/lattice.cuh"
#include "../modules/space-grid/space-grid.cuh"
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

template<typename Interaction, size_t Dim>
class TimeScheduler {
private:

    Lattice<D>& lattice();
    SpaceGrid<Interaction, Dim> space_grid();
    
public:
    //implete proper initialization method for lattice instance: random, periodic, or read from file
    initAtomPosition(){
    };

    //register atoms of the lattice in the space_grids otherwise space_grid is unawared of the atoms in lattice
    registerAtoms(){
        if (lattice.atoms.initialized){
            size_t num_atoms=1;
            for (int i=0; i<Dim; i++){
                num_atoms*=lattice.atoms.dims[i];
            }

            int threadsPerBlock = 256;
            int blocksPerGrid= (num_atoms+threadsPerBlock-1)/threadsPerBlock; 
            registerAtomsKernel<Interaction, Dim><<<blocksPerGrid, threadsPerBlock>>>(lattice.lattice_device, space_grid.space_grid_device);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                // Handle the error
                printf("CUDA error: %s\n", cudaGetErrorString(err));
            }
        }
        else {
            throw std::runtime_error("atoms' positions are not initialized thus cannot be registered");
        }
    };



    void minimizeConfig (size_t iteration){
        // Create the directory for the relaxation process images
        std::experimental::filesystem::create_directories("relaxation_process");
        // Set up the video writer
        int scale   = 50;
        int width   = scale*(d+2);
        int height  = scale*(M+2);
        cv::VideoWriter video("video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 20, cv::Size(height, width));


        while (iteration>0){
            space_grid.calGradientHessian();
            space_grid.equationsOfMotion(minimization=true);
            space_grid.minimizeMetric();
            registerAtoms();

            //draw image to video
            lattice.drawConfig(iteration);
            // Read the image and write it to the video
            cv::Mat img = cv::imread("relaxation_process/plot" + std::to_string(i) + ".png");
            video.write(img);
            // Delete the image
            std::experimental::filesystem::remove("relaxation_process/plot" + std::to_string(i) + ".png");


            iteration--;
        }
    };

    //equations of motion of the atoms
    void EOM(size_t duration){
        // Create the directory for the relaxation process images
        std::experimental::filesystem::create_directories("relaxation_process");
        // Set up the video writer
        int scale   = 50;
        int width   = scale*(d+2);
        int height  = scale*(M+2);
        cv::VideoWriter video("video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 20, cv::Size(height, width));


        while (duration>0){
            space_grid.calGradientHessian();
            space_grid.equationsOfMotion();
            registerAtoms();

            //draw image to video
            lattice.drawConfig(iteration);
            // Read the image and write it to the video
            cv::Mat img = cv::imread("relaxation_process/plot" + std::to_string(i) + ".png");
            video.write(img);
            // Delete the image
            std::experimental::filesystem::remove("relaxation_process/plot" + std::to_string(i) + ".png");

            duration--;
        }
    }
};








//cuda kernel called by the director class
template <class Interaction, size_t Dim>
__global__ void registerAtomsKernel(Lattice<D>:: LatticeDevice* ld ,SpaceGrid<Interaction, Dim>:: SpaceGridDevice* sgd){
    size_t thread_id=threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id<(ld->atoms_device->curr_dim)){
        Atom& atom=*(ld->atoms_device)[thread_id];

        //find the index of grid given atom position
        int grid_index[D];
        sgd->findGridIndex(&atom.posi, grid_index);

        //write the address of atom to the atom_ptrs data of the grid
        AtomsPerGrid& atoms_per_grid=*(sgd->atoms_in_grids_device)[grid_index];
        atomicAdd(atoms_per_grid.N, 1);
        atoms_per_grid.atom_ptrs[atoms_per_grid.N-1]=&atom;
    }
};