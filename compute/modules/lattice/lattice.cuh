#ifndef LATTICE_CUH
#define LATTICE_CUH

#include "data-type-lattice.cuh"
#include "../tensor-unified/tensor-unified.cuh"
#include "../space-grid/space-grid.cuh"



/*after experiments,  I found out three ways to pass host class to a cuda kernel
1. using template: create an instance of the type specified in the template and pass to cuda kernel
2. create an instance of the class on host and pass it by value to the argument of cuda kernel: the member data should be allocated on GPU ram as unified memory 
                                                                          and the member function should be device\
3. create an instance of the class as unified memory
*/


template <size_t Dim /*dimension of the lattice*/>
class Lattice {
    //member data and function to be passed to a cuda kernel 
    class LatticeDevice { 
    public:
        //Lagrangian description of atoms
        TensorUnified<Atom<D>,1>* atoms_device;
        __device__ findDislocation();
    };

public:
    Boundary<D>& geometry;
    //host data and device functions to be called by cuda kernel
    LatticeDevice* lattice_device;

    //Lagrangian description of atoms
    TensorUnifiedProxy<Atom<D>,1> atoms;

    //the center position and spatial bounds of the lattice
    ShapeLocation<D> shape_location;

    //toggle LatticeDevice data pointer to the direction of unified memory data 
    Lattice(){
        cudaMallocManaged(&lattice_device, sizeof(LatticeDevice)); 
        lattice_device->atoms_device=atoms.tensor_unified_ptr; 
        lattice_device->interaction_device=interaction.tensor_unified_ptr;
    };
    ~Lattice(cudaFree(lattice_device));

    //locate the center position and boundary_size of the lattice
    void findShapeLocation();

    void findDislocation(UnifiedData& device_data, int m, int n);

    void drawConfig(int label=0);  //draw the current configuration of lattice   
};

#include "lattice.cu"

#endif