#ifndef SPACE_GRID_CUH
#define SPACE_GRID_CUH

#include "../lattice/lattice.cuh"
#include "../tensor-unified/tensor-unified.cuh"
#include "space-block.cuh"

template<size_t Dim>
class GridGeometry{

enum BoundaryCondition {
    PERIODIC,
    FIXBOUNDARY
};

public:
/**
 * @brief boundary_size the size of the boundary for periodic and fix boundary condition
 * dims the number of SpaceSimplex along each dimension
 * center_position the position of the middle block indexed by the floor of dims/2.
 * 
 */
    constexpr BoundaryCondition condition;
    constexpr double boundary_size[Dim][2];

    size_t dims[Dim];
    constexpr double block_size[Dim];
    double center_position[Dim];

    //for other boundary conditions
    GridGeometry(BoundaryCondition& condi, double bound[Dim][2]): condition(condi), boundary_size(bound){
        constexpr double EPSILON = 1E-9;
    	inline bool isCloseToZero = [EPSILON](double val) { return std::abs(val) < EPSILON; };

        for (int i=0; i<Dim; i++){
            if (isCloseToZero(bounds[i][0]) || isCloseToZero(bounds[i][1])){__device__ void findGridIndex(double position[Dim], int grid_index[Dim]);boundary condition);
                throw std::runtime_error("bounds almost zero is not accepted");
            }
        }

    };
};

/**
 * @brief using self-defined TensorUnifiedProxy to automatize unified memory allocation and deallocation
 * TensorUnified* is the pointer wrapped by TensorUnifiedProxy
 * 
 * @tparam SpaceBlockDerived basic block used to discretize the whole space. it already contains interaction for fields and actors.
 * @tparam Params:: Dim Dimension of the space, e.g. 2, 3, 4 ::Fdim dimension of the fields, :: Buffersize maximal number of actors per type in one spaceblock
 * ::Level higher level means more coasered blocks
 * @tparam Finnerblock low level, smaller space blocks
 * @tparam ActorTypes all actor types that can register in the space block
 */

template<template <typename, typename, typename... > class SpaceBlockDerived, typename Params, typename FinnerBlock, typename... Actortypes>
class SpaceGrid{
    //member data and function to be passed to a cuda kernel
    class SpaceGridDevice{

        GridGeometry<Params::Dim>& grid_geometry; 
        TensorUnified<SpaceBlockDerived<Params, FinnerBlock, Actortypes...>>& blocks;

        //given position of an object, find the index of the corresponding gird,
        //return true if the position is winthin the bounds and false if the position is out of bounds
        __device__ void findGridIndex(double position[Dim], int grid_index[Dim]);
    };

public:
    
    TensorUnifiedProxy<GridGeometry<Params::Dim>, 1> grid_geometry(1); 
    TensorUnifiedProxy<SpaceBlockDerived<Params, FinnerBlock, ActorTypes...>, Params::Dim> blocks;

    //the unified data and device functions called by cuda kernel
    SpaceGridDevice* space_grid_device;

    SpaceGrid(const Boundary& bound) boundary(bound){
        cudaMallocManaged(&space_grid_device, sizeof(SpaceGridDevice));
        //attach space_grid_device data to unified memory allocated by TensorUnifiedProxy
        space_grid_device->grid_geometry=grid_geometry[0];
        space_grid_device->blocks=*(blocks.tensor_unified_ptr);
    };

    ~SpaceGrid(){cudaFree(space_grid_device)};

    //build grid for fixboundary and periodic boundary conditions
    void buildGrid();
    //build grid for open boundary conditions thus the initial size of the boundary is needed
    void buildGrid(const double bounds[Dim][2]);

    void equationsOfMotion(bool minimization=false);

    void findLocalLattice();
};

#include "space-grid.cu"

#endif