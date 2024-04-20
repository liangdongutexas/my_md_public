#ifndef SPACE_GRID_CU
#define SPACE_GRID_CU


#include "space-grid.cuh"
#include "../src/utils.cu"
#include "space-grid-kernels.cu"

template<template <typename, typename, typename... > class SpaceBlockDerived, typename Params, typename FinnerBlock, typename... ActorTypes>
__device__ void SpaceGrid<SpaceBlockDerived, Params, FinnerBlock, ActorTypes...>:: SpaceGridDevice:: findGridIndex(double position[Dim], int grid_index[Dim]){
	GridGeometry gg=(*geometry_device)[0];
	for (int i=0; i<Dim; i++){
		grid_index[i]= static_cast(int)((position[i]- gg.center_position[i])/(gg.block_size[i])+gg.dims[i]/2-1);
	}
};


template<template <typename, typename, typename... > class SpaceBlockDerived, typename Params, typename FinnerBlock, typename... ActorTypes>
void SpaceGrid<SpaceBlockDerived, Params, FinnerBlock, ActorTypes...>::buildGrid(){
	//determine grid size using the average distance between objects suggested by the interaction
	for (int i=0; i<Dim; i++){
		grid_geometry[0].block_size[i]=interaction[0].length_scale*pow(AtomsPerGrid::Nmax, 1/Dim);
	}

	if (geometry[0].condition==OPENBOUNDARY){
		lattice.locateLattice();
		for (int i=0; i<Dim; i++){
			grid_geometry[0].center_position[i]=lattice.shape_location.center_position[i];
		}
		for (int i=0; i<Dim; i++){
			grid_geometry[0].dims[i]=static_cast<size_t>((lattice.shape_location.bounds[i][1]-lattice.shape_location.bounds[i][0])/block_size[i]+2);
		}
	}
	else if (geometry[0].condition==FIXBOUNDARY || geometry[0].condition==PERIODIC){
		//choose a small square that presumably contains less atom than Nmax
		for (int i=0; i<Dim; i++){
			grid_geometry[0].center_position[i]=(geometry[0].boundary_size[i][0]+geometry[0].boundary_size[i][1])/2
		}
		for (int i=0; i<Dim; i++){
			grid_geometry[0].dims[i]=static_cast<size_t>((geometry[0].boundary_size[i][1]-geometry[0].boundary_size[i][0])/block_size[i]+2);
		}
	}

	//resize the unified tensor, computationally costly.
	atoms_in_grids.resize(grid_geometry[0].dims);
};

template<typename T, class Interaction, size_t Dim>
void SpaceGrid<T, Interaction, D>::buildGrid(const double bounds[Dim][2]){
	for (int i=0; i<Dim; i++){
		grid_geometry[0].block_size[i]=interaction[0].length_scale*pow(AtomsPerGrid::Nmax, 1/Dim);
	}

	if (geometry[0].condition==OPENBOUNDARY){
		lattice.locateLattice();
		for (int i=0; i<Dim; i++){
			grid_geometry[0].center_position[i]=lattice.shape_location.center_position[i];
		}
		for (int i=0; i<Dim; i++){
			grid_geometry[0].dims[i]=static_cast<size_t>((lattice.shape_location.bounds[i][1]-lattice.shape_location.bounds[i][0])/block_size[i]+2);
		}
	}
	else if (geometry[0].condition==FIXBOUNDARY || geometry[0].condition==PERIODIC){
		//choose a small square that presumably contains less atom than Nmax
		for (int i=0; i<Dim; i++){
			grid_geometry[0].center_position[i]=(geometry[0].boundary_size[i][0]+geometry[0].boundary_size[i][1])/2
		}
		for (int i=0; i<Dim; i++){
			grid_geometry[0].dims[i]=static_cast<size_t>((geometry[0].boundary_size[i][1]-geometry[0].boundary_size[i][0])/block_size[i]+2);
		}
	}

	//resize the unified tensor, computationally costly.
	atoms_in_grids.resize(grid_geometry[0].dims);
};


template<typename T, class Interaction, size_t Dim>
void SpaceGrid<T, Interaction, D>::calGradientHessian(bool hessian=false){
	int num_kernel=1;
	for (int i=0; i<Dim; i++){
		num_kernel*=grid_geometry[0].dims[i];
	}

	int threadPerBlock=256;
	int blockPerGrid=(num_kernel+threadPerBlock-1)/threadPerBlock;
	calGradientHessianKernel<Interaction, Dim><<<threadPerBlock, blockPerGrid>>>(space_grid_device, hessian);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		// Handle the error
		printf("CUDA error: %s\n", cudaGetErrorString(err));
	}
};

#endif

