#ifndef SPACE_GRID_KERNELS_CU
#define SPACE_GRID_KERNELS_CU

#include "space-grid.cuh"
#include "../lattice/lattice.cuh"
#include "../src/utils.cu"


template<class Interaction, size_t Dim>
__global__ void calGradientHessianKernel(SpaceGrid<Interaction, Dim>:: SpaceGridDevice* sgd){
	//expected flat geometry of cuda kernel since all input data are flattened
	int gpu_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    size_t grid_index[D];
    tensorizeIndex(gpu_index, grid_index, sgd->atoms_in_grids.dims);
    bool isValid=true;
    for (int i=0; i<D; i++){
        isValid=isValid && (grid_index[i]<sgd->atoms_in_grids.dims[i]);
    }

    //lanuch mutiple gpu threads for each atom in the current grid
	if (isValid) {
        calSingleAtom<<<sgd->atoms_in_grids(grid_index).N, 1>>>(sgd, grid_index, hessian);
        cudaDeviceSynchronize();
    };	
};


template<class Interaction, size_t Dim>
__global__ void calSingleAtom(SpaceGrid<Interaction, Dim>:: SpaceGridDevice* sgd, size_t curr_grid_index[D]){
    int gpu_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (gpu_index<(sgd->atoms_in_grids)(curr_grid_index).N){
        //the current atom the thread is working on
        Atom& curr_atom=*(sgd->atoms_in_grids(grid_index).atom_ptrs[gpu_index]);
        if(curr_atom.queryContriPoten==CONTRIBUTING){
            recurCalculate(curr_atom, curr_grid_index, 0, sgd);
        }
    } 



    //calculate the interaction between curr_atom and all atoms in adjacent blocks
    void recurCalculate (Atom& curr_atom, const size_t adja_grid_index[D], size_t posi,  SpaceGrid<Interaction, D>:: SpaceGridDevice* sgd){
        bool isValid=true;
        for (int i=0; i<D; i++){
            isValid=isValid && (adja_grid_index[i]<(sgd->atoms_in_grids.dims[i]));
        }
        
        if (isValid && posi==D-1){
            for (i=0; i<sgd->atoms_in_grids(adja_grid_index).N; i++){
                const Atom& adja_atom= *(sgd->atoms_in_grids(adja_grid_index).atom_ptrs[i]);

                //exclude cases where two atoms are the same or the other atom is not contributing
                if (&curr_atom!=&adja_atom && adja_atom.queryContriPoten==CONTRIBUTING){
                    //calculate the interaction from adjacent atoms from all nearest supercells
                    for (int i=0; i<D; i++){
                        displacement[i]=curr_atom.posi[i]-adja_atom.posi[i];
                    };

                    if (sgd->geometry.condition==PERIODIC){

                        double bounds[D];
                        for (int i=0; i<D; i++){
                            bounds[i]=sgd->geometry.boundary_size[i][1]-sgd->geometry_device.boundary_size[i][0];
                        }

                        double displacement[D];
                        for (int i=0; i<D; i++){
                            displacement[i]=curr_atom.posi[i]-adja_atom.posi[i];
                        }

                        iterateSupercell(curr_atom, adja_atom, bounds, displacement, 0);

                        void iterateSupercell(Atom& curr_atom, Atom& adja_atom, const double& bounds[D], const double displacement[D], int posi){
                            if (posi==D-1){
                                for (int i=0; i<D; i++){
                                    double refere_dir[D]={};
                                    refere_dir[i]=1;
                                    double my_cosine=device_cosine<D>(refere_dir, displacement);
                                    if (my_cosine>curr_atom.my_cosine[i][0]){
                                        curr_atom.adja_atoms[0]=&adja_atom;
                                    }
                                    else (-my_cosine>curr_atom.my_cosine[i][1]){
                                        curr_atom.adja_atoms[1]=&adja_atom;
                                    }
                                }

                                //this is where the atual calculation happens
                                sgd->interaction.pseudo_grad(displacement_copy, sgd->geometry.metric, curr_atom.grad, curr_atom.metric_grad);

                                sgd->interaction.pseudo_hess(displacement_copy, sgd->geometry.metric, curr_atom.hessian, curr_atom.metric_hessian);

                                curr_atom.poten+= 0.5*sgd->interaction.poten(displacement_copy, sgd->geometry.metric);  //0.5 to avoid double counting
                                
                                //find the largest component of the relative displacement
                                size_t index=0;
                                double value=fabs(displacement_copy[0]);
                                for (int i=1; i<D; i++){
                                    if (fabs(displacement_copy[i])>value){
                                        value=fabs(displacement_copy[i]);
                                        index=i;
                                    }
                                }

                                //find if the adja_atom is the closest one along the index direction
                                double norm=device_norm<double, D>(displacement_copy, sgd->geometry.metric);
                                if (displacement_copy[i]<0){
                                    if (norm<curr_atom.distances[index][0]){
                                        curr_atom.adja_atom_ptrs[index][0]=&adja_atom;
                                        curr_atom.distances[index][0]=norm;
                                    }
                                }
                                else {
                                    if (norm<curr_atom.distances[index][1]){
                                        curr_atom.adja_atom_ptrs[index][1]=&adja_atom;
                                        curr_atom.distances[index][1]=norm;
                                    }
                                }
                            };

                            for (int i: {-1, 0, 1}){
                                //make a copy and shift the position component by supercell size
                                double displacement_copy[D];
                                for (int i=0; i<D; i++){
                                    displacement_copy[i]=displacement[i];
                                }
                                displacement_copy[posi]+=i*bounds[posi];

                                iterateSupercell(curr_atom, bounds, displacement, posi+1);
                            }
                        }
                    }
                    else {
                        //find the largest component of the relative displacement
                        size_t index=0;
                        double value=fabs(displacement_copy[0]);
                        for (int i=1; i<D; i++){
                            if (fabs(displacement_copy[i])>value){
                                value=fabs(displacement_copy[i]);
                                index=i;
                            }
                        }

                        //find if the adja_atom is the closest one along the index direction
                        double norm=device_norm<double, D>(displacement_copy, sgd->geometry.metric);
                        if (displacement_copy[i]<0){
                            if (norm<curr_atom.distances[index][0]){
                                curr_atom.adja_atom_ptrs[index][0]=&adja_atom;
                                curr_atom.distances[index][0]=norm;
                            }
                        }
                        else {
                            if (norm<curr_atom.distances[index][1]){
                                curr_atom.adja_atom_ptrs[index][1]=&adja_atom;
                                curr_atom.distances[index][1]=norm;
                            }
                        }
                        //this is where the atual calculation happens
                        sgd->interaction.pseudo_grad(displacement, sgd->geometry.metric, curr_atom.grad, sgd->geometry.metric_grad);

                        sgd->interaction.pseudo_hess(displacement, sgd->geometry.metric, curr_atom.hessian, sgd->geometry.metric_hessian);

                        curr_atom.poten+= 0.5*sgd->interaction.poten(displacement, sgd->geometry.metric);  //0.5 to avoid double counting
                    }	
                }
            }
        }

        for (int i: {-1, 0, 1}){
            double adja_grid_index_cpy[D];
            for (int i=0; i<D; i++){
                adja_grid_index_cpy[i]=adja_grid_index[i];
            }
            adja_grid_index_copy[posi]+=i;
            recurCalculate(curr_atom, adja_grid_index_copy, posi+1, sgd);
        }   
    } 
};


#endif