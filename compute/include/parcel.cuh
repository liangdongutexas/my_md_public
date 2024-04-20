
#ifndef PARCEL_CUH
#define PARCEL_CUH

#include "defines.cuh"

//contains property of a single atoms: displacement field Lagrangian description
template<size_t Dim /*dimension of the atom*/>
class Parcel {
public:
    /**
     properties that are approximately conserved or static during the dynamics
    */
    size_t param_size;
    PrecisionType* params; 

    /**
     internal degrees of freedom that is dynamical
     dot_interDOF is the time derivative of the internal DOF
    */
    size_t interDOF_size;
    PrecisionType* interDOF;
    PrecisionType* dot_interDOF;


    //position of the atom in 2d plane
    PrecisionType posi[Dim]={};
    PrecisionType velocity[Dim]={};
    PrecisionType acceleration[Dim]={};
};

#endif