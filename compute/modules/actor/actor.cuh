#pragma once
#include "defines.cuh"

/**
 * @brief 
 * 
 * @tparam ChargeDim number of parameters of the actor that are conserved during dynamics 
 * @tparam DynamicDim dimension of the internal dynamic degrees of freedom 
 * @tparam internalDOFDot time derivative of the internal DOF
 * @tparam Dim dimensionality of the space
 */

template<size_t ChargeDim, size_t DynamicDim, size_t Dim>
class Actor {
public:
    PrecisionType charge[ChargeDim];

    PrecisionType internalDOF[DynamicDim];
    PrecisionType internalDOFDot[DynamicDim];


    PrecisionType position[Dim];
    PrecisionType velocity[Dim];

private:
    PrecisionType internalDOFDoubleDot[DynamicDim];
    PrecisionType acceleration[Dim];


};