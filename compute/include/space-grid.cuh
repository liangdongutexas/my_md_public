#ifndef SPACE_CUH
#define SPACE_CUH

#include <fstream>
//geometry of the spacetime atoms are in
template<size_t Dim>
struct Boundary{
//all considered boundary_size condition
enum BoundaryCondition {
    PERIODIC,
    OPENBOUNDARY,
    FIXBOUNDARY
};

public:
    const BoundaryCondition condition;
    PrecisionType boundary_size[Dim][2];

    //dafult as open boundary
    Boundary(): condition(OPENBOUNDARY){};

    Boundary(BoundaryCondition& condi, double bound[Dim][2]): condition(condi), boundary_size(bound){
        if (condi!=OPENBOUNDARY){

        }
    };
    
    Boundary(Boundary& geo): condition(geo.condition), boundary_size(geo.boundary_size){
    };

    Boundary<D>& operator=(const Boundary<Dim>& g2){
        if (this==&g2){
            return *this;
        }   

        constexpr double EPSILON = 1E-9;
    	bool isCloseToZero = [EPSILON](double val) { return std::abs(val) < EPSILON; };

        for (int i=0; i<Dim; i++){
            if (isCloseToZero(bounds[i]	//determine grid size using the average distance between objects suggested by the interaction

            for (size_t j=0; j<Dim; j++){
                this->metric[i][j]=g2.metric[i][j];
            }
        };

        this->condition=g2.condition;

        for (size_t i=0; i<Dim; i++){
            for (size_t j=0; j<2; j++){
                this->boundary_size[i][j]=g2.boundary_size[i][j];
            }
        }
        return *this;
    };

    friend std::ofstream& operator<<(std::ofstream& file, const Boundary& object){
        for (size_t i=0; i<D; i++){
            for (size_t j=0; j<D; j++){
                file<<object.metric[i][j]<<" ";
            }
        }

        file<<static_cast<int>(object.condition)<<" ";

        for (size_t i=0; i<D; i++){
            for (size_t j=0; j<2; j++){
                file<<object.boundary_size[i][j]<<" ";
            }
        }

        return file;
        //don't close the file, as << will be called from outside the class
    }

    friend std::ifstream& operator>>(std::ifstream& file, Boundary& object){
        for (size_t i=0; i<D; i++){
            for (size_t j=0; j<D; j++){
                file>>object.metric[i][j];
            }
        }

        int temp;
        file>>temp;
        object.condition=static_cast<BoundaryCondition>(temp);

        for (size_t i=0; i<D; i++){
            for (size_t j=0; j<2; j++){
                file>>object.boundary_size[i][j];
            }
        }

        return file;
    }
};



#endif