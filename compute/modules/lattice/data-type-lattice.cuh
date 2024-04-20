#ifndef DATA_TYPE_LATTICE_CUH
#define DATA_TYPE_LATTICE_CUH

#include <fstream>


//contains property of a single atoms: displacement field Lagrangian description
template<size_t Dim /*dimension of the atom*/>
struct Atom {
//define data type for the member data of Atom class
enum AtomType {   
        BULK,
        BOUNDARY,
        DISLOCATION,
        DISCLINATION,
        DEAD
    };
    
enum ContriToPotential {
    CONTRIBUTING = 0,
    NON_CONTRIBUTING = 1
};

enum FixedMini {
    OFF = 0,
    ON = 1
};

private:
    const double mass; 

	//type of the atoms
	AtomType atom_type=BULK;

	//whether it contributes to interaction
	ContriToPotential contri_to_potential=CONTRIBUTING;

	//whether fixed during minimization
	FixedMini fixed_mini=OFF;

public:
    Atom(double m): mass(m){
        for (int i=0; i<Dim; i++){
            distances[i][0]=DBL_MAX;
            distances[i][1]=DBL_MAX;
        }
    };
    
    //position of the atom in 2d plane
    double posi[Dim]={};
    double velocity[Dim]={};
    

    //the potential energy of the atom with others
    double poten=0;
    //force exerted on the atom
    double grad[Dim];
    //hessian matrix for minimization purposes
    double hessian[Dim][Dim];

    __host__ __device__ void move(const size_t& time_step, const double metric[Dim][Dim]);
    __host__ __device__ void minimize(const size_t& time_step, bool hessian=false);

    
    /*To understand why define the following data, we need to understand the algorithm to find all adjacent atoms: among all the potential atoms, 
      1. calculate relative displacement, 
      2. categorize them acoording to which component of the relative displacement is the largest
      3. find the atom of the smallest relative displacement in each category
      This algorithm indicates the data needed:
    */
    //all adjacent atoms       
    Atom* adja_atom_ptrs[Dim][2];
    //the current distances of each adjacent atom
    double distances[Dim][2];


	__device__ __host__ inline void killAtom(){
		atom_type=DEAD;
		contri_to_potential=CONTRIBUTING;
		fixed_mini=ON;
	}

    __device__ __host__ inline void setDislo(){
        if (atom_type!=DEAD){
            atom_type=DISLOCATION;
        }
        
	}

    __device__ __host__ inline void setBulk(){
        if (atom_type!=DEAD){
            atom_type=BULK;
        }
	}

    __device__ __host__ inline void setBound(){
        if (atom_type!=DEAD){
            atom_type=BOUNDARY;
        }
	}

    __device__ __host__ inline void setDiscl(){
        if (atom_type!=DEAD){
            atom_type=DISCLINATION;
        }
	}

    __device__ __host__ inline void fixMini(){
        fixed_mini=ON;
	}

    __device__ __host__ inline void relaxMini(){
        if (atom_type!=DEAD){
            fixed_mini=OFF;
        }
	}

    __device__ __host__ inline void ignorePoten(){
        contri_to_potential=NON_CONTRIBUTING;
	}

    __device__ __host__ inline void acknowledgePoten(){
        if (atom_type!=DEAD){
            contri_to_potential=CONTRIBUTING;
        }
	}

    __device__ __host__ inline AtomType queryAtomtype(){return atom_type;};

    __device__ __host__ inline ContriToPotential queryContriPoten(){return contri_to_potential;};

    __device__ __host__ inline FixedMini queryFixedMini(){return fixed_mini;};


    //save and load data of the atom class instance
    friend std::ofstream& operator<<(std::ofstream& file, const Atom& object){
        file<<objec.mass<<" "<<static_cast<int>(atom_type)<<" "<<static_cast<int>(contri_to_potential)
        <<" "<<static_cast<int>(fixed_mini)<<" ";    

        for (int i=0; i<D; i++){
            file<<posi[i]<<" ";
        }

        return file;
    }

    friend std::ifstream& operator>>(std::ifstream& file, Geometry& object){
        file>>objec.mass;
        int temp;
        file>>temp;
        atom_type=static_cast<AtomType>(temp);
        file>>temp;
        contri_to_potential=static_cast<ContriToPotential>(temp);
        file>>temp;
        fixed_mini=static_cast<FixedMini>(temp);
        for (int i=0; i<D; i++){
            file>>posi[i];
        }
        return file;
    }
};



template<size_t Dim>
struct ShapeLocation{
    double center_position[Dim]={};
    double bounds[Dim][2]={};

    __device__ __host__ ShapeLocation<Dim>& operator=(const ShapeLocation<Dim>& rhs) {
        if (&rhs == this) {
            return *this; // handle self assignment
        }

        for (size_t i = 0; i < Dim; i++){
            this->center_position[i] = rhs.center_position[i];
            for (size_t j = 0; j < 2; j++){
                this->bounds[i][j] = rhs.bounds[i][j];
            }
        }

        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const ShapeLocation<Dim>& obj) {
        for (size_t i = 0; i < D; i++){
            os << obj.center_position[i] << " ";
            for (size_t j = 0; j < 2; j++){
                os << obj.bounds[i][j] << " ";
            }
            os << "\n";
        }

        return os;
    }

    friend std::istream& operator>>(std::istream& is, ShapeLocation<Dim>& obj) {
        for (size_t i = 0; i < D; i++){
            is >> obj.center_position[i];
            for (size_t j = 0; j < 2; j++){
                is >> obj.bounds[i][j];
            }
        }

        return is;
    }
};

template <size_t Dim>
__host__ __device__ void Atom<D>:: move(const size_t& time_step, const double metric[Dim][Dim]) {
    for (int i=0; i<Dim; i++){
        posi[i]+=velocity[i]*time_step;
        velocity[i]+=-grad[i]
    }


}

template <size_t Dim>
__host__ __device__ void Atom<D>:: minimize(const size_t& time_step, bool hessian=false) {

}






#endif