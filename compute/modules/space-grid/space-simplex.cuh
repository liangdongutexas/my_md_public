#ifndef SPACE_BLOCK_CUH
#define SPACE_BLOCK_CUH

#include "../../include/defines.cuh" 

// Grouping for dimensionality parameters
struct SpaceParams {
    static constexpr size_t Fdim = 2;
    static constexpr size_t Buffersize=10; 
};

/**
 * @brief the basic space building block each initiate a cuda thread during calculation
 *        the reason for choosing its shape as simplex is for rendering convenient with Vulkan and potentially connection with simplicial quantum gravity
 *        ActorTypes represent all the Lagrangian description objects that has a comparable size effect with the symplex. Much smaller but scarse objects are
 *        ignored. And much larger objects are described by the Eulerian description contained in the field values.
 * @tparam Level size_t  the highers level space block has larger size and is a more coarse grained description
 * @tparam Dim size_t the dimensionality of the space e.g. 3 for three dimension space
 * @tparam FDim size_t the dimensionality of the field vector at each position e.g. maxwell fields plus fluid fields agrregated into a single large vector
 * @tparam Types typename all different types of actors that to be registered at each space block according to their position
 */

template <typename Derived, typename Params, size_t Dim, typename FinnerBlock, typename... ActorTypes>
class SpaceSimplex {
public:
/**
 * @brief smaller space block in case where more detailed simulation is needed
 * how to realize this is undecided by now
 */

/**
 * @brief metric tensor field
 * 
 */
    double metric[Dim][Dim];

/**
* @brief a vector represent internal states of the space block and its size information
*/
    double field[Params::Fdim];

private:
/**
 * @brief actors registered at the space block when their position is located within the block
 * @details actors_table is a pointer array with each type of actor corresponding to a pointer, numTypes is the total number of actor types
 * occupy_table labels whether the buffer locate is occupied by a actor or not
 * 
 */ 
    bool** occupy_table;
    void** actors_table;
    const size_t numTypes;

/**
 * @brief finer_mesh a pointer reserved for adaptive mesh refinement
 * 
 */
    FinnerBlock* finner_mesh;

public:
    SpaceSimplex() : numTypes(sizeof...(ActorTypes)) {
        // Allocate array of void actors_table
        CUDA_CHECK(cudaMallocManaged(&occupy_table, numTypes*sizeof(void*)));
        CUDA_CHECK(cudaMallocManaged(&actors_table, numTypes*sizeof(void*)));
        // Allocate managed memory for each type
        allocateActors(std::index_sequence_for<ActorTypes...>{});
    }

    ~SpaceSimplex() {
        freeActors(std::index_sequence_for<ActorTypes...>{});
        CUDA_CHECK(cudaFree(occupy_table));
        CUDA_CHECK(cudaFree(actors_table));
    }

    // Get pointer to managed memory for a given type index
    template <std::size_t N>
    auto get() -> typename std::tuple_element<N, std::tuple<ActorTypes...>>::type* {
        return static_cast<typename std::tuple_element<N, std::tuple<ActorTypes...>>::type*>(actors_table[N]);
    }

protected:
    __device__ __host__ void interactFieldField() {
        static_cast<Derived*>(this)->special_interactFieldField(actors_table[I], occupy_table[I]);
    }

    template<typename Actor>
    __device__ __host__ void interactFieldActor(Actor& actor) {
        static_cast<Derived*>(this)->special_interactActorField(actor); 
    }

    template <typename Actor1, typename Actor2>
    __device__ __host__ void interactActorActor (Actor1& actor1, Actor2& actor2){
        static_cast<Derived*>(this)->special_interactActorActor(actor1, actor2); 
    }

private:
    // Helper to allocate managed memory for each type in the variadic list
    template <std::size_t... I>
    void allocateActors(std::index_sequence<I...>) {
        // Using a fold-expression to call allocateManaged for each type
        (..., (allocateManaged<ActorTypes>(actors_table[I], occupy_table[I])));
    }

    // Utility to allocate memory and store the pointer in a void pointer
    template <typename T>
    void allocateManaged(void*& ptr, bool*& occupy_ptr) {
        CUDA_CHECK(cudaMallocManaged(&ptr, Params::Buffersize*sizeof(T)));
        CUDA_CHECK(cudaMallocManaged(&occupy_ptr, Params::Buffersize*sizeof(bool)));

        //call the default constructor
        for (size_t i=0; i<Params::Buffersize; ++i){
            new (static_cast<T*>(ptr) + i) T();
        }
    }

    // Helper to free managed memory for each type in the variadic list
    template <std::size_t... I>
    void freeActors(std::index_sequence<I...>) {
        // Using a fold-expression to call allocateManaged for each type
        (..., (freeManaged<ActorTypes>(actors_table[I])));
    }

    // Utility to free memory
    template <typename T>
    void freeManaged(void* ptr) {
        T* typedPtr = static_cast<T*>(ptr);
        //call the default constructor
        for (size_t i=0; i<Params::Buffersize; ++i){
            typedPtr[i].~T();
        }
        CUDA_CHECK(cudaFree(ptr));
    }

};

#endif