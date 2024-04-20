//this is the interface for user provided interactions using curiously recursive template pattern
template<typename T1, typename T2, /*T1, T2 are the data types that the two objects interacting with each other*/ class Derived, size_t Dim>
class Interaction {
public:
    /*device member function to be called by cuda kernel. displacement is the two dimensional relative displacement between two atoms*/
    __device__ __host__  void operator()(T1& obj1, T2& obj2){

    };
    
    __device__ __host__  double getLenthScale(){
        return static_cast<Derived*>(this)->myGetLengthScale();
    }; 

    __device__ __host__  double calPoten(T1& obj1, T2& obj2, double metric[Dim][Dim]){
        return static_cast<Derived*>(this)->myCalPoten(obj1, obj2, metric);
    };           

    //calculate the gradient of the state variable
    __device__ __host__  void calGrad(T1& obj1, T2& obj2, const double metric[Dim][Dim], double metric_gradient[Dim][Dim]){
        static_cast<Derived*>(this)->myCalGrad(obj1, obj2, metric, metric_gradient);   
    };  
    
    __device__ __host__  void calHessian(T1& obj1, T2& obj2, const double metric[Dim][Dim], double metric_hessian[Dim][Dim][Dim][Dim]){
        static_cast<Derived*>(this)->myCalHessian(obj1, obj2, metric, metric_hessian);
    };      
    
    __device__ __host__  void calPseudoGrad(T1& obj1, T2& obj2, const double metric[Dim][Dim], double metric_gradient[Dim][Dim]){
        static_cast<Derived*>(this)->myCalPseudoGrad(obj1, obj2, metric, metric_gradient);
    };  
    
    __device__ __host__  void calPseudoHessian(T1& obj1, T2& obj2, const double metric[Dim][Dim], double metric_hessian[Dim][Dim][Dim][Dim]){
        static_cast<Derived*>(this)->myCalPseudoHessian(obj1, obj2, metric, metric_hessian);
    };  

};