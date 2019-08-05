/*  Sourav
 *  Bihani
 *  souravbi
 */
#include <iostream>
#include <math.h>
#include <algorithm>
#include <functional>

#ifndef A3_HPP
#define A3_HPP

__global__ void Device(float *x_kernel , float *y_kernel , int num_size, float h){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const double pi = 3.1415;
    float sum = 0;
    if(idx < num_size)
    {
        for(int j = 1 ; j < num_size ; j++)
        {
            sum = sum + (( 1 / sqrt(2 * pi)) * exp ( - ( ( pow ( ( ( x_kernel[idx] - x_kernel[j] ) / h ), 2) ) / 2)));
        }
        y_kernel[idx] = sum / (num_size * h);
    }
}
void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
	const int thread_size = 1024;
	int num_blocks = (n + thread_size - 1) / thread_size;
	float* x_kernel;
	float* y_kernel;
	
	cudaMallocManaged(&x_kernel,sizeof(float)* n); 
   	cudaMallocManaged(&y_kernel,sizeof(float)* n);
	
	cudaMemcpy(x_kernel, x.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    
    Device<<<num_blocks , thread_size >>>(x_kernel , y_kernel , n , h); 
    
    cudaMemcpy( y.data(),y_kernel, sizeof(float) * n, cudaMemcpyDeviceToHost);  
        
	 
    cudaFree(x_kernel); 
	cudaFree(y_kernel);
	 

} // gaussian_kde

#endif // A3_HPP
