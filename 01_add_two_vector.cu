#include "stdio.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 50000 // number of element in array
// BUILD: nvcc -0 01_add_two_vector 01_add_two_vector.cu

__global__ void gpu_add(int *d_a, int* d_b, int *d_c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < N){
        d_c[tid] = d_a[tid] + d_b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(void){
    int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;

    /* cuda allocate the memmory*/
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    /* init input array */
    for(int i = 0; i < N; i++){
        h_a[i] = 2*i*i;
        h_b[i] = i;
    }
    // std::cout << "Vector 1: ";
    // for(int i = 0; i < N; i++){
    //     std::cout << h_a[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "Vector 2: ";
    // for(int i = 0; i < N; i++){
    //     std::cout << h_b[i] << " ";
    // }
    // std::cout << std::endl;

    /* copy array data from host to device */
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    /* call kernel function */
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );
    gpu_add<<<512,512>>> (d_a, d_b, d_c);
    cudaDeviceSynchronize();
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    /* Copy data from device to host */
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // std::cout << "Output: ";
    // for(int i = 0; i < N; i++){
    //     std::cout << h_c[i] << " ";
    // }
    // std::cout << std::endl;

    int error = 0;
    for (int i = 0; i < N; i++)
    {
        if ((h_a[i] + h_b[i] != h_c[i]))
        { error++; }
    }
    std::cout << "Error: " << error << std::endl;

    std::cout << "GPU Times: " << elapsedTime << " ms" << std::endl;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}