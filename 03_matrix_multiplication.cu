#include "stdio.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#define TILE_SIZE 2

__global__ void gpu_Matrix_Mul_nonshared(float *d_a, float *d_b, float *d_c, const int size)
{
    int row, col;
    col = TILE_SIZE * blockIdx.x + threadIdx.x;
    row = TILE_SIZE * blockIdx.y + threadIdx.y;

    for(int k = 0; k < size; k++){
        d_c[row * size + col] += d_a[row * size + k] * d_b[k * size + col];
    }
}

__global__ void gpu_Matrix_Mul_shared(float *d_a, float *d_b, float *d_c, const int size)
{
    int row, col;
    col = TILE_SIZE * blockIdx.x + threadIdx.x;
    row = TILE_SIZE * blockIdx.y + threadIdx.y;

    __shared__ float share_a[TILE_SIZE][TILE_SIZE];
    __shared__ float share_b[TILE_SIZE][TILE_SIZE];

    for(int i = 0; i < size/TILE_SIZE; i++){
        share_a[threadIdx.y][threadIdx.x] = d_a[row * size + (i*TILE_SIZE + threadIdx.x)];
        share_b[threadIdx.y][threadIdx.x] = d_b[(i*TILE_SIZE + threadIdx.x) * size + col];
        __syncthreads();

        for(int j = 0; j < TILE_SIZE; j++){
            d_c[row * size + col] += share_a[threadIdx.y][j] * share_b[j][threadIdx.x];
        }
        __syncthreads();
    }
}

int main(void){
    const int size = 4;

    float h_a[size][size];
    float h_b[size][size];
    float h_result[size][size];

    float *d_a, *d_b, *d_result;

    // Init array
    for(int i=0; i < size; i++){
        for(int j=0; j<size;j++){
            h_a[i][j] = i;
            h_b[i][j] = j;
        }
    }

    cudaMalloc((void **)&d_a, size * size * sizeof(int));
    cudaMalloc((void **)&d_b, size * size * sizeof(int));
    cudaMalloc((void **)&d_result, size * size * sizeof(int));

    cudaMemcpy(d_a, h_a, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(size/TILE_SIZE, size/TILE_SIZE, 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

    // gpu_Matrix_Mul_nonshared << <dimGrid, dimBlock >> > (d_a, d_b, d_result, size);
    gpu_Matrix_Mul_shared<<<dimGrid, dimBlock>>> (d_a, d_b, d_result,size);

    cudaMemcpy(h_result, d_result, size*size*sizeof(int), cudaMemcpyDeviceToHost);
	printf("The result of Matrix multiplication is: \n");
	
	for (int i = 0; i< size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			printf("%f   ", h_result[i][j]);
		}
		printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    
    return 0;
}