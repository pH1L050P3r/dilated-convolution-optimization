#include "cuda.h"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

using namespace std;

__global__ void convolution
(
    int input_row,
    int input_col,
    int *input,
    int kernel_row,
    int kernel_col,
    int *kernel,
    int output_row,
    int output_col,
    long long unsigned int *output
)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < output_row && col < output_col) {
        int kernel_base = 0;
        long long unsigned int acc = 0;
        int input_base = 0;
        for (int i = 0; i < kernel_row; i++){
            int r = (row + i + i) % input_row;
            input_base = r * input_col;
            for (int j = 0; j < kernel_col; j++) {
                int c = (col + j + j) % input_col;
                acc += input[input_base + c] * kernel[kernel_base + j];
            }
            kernel_base += kernel_col;
        }
        output[row * output_col + col] = acc;
    }
}

// Fill in this function
void gpuThread(
    int input_row,
    int input_col,
    int *input,
    int kernel_row,
    int kernel_col,
    int *kernel,
    int output_row,
    int output_col,
    long long unsigned int *output
) {

    dim3 blocks(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    size_t grid_x = (output_col + blocks.x - 1) / blocks.x;
    size_t grid_y = (output_row + blocks.y - 1) / blocks.y;
    dim3 grid(grid_x, grid_y);

    int *d_input, *d_kernel;
    long long unsigned int *d_out;

    cudaMalloc(&d_kernel, sizeof(int) * kernel_row * kernel_col);
    cudaMalloc(&d_input, sizeof(int) * input_row * input_col);
    cudaMalloc(&d_out, sizeof(long long unsigned int) * output_row * output_col);

    cudaMemcpy(d_input, input, sizeof(int) * input_row * input_col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(int) * kernel_row * kernel_col, cudaMemcpyHostToDevice);

    convolution<<<grid, blocks>>>(input_row,
                                  input_col,
                                  d_input,
                                  kernel_row,
                                  kernel_col,
                                  d_kernel,
                                  output_row,
                                  output_col,
                                  d_out);

    cudaMemcpy(output, d_out, sizeof(long long unsigned int) * output_row * output_col, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_out);
    cudaFree(d_kernel);
}