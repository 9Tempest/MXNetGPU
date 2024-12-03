
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 16
namespace mxnet
{
namespace op
{

__global__ void forward_kernel_new(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out, const int W_grid, const int H_grid)
{
// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int y_grid = blockIdx.z;
    const int x_TILE_WIDTH = TILE_WIDTH+K-1;

    const int w_base = y_grid % W_grid * TILE_WIDTH;
    const int h_base = y_grid / W_grid * TILE_WIDTH;
    // assert(h_base == (y_grid % W_grid * TILE_WIDTH) && "My assumption is wrong");
    const int h0 = threadIdx.y;
    const int w0 = threadIdx.x;
    const int w = w_base + w0;
    const int h = h_base + h0;

    extern __shared__ float shmem[];
    float* X_shared = shmem;
    float* W_shared = shmem + x_TILE_WIDTH * x_TILE_WIDTH;
    float acc = 0.0f;

    for (int c = 0; c < C; c++){
        // first load weights into shared memory [kxk]; TODO: put this on constant memory!
        if ((h0 < K) && (w0 < K)){
            W_shared[h0 * K + w0] = k4d(m, c, h0, w0);
        }
        __syncthreads();

        // Load input tile into shared memory
        // Each thread loads multiple elements to cover the entire tile
        for (int i = h0; i < x_TILE_WIDTH; i += TILE_WIDTH) {
            for (int j = w0; j < x_TILE_WIDTH; j += TILE_WIDTH) {
                int x_row = h_base + i;
                int x_col = w_base + j;
                if (x_row < H && x_col < W) {
                    X_shared[i * x_TILE_WIDTH + j] = x4d(b, c, x_row, x_col);
                } else {
                    X_shared[i * x_TILE_WIDTH + j] = 0.0f; // Handle boundary
                }
            }
        }
        __syncthreads();

        for (int p = 0; p < K; p++){
            for (int q = 0; q < K; q++){
                acc += X_shared[(h0 + p) * x_TILE_WIDTH + w0 + q] * W_shared[p * K + q];
            }
        }
        __syncthreads();

    }

    if ((h < H_out) && (w < W_out)){
        y4d(b, m, h, w) = acc;
    }
}

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b = blockDim.x * blockIdx.x + threadIdx.x;

    if (b < B) // for each image in the batch
    {
        for (int m = 0; m < M; m++)         // for each output feature maps
            for (int h = 0; h < H_out; h++) // for each output element
                for (int w = 0; w < W_out; w++)
                {
                    y4d(b, m, h, w) = 0;
                    for (int c = 0; c < C; c++)     // sum over all input feature maps
                        for (int p = 0; p < K; p++) // KxK filter
                            for (int q = 0; q < K; q++)
                                y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
                }
    }

#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // // Use mxnet's CHECK_EQ to do assertions.
    // // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    const int B = x.shape_[0];
    const int M = y.shape_[1]; // num_filter
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    // first calculate output dimensions
    const int H_out = H - K + 1; 
    const int W_out = W - K + 1; 

    // calculate W_grid and H_grid
    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of horizontal tiles per output map
    const int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of vertical tiles per output map
    const int Y = W_grid * H_grid;

    dim3 gridDimNew(B, M, Y);
    dim3 blockDimNew(TILE_WIDTH, TILE_WIDTH,1);
    // Calculate shared memory size: X_shared + W_shared
    size_t shared_mem_size = ((TILE_WIDTH + K) * (TILE_WIDTH + K) + K * K) * sizeof(float);

    // dim3 gridDim((B + 511) / 512);
    // dim3 blockDim(512);

    // // correct implementation
    // MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    // forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    // MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    // my optimized implementation
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    forward_kernel_new<<<gridDimNew, blockDimNew, shared_mem_size>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K, H_out, W_out, W_grid, H_grid);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    assert(0 && "No forward implementation for other datatypes needed for ECE408");
}
}
}

#endif