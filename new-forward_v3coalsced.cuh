
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_



#include <mxnet/base.h>
#define TILE_WIDTH 16
// #define STATS

#ifdef STATS
#define PRINT(s) std::cout << s << std::endl;
#else
#define PRINT(s)
#endif
namespace mxnet
{
namespace op
{

#define MAX_M 24  // Maximum number of output feature maps
#define MAX_C 12  // Maximum number of input feature maps
#define K 7       // Kernel size (assuming K is fixed at 7)

__constant__ float const_k[MAX_M * MAX_C * K * K];

__global__ void forward_kernel_new(float *y, const float *x, const int B, const int M, const int C,
                                   const int H, const int W, const int H_out, const int W_out,
                                   const int W_grid, const int H_grid)
{
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int y_grid = blockIdx.z;
    constexpr int x_TILE_WIDTH = TILE_WIDTH + K - 1;

    const int w_base = (y_grid % W_grid) * TILE_WIDTH;
    const int h_base = (y_grid / W_grid) * TILE_WIDTH;
    const int h0 = threadIdx.y;
    const int w0 = threadIdx.x;
    const int w = w_base + w0;
    const int h = h_base + h0;

    extern __shared__ float X_shared[];

    // Precompute strides
    int x_stride_c = H * W;
    int x_stride_h = W;

    int y_stride_m = H_out * W_out;
    int y_stride_h = W_out;

    int k_stride_c = K * K;
    int k_stride_p = K;

    const float *x_base = x + b * C * H * W;
    float *y_base = y + b * M * H_out * W_out;

    float acc = 0.0f;

    for (int c = 0; c < C; c++)
    {
        const float *x_c_base = x_base + c * x_stride_c;
        const float *k_c_base = const_k + m * C * K * K + c * k_stride_c;

        // Load input tile into shared memory
        for (int i = h0; i < x_TILE_WIDTH; i += blockDim.y)
        {
            for (int j = w0; j < x_TILE_WIDTH; j += blockDim.x)
            {
                int x_row = h_base + i;
                int x_col = w_base + j;
                float value = 0.0f;
                if (x_row < H && x_col < W)
                {
                    value = x_c_base[x_row * x_stride_h + x_col];
                }
                X_shared[i * x_TILE_WIDTH + j] = value;
            }
        }
        __syncthreads();

        // Perform convolution using weights from constant memory
        int x_shared_base = h0 * x_TILE_WIDTH + w0;
        #pragma unroll
        for (int p = 0; p < K; p++)
        {
            int x_row_offset = p * x_TILE_WIDTH;
            #pragma unroll
            for (int q = 0; q < K; q++)
            {
                int shared_mem_idx = x_shared_base + x_row_offset + q;
                float x_val = X_shared[shared_mem_idx];
                float k_val = k_c_base[p * K + q];
                acc += x_val * k_val;
            }
        }
        __syncthreads();
    }

    if (h < H_out && w < W_out)
    {
        y_base[m * y_stride_m + h * y_stride_h + w] = acc;
    }
}
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

    // Print all these shapes 
    PRINT("B is " << B << " M is " << M << " C is " << C << " H is " << H << " W is " << W << " K is " << K); 

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
    size_t shared_mem_size = ((TILE_WIDTH + K -1) * (TILE_WIDTH + K -1)) * sizeof(float);
    // print shared memory size
    PRINT("Shared memory size is " << shared_mem_size);
    size_t weight_size = M * C * K * K * sizeof(float);
    // my optimized implementation
    cudaMemcpyToSymbol(const_k, w.dptr_, weight_size);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    forward_kernel_new<<<gridDimNew, blockDimNew, shared_mem_size>>>(y.dptr_, x.dptr_,B, M, C, H, W, H_out, W_out, W_grid, H_grid);
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
