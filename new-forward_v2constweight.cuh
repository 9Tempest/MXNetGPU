
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_



#include <mxnet/base.h>
#define TILE_WIDTH 16
#define STATS

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
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + \
                                    (i2) * (H_out * W_out) + \
                                    (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + \
                                    (i2) * (H * W) + \
                                    (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) const_k[(i3) * (C * K * K) + \
                                         (i2) * (K * K) + \
                                         (i1) * (K) + i0]

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

    float acc = 0.0f;

    for (int c = 0; c < C; c++)
    {
        // Load input tile into shared memory
        for (int i = h0; i < x_TILE_WIDTH; i += TILE_WIDTH)
        {
            for (int j = w0; j < x_TILE_WIDTH; j += TILE_WIDTH)
            {
                int x_row = h_base + i;
                int x_col = w_base + j;
                if (x_row < H && x_col < W)
                {
                    X_shared[i * x_TILE_WIDTH + j] = x4d(b, c, x_row, x_col);
                }
                else
                {
                    X_shared[i * x_TILE_WIDTH + j] = 0.0f; // Handle boundary
                }
            }
        }
        __syncthreads();

        // Perform convolution using weights from constant memory
        #pragma unroll
        for (int p = 0; p < K; p++)
        {
            #pragma unroll
            for (int q = 0; q < K; q++)
            {
                int x_row = h0 + p;
                int x_col = w0 + q;
                acc += X_shared[x_row * x_TILE_WIDTH + x_col] * k4d(m, c, p, q);
            }
        }
        __syncthreads();
    }

    if (h < H_out && w < W_out)
    {
        y4d(b, m, h, w) = acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
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
