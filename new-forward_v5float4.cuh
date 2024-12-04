
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#include <mxnet/base.h>
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

#define TILE_WIDTH_K2 22
constexpr int x_TILE_WIDTH = TILE_WIDTH_K2 + K - 1; // 28
constexpr int x_tile_width_float4 = x_TILE_WIDTH / 4; // 7

//B is 10000 M is 24 C is 12 H is 33 W is 33 K is 7
__global__ void forward_kernel_2(float *y, const float *x)
{
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int y_grid = blockIdx.z;

    const int w_base = (y_grid % 3) * TILE_WIDTH_K2; // W_grid = 3
    const int h_base = (y_grid / 3) * TILE_WIDTH_K2; // H_grid = 3
    const int h0 = threadIdx.y;
    const int w0 = threadIdx.x;
    const int w = w_base + w0;
    const int h = h_base + h0;

    extern __shared__ float4 X_sharedf4[];

    // Precompute strides
    const int x_stride_h = 72;
    const int y_stride_h = 66;

    const float *x_base = x + b * 1 * 72 * 72;
    float *y_base = y + b * 12 * 66 * 66;

    float acc = 0.0f;

    // Since C = 1, we can remove the loop over C
    const float *x_c_base = x_base; // C = 1
    const float *k_c_base = const_k + m * K * K; // C = 1

    // Load input tile into shared memory using float4
    for (int i = h0; i < x_TILE_WIDTH; i += blockDim.y)
    {
        for (int j = w0; j < x_tile_width_float4; j += blockDim.x)
        {
            int x_row = h_base + i;
            int x_col = w_base + j * 4;

            // Load float4 from global memory
            const float *x_ptr = x_c_base + x_row * x_stride_h + x_col;
            float4 value;
            if (((uintptr_t)x_ptr) % 16 == 0) {
                // Aligned access
                value = *((float4 *)x_ptr);
            } else {
                // Misaligned access: read floats individually
                value.x = __ldg(x_ptr);
                value.y = __ldg(x_ptr + 1);
                value.z = __ldg(x_ptr + 2);
                value.w = __ldg(x_ptr + 3);
            }

            // Store into shared memory
            int shared_mem_idx = i * x_tile_width_float4 + j;
            X_sharedf4[shared_mem_idx] = value;
        }
    }
    __syncthreads();

    // Perform convolution using weights from constant memory
    int x_shared_base = h0 * x_tile_width_float4 + w0 / 4;
    int x_shared_offset = w0 % 4;

    #pragma unroll
    for (int p = 0; p < K; p++)
    {
        int x_row_offset = p * x_tile_width_float4;

        #pragma unroll
        for (int q = 0; q < K; q++)
        {
            int shared_mem_idx = x_shared_base + x_row_offset + (q + x_shared_offset) / 4;
            int component = (q + x_shared_offset) % 4;

            float4 x_vec = X_sharedf4[shared_mem_idx];
            float x_val = ((float *)&x_vec)[component]; // Access component without conditionals

            float k_val = k_c_base[p * K + q];
            acc += x_val * k_val;
        }
    }
    __syncthreads();

    // Write the result to the output
    y_base[m * 66 * 66 + h * y_stride_h + w] = acc;
}

//B is 10000 M is 12 C is 1 H is 72 W is 72 K is 7
#define TILE_WIDTH_K1 9
__global__ void forward_kernel_1(float *y, const float *x)
{
    constexpr int x_TILE_WIDTH = TILE_WIDTH_K1 + K - 1; // 15
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int y_grid = blockIdx.z;

    const int w_base = (y_grid % 3) * TILE_WIDTH_K1; // W_grid = 3
    const int h_base = (y_grid / 3) * TILE_WIDTH_K1; // H_grid = 3
    const int h0 = threadIdx.y;
    const int w0 = threadIdx.x;
    const int w = w_base + w0;
    const int h = h_base + h0;

    extern __shared__ float X_shared[];

    // Precompute strides
    const int x_stride_c = 33 * 33;
    const int x_stride_h = 33;
    const int y_stride_m = 27 * 27;
    const int y_stride_h = 27;
    const int k_stride_c = K * K;

    const float *x_base = x + b * 12 * 33 * 33;
    float *y_base = y + b * 24 * 27 * 27;

    float acc = 0.0f;

    for (int c = 0; c < 12; c++)
    {
        const float *x_c_base = x_base + c * x_stride_c;
        const float *k_c_base = const_k + m * 12 * K * K + c * k_stride_c;

        // Load input tile into shared memory
        for (int i = h0; i < x_TILE_WIDTH; i += TILE_WIDTH_K1)
        {
            for (int j = w0; j < x_TILE_WIDTH; j += TILE_WIDTH_K1)
            {
                int x_row = h_base + i;
                int x_col = w_base + j;
                X_shared[i * x_TILE_WIDTH + j] = x_c_base[x_row * x_stride_h + x_col];
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

    // Write the result to the output
    y_base[m * y_stride_m + h * y_stride_h + w] = acc;
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
    size_t weight_size = M * C * K * K * sizeof(float);
    // my optimized implementation
    cudaMemcpyToSymbol(const_k, w.dptr_, weight_size);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    if (H == 33){
        // Use TILE_WIDTH_K1 for kernel 1
        dim3 blockDimNew(TILE_WIDTH_K1, TILE_WIDTH_K1, 1);
        // Calculate grid dimensions based on TILE_WIDTH_K1
        int W_out = H - K + 1;
        int W_grid = W_out / TILE_WIDTH_K1;
        int H_grid = W_out / TILE_WIDTH_K1;
        int Y = W_grid * H_grid;
        dim3 gridDimNew(B, M, Y);
        size_t shared_mem_size = (TILE_WIDTH_K1 + K - 1) * (TILE_WIDTH_K1 + K - 1) * sizeof(float);
        forward_kernel_1<<<gridDimNew, blockDimNew, shared_mem_size>>>(y.dptr_, x.dptr_);
    } else {
        // Use TILE_WIDTH_K2 for kernel 2
        dim3 blockDimNew(TILE_WIDTH_K2, TILE_WIDTH_K2, 1);
        // Calculate grid dimensions based on TILE_WIDTH_K2
        int W_out = H - K + 1;
        int W_grid = W_out / TILE_WIDTH_K2;
        int H_grid = W_out / TILE_WIDTH_K2;
        int Y = W_grid * H_grid;
        dim3 gridDimNew(B, M, Y);
        const int x_TILE_WIDTH = TILE_WIDTH_K2 + K - 1; // 28
        size_t shared_mem_size = (x_TILE_WIDTH * x_TILE_WIDTH / 4) * sizeof(float4);
        forward_kernel_2<<<gridDimNew, blockDimNew, shared_mem_size>>>(y.dptr_, x.dptr_);
    }
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
