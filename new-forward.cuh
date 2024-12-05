
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_


#include <cuda_runtime.h>
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
__global__ void forward_kernel_2(float *y, const float *x)
{
    constexpr int x_TILE_WIDTH = TILE_WIDTH_K2 + K - 1; // 28
    constexpr int x_TILE_WIDTH_PADDED = x_TILE_WIDTH + 1; // 29

    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int y_grid = blockIdx.z;

    const int w_base = (y_grid % 3) * TILE_WIDTH_K2; // W_grid = 3
    const int h_base = (y_grid / 3) * TILE_WIDTH_K2; // H_grid = 3
    const int h0 = threadIdx.y;
    const int w0 = threadIdx.x;
    const int w = w_base + w0;
    const int h = h_base + h0;

    extern __shared__ float X_shared[];

    // Precompute strides
    constexpr int x_stride_h = 72;
    constexpr int y_stride_h = 66;

    const float *x_base = x + b * 1 * 72 * 72;
    float *y_base = y + b * 12 * 66 * 66;

    float acc = 0.0f;

    // Since C = 1, we can remove the loop over C
    const float *x_c_base = x_base;
    const float *k_c_base = const_k + m * K * K;

    // Load input tile into shared memory with padding
    for (int i = h0; i < x_TILE_WIDTH; i += TILE_WIDTH_K2)
    {
        for (int j = w0; j < x_TILE_WIDTH; j += TILE_WIDTH_K2)
        {
            int x_row = h_base + i;
            int x_col = w_base + j;
            X_shared[i * x_TILE_WIDTH_PADDED + j] = x_c_base[x_row * x_stride_h + x_col];
        }
    }
    __syncthreads();

    // Perform convolution using weights from constant memory
    int x_shared_base = h0 * x_TILE_WIDTH_PADDED + w0;

    #pragma unroll
    for (int p = 0; p < K; p++)
    {
        int x_row_offset = p * x_TILE_WIDTH_PADDED;
        #pragma unroll
        for (int q = 0; q < K; q++)
        {
            int shared_mem_idx = x_shared_base + x_row_offset + q;
            float x_val = X_shared[shared_mem_idx];
            float k_val = k_c_base[p * K + q];
            acc += x_val * k_val;
        }
    }

    // Write the result to the output
    y_base[m * 66 * 66 + h * y_stride_h + w] = acc;
}

#define TILE_WIDTH_K1 9
__global__ void forward_kernel_1(float *y, const float *x)
{
    constexpr int x_TILE_WIDTH = TILE_WIDTH_K1 + K - 1; // 15
    constexpr int x_TILE_WIDTH_PADDED = x_TILE_WIDTH + 1; // 16

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

    // Shared memory double buffering
    float *X_shared_A = X_shared;
    float *X_shared_B = X_shared + x_TILE_WIDTH * x_TILE_WIDTH_PADDED;

    // Precompute strides
    constexpr int x_stride_c = 33 * 33;
    constexpr int x_stride_h = 33;
    constexpr int y_stride_m = 27 * 27;
    constexpr int y_stride_h = 27;

    const float *x_base = x + b * 12 * 33 * 33;
    float *y_base = y + b * 24 * 27 * 27;

    float acc = 0.0f;

    // Load data for the first channel into Buffer A
    int c = 0;
    {
        const float *x_c_base = x_base + c * x_stride_c;

        // Load input tile into shared memory with padding
        for (int i = h0; i < x_TILE_WIDTH; i += TILE_WIDTH_K1)
        {
            for (int j = w0; j < x_TILE_WIDTH; j += TILE_WIDTH_K1)
            {
                int x_row = h_base + i;
                int x_col = w_base + j;
                X_shared_A[i * x_TILE_WIDTH_PADDED + j] = x_c_base[x_row * x_stride_h + x_col];
            }
        }
    }
    __syncthreads();

    // Start from the first channel
    for (c = 0; c < 12; c++)
    {
        // Determine buffers for current and next channels
        float *X_shared_current = (c % 2 == 0) ? X_shared_A : X_shared_B;
        float *X_shared_next = (c % 2 == 0) ? X_shared_B : X_shared_A;

        // Load data for the next channel (if not the last channel)
        if (c + 1 < 12)
        {
            const float *x_c_base_next = x_base + (c + 1) * x_stride_c;

            // Load input tile into shared memory with padding
            for (int i = h0; i < x_TILE_WIDTH; i += TILE_WIDTH_K1)
            {
                for (int j = w0; j < x_TILE_WIDTH; j += TILE_WIDTH_K1)
                {
                    int x_row = h_base + i;
                    int x_col = w_base + j;
                    X_shared_next[i * x_TILE_WIDTH_PADDED + j] = x_c_base_next[x_row * x_stride_h + x_col];
                }
            }
        }

        // // Wait for data to be loaded for the current channel
        // __syncthreads();

        // Perform convolution using weights from constant memory
        const float *k_c_base = const_k + m * 12 * K * K + c * K * K;
        int x_shared_base = h0 * x_TILE_WIDTH_PADDED + w0;
        #pragma unroll
        for (int p = 0; p < K; p++)
        {
            int x_row_offset = p * x_TILE_WIDTH_PADDED;
            #pragma unroll
            for (int q = 0; q < K; q++)
            {
                int shared_mem_idx = x_shared_base + x_row_offset + q;
                float x_val = X_shared_current[shared_mem_idx];
                float k_val = k_c_base[p * K + q];
                acc += x_val * k_val;
            }
        }

        // Synchronize before next iteration if not the last channel
        if (c + 1 < 12)
        {
            __syncthreads();
        }
    }

    // Write the result to the output
    y_base[m * y_stride_m + h * y_stride_h + w] = acc;
}

template <>
void forward<gpu, float>(
    mshadow::Tensor<gpu, 4, float>& y,
    const mshadow::Tensor<gpu, 4, float>& x,
    const mshadow::Tensor<gpu, 4, float>& w)
{
    const int B = x.shape_[0];
    const int M = y.shape_[1]; // num_filter
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];

    // Print shapes for debugging if needed
    PRINT("B is " << B << " M is " << M << " C is " << C << " H is " << H << " W is " << W << " K is " << K);

    // Copy weights to constant memory
    size_t weight_size = M * C * K * K * sizeof(float);
    cudaMemcpyToSymbol(const_k, w.dptr_, weight_size);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    // Create CUDA Graph
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Begin capturing the graph
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Launch the appropriate kernel
    if (H == 33) {
        // Use TILE_WIDTH_K1 for kernel 1
        dim3 blockDimNew(TILE_WIDTH_K1, TILE_WIDTH_K1, 1);
        // Calculate grid dimensions based on TILE_WIDTH_K1
        int W_out = H - K + 1;
        int W_grid = W_out / TILE_WIDTH_K1;
        int H_grid = W_out / TILE_WIDTH_K1;
        int Y = W_grid * H_grid;
        dim3 gridDimNew(B, M, Y);
        // size is x_TILE_WIDTH * x_TILE_WIDTH_PADDED * sizeof(float)
        size_t shared_mem_size = 2 * (TILE_WIDTH_K1 + K - 1) * (TILE_WIDTH_K1 + K) * sizeof(float);

        // Launch kernel on the stream
        forward_kernel_1<<<gridDimNew, blockDimNew, shared_mem_size, stream>>>(y.dptr_, x.dptr_);
    } else {
        // Use TILE_WIDTH_K2 for kernel 2
        dim3 blockDimNew(TILE_WIDTH_K2, TILE_WIDTH_K2, 1);
        // Calculate grid dimensions based on TILE_WIDTH_K2
        int W_out = H - K + 1;
        int W_grid = W_out / TILE_WIDTH_K2;
        int H_grid = W_out / TILE_WIDTH_K2;
        int Y = W_grid * H_grid;
        dim3 gridDimNew(B, M, Y);
        // size is x_TILE_WIDTH * x_TILE_WIDTH_PADDED * sizeof(float)
        size_t shared_mem_size = (TILE_WIDTH_K2 + K - 1) * (TILE_WIDTH_K2 + K) * sizeof(float);

        // Launch kernel on the stream
        forward_kernel_2<<<gridDimNew, blockDimNew, shared_mem_size, stream>>>(y.dptr_, x.dptr_);
    }

    // End capturing the graph
    cudaStreamEndCapture(stream, &graph);

    // Instantiate the graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Destroy the stream used for capturing
    cudaStreamDestroy(stream);

    // Now, whenever you need to execute the operations, launch the graph
    cudaGraphLaunch(graphExec, 0);

    // Wait for the graph execution to complete
    cudaStreamSynchronize(0);

    // Clean up the graph resources
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);

    // Synchronize the device
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
