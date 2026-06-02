#include <cuda/cmath>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <float.h>
#include <limits>
#include <stdlib.h>
#include <stdio.h>

#include "softmax_cuda.h"

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

__device__ __inline__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __inline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __inline__ float blockReduceMax(float val, float* shared_cache) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    val = warpReduceMax(val);

    if (lane == 0) shared_cache[wid] = val;
    __syncthreads();

    float final_max = (tid < NUM_WARPS) ? shared_cache[lane] : -FLT_MAX;
    if (wid == 0) final_max = warpReduceMax(final_max);

    if (tid == 0) shared_cache[0] = final_max;
    __syncthreads();

    return shared_cache[0];
}

__device__ __inline__ float blockReduceSum(float val, float* shared_cache) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    val = warpReduceSum(val);

    if (lane == 0) shared_cache[wid] = val;
    __syncthreads();

    float final_sum = (tid < NUM_WARPS) ? shared_cache[lane] : 0.0f;
    if (wid == 0) final_sum = warpReduceSum(final_sum);

    if (tid == 0) shared_cache[0] = final_sum;
    __syncthreads();

    return shared_cache[0];
}

__global__ void vecSoftmax(const float* X, float* Y, int row_size, int col_size) {
    int row = blockIdx.x; 
    if (row >= row_size) return;

    int tid = threadIdx.x;
    __shared__ float shared_cache[NUM_WARPS];

    float local_max = -FLT_MAX;
    for (int col = tid; col < col_size; col += BLOCK_SIZE) {
        local_max = fmaxf(local_max, X[row * col_size + col]);
    }
    float final_max = blockReduceMax(local_max, shared_cache);

    float local_exp_sum = 0.0f;
    for (int col = tid; col < col_size; col += BLOCK_SIZE) {
        local_exp_sum += expf(X[row * col_size + col] - final_max);
    }
    float final_sum = blockReduceSum(local_exp_sum, shared_cache);

    for (int col = tid; col < col_size; col += BLOCK_SIZE) {
        int idx = row * col_size + col;
        Y[idx] = expf(X[idx] - final_max) / final_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_size) {
    size_t col_size = input.size() / row_size;
    size_t bytes = input.size() * sizeof(float);

    float* X = nullptr;
    float* Y = nullptr;

    cudaMalloc(&X, bytes);
    cudaMalloc(&Y, bytes);

    cudaMemcpy(X, input.data(), bytes, cudaMemcpyHostToDevice);

    vecSoftmax<<<row_size, BLOCK_SIZE>>>(X, Y, row_size, col_size);

    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), Y, bytes, cudaMemcpyDeviceToHost);

    cudaFree(X);
    cudaFree(Y);

    return output;
}

#if 0
std::vector<float> SoftmaxRef(const std::vector<float>& input, int row_size) {
    size_t col_size = input.size() / row_size;
    std::vector<float> output(row_size * col_size);

    const float* inptr = input.data();
    float* outptr = output.data();

    for (size_t i = 0; i < row_size; i++) {
        const float* row_in = inptr + i * col_size;
        float* row_out = outptr + i * col_size;

        float max = std::numeric_limits<float>::lowest();
        for (size_t j = 0; j < col_size; j++) {
            if (row_in[j] > max) {
               max = row_in[j]; 
            }
        }

        float sum = 0.f;
        std::vector<float> exps(col_size);
        for (size_t j = 0; j < col_size; j++) {
            float e = std::exp(row_in[j] - max);
            exps[j] = e;
            sum += e;
        }

        for (size_t j = 0; j < col_size; j++) {
            row_out[j] = exps[j] / sum;
        }
    }

    return output;
}

int main() {
    size_t row_size = 8192;
    size_t col_size = 16384;
    std::vector<float> input(row_size * col_size);
    for (size_t i = 0; i < row_size * col_size; i++) {
        input[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }

    // Warming-up
    auto output = SoftmaxCUDA(input, row_size);

    std::vector<float> outref = SoftmaxRef(input, row_size);
    float err = 0.f;
    for (size_t i = 0; i < row_size * col_size; i++) {
        err = std::max(err, std::abs(output[i] - outref[i]));
    }
    printf("max absolute error = %.5g\n", err);
    
    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto output = SoftmaxCUDA(input, row_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.4f\n", time);

    return 0;
}
#endif
