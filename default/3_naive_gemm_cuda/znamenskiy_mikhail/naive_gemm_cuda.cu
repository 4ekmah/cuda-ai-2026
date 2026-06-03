#include "naive_gemm_cuda.h"

#include <cuda/cmath>

__global__ void NaiveGemmCUDAImpl(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[i * n + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const int mxSize = a.size();
    const int mxSizeBytes = mxSize * sizeof(float);

    float* gpuBuffer = nullptr;
    // Allocating Cuda memory once
    cudaMalloc(&gpuBuffer, 3 * mxSizeBytes);
    cudaMemcpy(gpuBuffer, a.data(), mxSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuBuffer + mxSize, b.data(), mxSizeBytes, cudaMemcpyHostToDevice);
    cudaMemset(gpuBuffer + 2 * mxSize, 0, mxSizeBytes);

    constexpr int threads = 16;
    int blocks = cuda::ceil_div(n, threads);
    dim3 threadsXY(threads, threads);
    dim3 blocksXY(blocks, blocks);
    NaiveGemmCUDAImpl<<<blocksXY, threadsXY>>>(gpuBuffer, gpuBuffer + mxSize, gpuBuffer + 2 * mxSize, n);

    // Allocating result buffer while CUDA calculations are running
    std::vector<float> c(mxSize);
    float* cData = c.data();

    cudaDeviceSynchronize();
    cudaMemcpy(cData, gpuBuffer + 2 * mxSize, mxSizeBytes, cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);

    return c;
}
