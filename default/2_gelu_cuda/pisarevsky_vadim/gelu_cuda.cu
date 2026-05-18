#include <cuda/cmath>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#include "gelu_cuda.h"

__device__ __forceinline__ float fastTanh(float x) {
    x = fmaxf(-9.f, fminf(9.f, x));
    float e = __expf(2.f * x);
    return (e - 1.f) / (e + 1.f);
}

__device__ __forceinline__ float fastGelu(float x) {
    float y = 0.79788456f * x * (1.f + 0.044715f * x * x);
    return 0.5f * x * (1.f + fastTanh(y));
}

__global__ void GeluCUDAKernel(float* X, size_t n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        float x = X[i];
        X[i] = fastGelu(x);
    }
}

void GeluCUDA3xFaster(const std::vector<float>& input, std::vector<float>& output) {
    size_t n = input.size();
    const float* inptr = input.data();

    float* X = nullptr;
    cudaMalloc(&X, n * sizeof(float));
    cudaMemcpy(X, inptr, n * sizeof(float), cudaMemcpyHostToDevice);

    size_t threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    GeluCUDAKernel<<<blocks, threads>>>(X, n);
    output.resize(n);
    float* outptr = output.data();
    cudaDeviceSynchronize();

    cudaMemcpy(outptr, X, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(X);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    std::vector<float> output;
    GeluCUDA3xFaster(input, output);
    return output;
}

#ifdef ADD_TEST
static float geluRef(float x) {
    float y = 0.5f*x*(1 + std::tanh(std::sqrt(2.f/M_PI)*x*(1.f + 0.044715f*x*x)));
    return y;
}

int main() {
    size_t n = 134217728u;
    std::vector<float> x(n), y;
    for (size_t i = 0; i < n; i++) {
        x[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }

    // Warming-up
    y = GeluCUDA(x);

    float err = 0.f;
    for (size_t i = 0; i < n; i++) {
        err = std::max(err, std::abs(y[i] - geluRef(x[i])));
    }
    printf("max absolute error = %.5g\n", err);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 5; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
    #if 0
        auto ytmp = GeluCUDA(x);
    #else
        GeluCUDA3xFaster(x, y);
    #endif
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.2f\n", time);

    return 0;
}
#endif
