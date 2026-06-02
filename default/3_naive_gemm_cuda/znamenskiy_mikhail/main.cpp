#include <naive_gemm_cuda.h>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>

namespace {
    std::vector<float> NaiveGemmRef(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
        std::vector<float> c(n*n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int cIndex = i * n + j;
                for (int k = 0; k < n; k++) {
                    int aIndex = i * n + k;
                    int bIndex = k * n + j;
                    c[cIndex] += a[aIndex] * b[bIndex];
                }
            }
        }

        return c;
    }
}

int main() {

constexpr size_t N = 1024;
constexpr size_t N2 = N * N;

std::vector<float> a(N2);
std::vector<float> b(N2);

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(-10.f, 10.f);
std::cout << "Generating random data" << std::endl;
for (size_t n = 0; n < N2; ++n) {
    a[n] = dis(gen);
    b[n] = dis(gen);
    }
std::cout << "Generating DONE" << std::endl;

std::cout << "Ref calculations" << std::endl;
std::chrono::steady_clock::time_point beginRef = std::chrono::steady_clock::now();
auto cRef = NaiveGemmRef(a, b, N);
std::chrono::steady_clock::time_point endRef = std::chrono::steady_clock::now();
std::cout << "Ref calculations DONE" << std::endl;
std::cout << "Time REF = " << std::chrono::duration_cast<std::chrono::microseconds>(endRef - beginRef).count() << "[us]" << std::endl;

std::cout << "Warming up" << std::endl;
NaiveGemmCUDA(a, b, N);
std::cout << "Warming up DONE" << std::endl;

std::cout << "Measurements" << std::endl;
std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
auto c = NaiveGemmCUDA(a, b, N);
std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
std::cout << "Measurements done" << std::endl;
std::cout << "Time OPT = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

std::cout << "Accuracy check" << std::endl;
float error = 0.0f;
for (size_t n = 0; n < N2; ++n) {
    error = std::max(std::abs(c[n] - cRef[n]), error);
    if (std::isnan(error)) {
        std::cout << "NAN error - index = " << n << " result = " << c[n] << " ref = " << cRef[n] << std::endl;
        return 1;
    }
}
std::cout << "Accuracy check FINISHED" << std::endl;
std::cout << "Max error = " << error << std::endl;

return 0;
}
