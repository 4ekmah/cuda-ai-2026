#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    constexpr float sqrt_2_pi = M_2_SQRTPI * M_SQRT1_2;
    std::vector<float> result(input);
#pragma omp parallel for
    for (auto& el : result) {
        el = 0.5f * el * (1 + std::tanh(sqrt_2_pi * (el + 0.044715 * el * el * el)));
    }
    return result;
}

#if 0
int main() {
    const size_t input_size = 2000000;
    std::vector<float> input(input_size);
    // Warming-up
    GeluOMP(input);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 10; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        GeluOMP(input);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    std::cout << time << std::endl;
}
#endif