/*
 * AMD-GPU benchmarks
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Measuring page faults in managed-memory allocations
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>

#include "../common/cuda_runtime.hpp"
#include "../common/timing.cuh"

__global__ void accessKernel(double *data, std::size_t n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  data[i] = i;
}

std::tuple<double, double> accessGpu(double *data, const std::size_t n) {
  using clock = std::chrono::high_resolution_clock;
  const auto start = clock::now();

  constexpr unsigned threadsPerBlock = 1024;
  unsigned blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  accessKernel<<<blocks, threadsPerBlock>>>(data, n);
  checkGpuErrors(cudaGetLastError());
  checkGpuErrors(cudaDeviceSynchronize());

  const auto end = clock::now();
  const double time = std::chrono::duration<double>(end - start).count();
  const double bandwidth = n * sizeof(double) / 1e9 / time;
  return {time, bandwidth};
}

[[gnu::noinline]] std::tuple<double, double> accessCpu(double *data,
                                                       std::size_t n) {
  using clock = std::chrono::high_resolution_clock;
  const auto start = clock::now();

#pragma omp parallel for simd nontemporal(data)
  for (std::size_t i = 0; i < n; ++i)
    data[i] = i;

  const auto end = clock::now();
  const double time = std::chrono::duration<double>(end - start).count();
  const double bandwidth = n * sizeof(double) / 1e9 / time;
  return {time, bandwidth};
}

int main(int argc, const char *argv[]) {
  constexpr int runs = 3;

  std::size_t n = 1024ul * 1024 * 1024 * 1024;
  std::size_t nAccessed = 1024ul * 1024 * 1024;

  if (argc > 1)
    n = nAccessed;

  std::printf("=== CPU ===\n");
  for (int run = 0; run < runs; ++run) {
    printf(" Run %d:\n", run + 1);

    double *data;
    checkGpuErrors(cudaMallocManaged(&data, n * sizeof(double)));

    auto [firstAccessTime, firstAccessBandwidth] = accessCpu(data, nAccessed);
    std::printf("  1st access took %7.5fs (BW: %6.1fGB/s)\n", firstAccessTime,
                firstAccessBandwidth);

    auto [secondAccessTime, secondAccessBandwidth] = accessCpu(data, nAccessed);
    std::printf("  2nd access took %7.5fs (BW: %6.1fGB/s)\n", secondAccessTime,
                secondAccessBandwidth);

    checkGpuErrors(cudaFree(data));
  }

  std::printf("=== GPU ===\n");
  for (int run = 0; run < runs; ++run) {
    printf(" Run %d:\n", run + 1);

    double *data;
    checkGpuErrors(cudaMallocManaged(&data, n * sizeof(double)));

    auto [firstAccessTime, firstAccessBandwidth] = accessGpu(data, nAccessed);
    std::printf("  1st access took %7.5fs (BW: %6.1fGB/s)\n", firstAccessTime,
                firstAccessBandwidth);

    auto [secondAccessTime, secondAccessBandwidth] = accessGpu(data, nAccessed);
    std::printf("  2nd access took %7.5fs (BW: %6.1fGB/s)\n", secondAccessTime,
                secondAccessBandwidth);

    checkGpuErrors(cudaFree(data));
  }

  std::printf("=== CPU, then GPU ===\n");
  for (int run = 0; run < runs; ++run) {
    printf(" Run %d:\n", run + 1);

    double *data;
    checkGpuErrors(cudaMallocManaged(&data, n * sizeof(double)));

    auto [firstAccessTime, firstAccessBandwidth] = accessCpu(data, nAccessed);
    std::printf("  1st access took %7.5fs (BW: %6.1fGB/s)\n", firstAccessTime,
                firstAccessBandwidth);

    auto [secondAccessTime, secondAccessBandwidth] = accessGpu(data, nAccessed);
    std::printf("  2nd access took %7.5fs (BW: %6.1fGB/s)\n", secondAccessTime,
                secondAccessBandwidth);

    checkGpuErrors(cudaFree(data));
  }

  std::printf("=== GPU, then CPU ===\n");
  for (int run = 0; run < runs; ++run) {
    printf(" Run %d:\n", run + 1);

    double *data;
    checkGpuErrors(cudaMallocManaged(&data, n * sizeof(double)));

    auto [firstAccessTime, firstAccessBandwidth] = accessGpu(data, nAccessed);
    std::printf("  1st access took %7.5fs (BW: %6.1fGB/s)\n", firstAccessTime,
                firstAccessBandwidth);

    auto [secondAccessTime, secondAccessBandwidth] = accessCpu(data, nAccessed);
    std::printf("  2nd access took %7.5fs (BW: %6.1fGB/s)\n", secondAccessTime,
                secondAccessBandwidth);

    checkGpuErrors(cudaFree(data));
  }

  return 0;
}
