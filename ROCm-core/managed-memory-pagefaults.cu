#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>

#ifdef __HIP__
#include <hip/hip_runtime.h>
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMallocManaged hipMallocManaged
#define cudaSuccess hipSuccess
#else
#include <cuda_runtime.h>
#endif

#define CHECK(x)                                                               \
  do {                                                                         \
    cudaError_t err = x;                                                       \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "%s:%d: %s - %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorName(err), cudaGetErrorString(err));            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

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
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

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
    CHECK(cudaMallocManaged(&data, n * sizeof(double)));

    auto [firstAccessTime, firstAccessBandwidth] = accessCpu(data, nAccessed);
    std::printf("  1st access took %7.5fs (BW: %6.1fGB/s)\n", firstAccessTime,
                firstAccessBandwidth);

    auto [secondAccessTime, secondAccessBandwidth] = accessCpu(data, nAccessed);
    std::printf("  2nd access took %7.5fs (BW: %6.1fGB/s)\n", secondAccessTime,
                secondAccessBandwidth);

    CHECK(cudaFree(data));
  }

  std::printf("=== GPU ===\n");
  for (int run = 0; run < runs; ++run) {
    printf(" Run %d:\n", run + 1);

    double *data;
    CHECK(cudaMallocManaged(&data, n * sizeof(double)));

    auto [firstAccessTime, firstAccessBandwidth] = accessGpu(data, nAccessed);
    std::printf("  1st access took %7.5fs (BW: %6.1fGB/s)\n", firstAccessTime,
                firstAccessBandwidth);

    auto [secondAccessTime, secondAccessBandwidth] = accessGpu(data, nAccessed);
    std::printf("  2nd access took %7.5fs (BW: %6.1fGB/s)\n", secondAccessTime,
                secondAccessBandwidth);

    CHECK(cudaFree(data));
  }

  std::printf("=== CPU, then GPU ===\n");
  for (int run = 0; run < runs; ++run) {
    printf(" Run %d:\n", run + 1);

    double *data;
    CHECK(cudaMallocManaged(&data, n * sizeof(double)));

    auto [firstAccessTime, firstAccessBandwidth] = accessCpu(data, nAccessed);
    std::printf("  1st access took %7.5fs (BW: %6.1fGB/s)\n", firstAccessTime,
                firstAccessBandwidth);

    auto [secondAccessTime, secondAccessBandwidth] = accessGpu(data, nAccessed);
    std::printf("  2nd access took %7.5fs (BW: %6.1fGB/s)\n", secondAccessTime,
                secondAccessBandwidth);

    CHECK(cudaFree(data));
  }

  std::printf("=== GPU, then CPU ===\n");
  for (int run = 0; run < runs; ++run) {
    printf(" Run %d:\n", run + 1);

    double *data;
    CHECK(cudaMallocManaged(&data, n * sizeof(double)));

    auto [firstAccessTime, firstAccessBandwidth] = accessGpu(data, nAccessed);
    std::printf("  1st access took %7.5fs (BW: %6.1fGB/s)\n", firstAccessTime,
                firstAccessBandwidth);

    auto [secondAccessTime, secondAccessBandwidth] = accessCpu(data, nAccessed);
    std::printf("  2nd access took %7.5fs (BW: %6.1fGB/s)\n", secondAccessTime,
                secondAccessBandwidth);

    CHECK(cudaFree(data));
  }

  return 0;
}
