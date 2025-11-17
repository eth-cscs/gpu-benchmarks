#include <CLI/CLI.hpp>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "gpu_fft_api.hpp"
#include "gpu_runtime_api.hpp"

// Benchmark a given FFT size in 1D, 2D or 3D with given batch size and number
// of samples to measure.
// Returns the mean time of execution measured on GPU.
template <typename T>
float bench_fft(std::vector<int> sizes, int n_batch, int n_sample) {
  using complex_t = typename gpu::fft::ComplexType<T>::type;

  const int batch_size =
      std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int>());
  std::vector<complex_t> input_host(n_batch * batch_size);

  complex_t *input_device;
  complex_t *output_device;

  gpu::api::malloc((void **)&input_device,
                   input_host.size() * sizeof(complex_t));
  gpu::api::malloc((void **)&output_device,
                   input_host.size() * sizeof(complex_t));

  std::minstd_rand rand_gen(42);
  std::uniform_real_distribution<T> rand_dist(-3, 3);

  // initialize input to avoid potential performance impact of nonsensical
  // input to sin / cos functions in FFT
  for (auto &val : input_host) {
    val.x = rand_dist(rand_gen);
    val.y = rand_dist(rand_gen);
  }

  gpu::api::memcpy(input_device, input_host.data(),
                   input_host.size() * sizeof(complex_t),
                   gpu::api::flag::MemcpyHostToDevice);

  gpu::fft::HandleType plan;

  gpu::fft::plan_many(&plan, sizes.size(), sizes.data(), nullptr, 1, batch_size,
                      nullptr, 1, batch_size,
                      gpu ::fft::TransformType::ComplexToComplex<T>::value,
                      n_batch);

  gpu::api::EventType event_start, event_end;

  gpu::api::event_create(&event_start);
  gpu::api::event_create(&event_end);

  // run once to warm up
  gpu::fft::execute(plan, input_device, output_device,
                    gpu::fft::TransformDirection::Backward);

  // compute n_sample FFTs
  gpu::api::event_record(event_start);
  for (int i = 0; i < n_sample; ++i) {
    gpu::fft::execute(plan, input_device, output_device,
                      gpu::fft::TransformDirection::Backward);
  }
  gpu::api::event_record(event_end);
  float time = 0;
  gpu::api::event_synchronize(event_end);
  gpu::api::event_elapsed_time(&time, event_start, event_end);

  std::ignore = gpu::api::event_destroy(event_start);
  std::ignore = gpu::api::event_destroy(event_end);
  std::ignore = gpu::fft::destroy(plan);
  std::ignore = gpu::api::free(input_device);
  std::ignore = gpu::api::free(output_device);

  return time / n_sample;
}

int main(int argc, char **argv) {
  std::vector<int> sizes;
  std::string precision;
  int n_sample = 1;
  int n_batch = 1;

  CLI::App app{"FFT benchmark"};
  app.add_option("-n", sizes,
                 "FFT size. Between 1 and 3 space-separated values for 1D, 2D or 3D FFTs.")
      ->required()
      ->expected(1, 3)
      ->check(CLI::NonNegativeNumber);
  app.add_option("-s", n_sample, "Number of samples for time measurement")
      ->check(CLI::NonNegativeNumber)
      ->default_val(10);
  app.add_option("-b", n_batch, "FFT batch size")
      ->check(CLI::NonNegativeNumber)
      ->default_val(1);
  app.add_option("-p", precision, "Precision. \"single\" or \"double\"")
      ->check(CLI::IsMember({"single", "double"}))
      ->default_val("double");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  const auto time = precision == "single"
                        ? bench_fft<float>(sizes, n_batch, n_sample)
                        : bench_fft<double>(sizes, n_batch, n_sample);

  std::cout << "==== FFT Benchmark ====" << std::endl;

  std::cout << "Parameters: size = (" << sizes[0];
  if (sizes.size() >= 2) std::cout << ", " << sizes[1];
  if (sizes.size() >= 3) std::cout << ", " << sizes[2];
  std::cout << ")";

  std::cout << ", batch size = " << n_batch << ", samples = " << n_sample
            << ", precision = " << precision << std::endl;

  std::cout << "Mean time [ms]: " << time << std::endl;

  return 0;
}
