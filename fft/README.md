# FFT benchmarks

## Build
CMake options:
- `FFT_BENCH_GPU_BACKEND `: `ROCM` or `CUDA`. Build with ROCm or CUDA.
- `FFT_BENCH_BUNDLED_CLI11`: `ON` or `OFF`. Download header-only library for command line parsing.

## Running
The application accepts the following parameters:
- `-n`: Size of the FFT. Between one and three numbers for 1D, 2D or 3D.
- `-b`: FFT batch size.
- `-s`: Number for samples to combute the mean execution time.
- `-p`: The precision to use for computation (`single` or `double`).

Example:
```
./fft_bench -n -256 256 256 -p double -b 1 -s 10
```

