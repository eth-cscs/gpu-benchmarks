cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(rocfft_bench LANGUAGES CXX VERSION 0.1)

# set language and standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

cmake_policy(SET CMP0135 NEW)

# set default build type to RELEASE
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo"
	)
endif()

# Options
option(FFT_BENCH_BUNDLED_CLI11 "Use bundled CLI11" ON)
set(FFT_BENCH_GPU_BACKEND "ROCM" CACHE STRING "GPU backend")
set_property(CACHE FFT_BENCH_GPU_BACKEND PROPERTY STRINGS
	"CUDA" "ROCM"
	)

# Get GNU standard install prefixes
include(GNUInstallDirs)

set(FFT_BENCH_EXTERNAL_LIBS)

# Options combination check
set(FFT_BENCH_CUDA OFF)
set(FFT_BENCH_ROCM OFF)
if(FFT_BENCH_GPU_BACKEND STREQUAL "CUDA")
	set(FFT_BENCH_CUDA ON)
elseif(FFT_BENCH_GPU_BACKEND STREQUAL "ROCM")
	set(FFT_BENCH_ROCM ON)
else()
	message(FATAL_ERROR "Invalid GPU backend option")
endif()


# add command line parser
include(FetchContent)
if(FFT_BENCH_BUNDLED_CLI11)
  FetchContent_Declare(
    cli11
    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.6.1.tar.gz
    URL_MD5 e22053ecf400a7205a1bd3029d001ac6
  )
  FetchContent_MakeAvailable(cli11)
else()
  find_package(CLI11 CONFIG REQUIRED)
endif()
list(APPEND FFT_BENCH_EXTERNAL_LIBS CLI11::CLI11)


if(FFT_BENCH_CUDA)
	find_package(CUDAToolkit REQUIRED)
	list(APPEND FFT_BENCH_EXTERNAL_LIBS CUDA::cudart CUDA::cufft)
endif()

if(FFT_BENCH_ROCM)
	find_package(hip CONFIG REQUIRED)
	find_package(hipfft CONFIG REQUIRED)
	list(APPEND FFT_BENCH_EXTERNAL_LIBS hip::hipfft hip::host)
endif()

enable_testing()

add_executable(fft_bench fft_bench.cpp)
target_link_libraries(fft_bench PRIVATE ${FFT_BENCH_EXTERNAL_LIBS})

if(FFT_BENCH_CUDA)
  target_compile_options(fft_bench PRIVATE -DFFT_BENCH_CUDA)
endif()
if(FFT_BENCH_ROCM)
  target_compile_options(fft_bench PRIVATE -DFFT_BENCH_ROCM)
endif()

add_test(NAME fft COMMAND fft_bench -n 128 128 -p double -s 5 -b 1)
