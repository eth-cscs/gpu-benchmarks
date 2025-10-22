/*
 * GPU benchmarks
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GPU timing utility
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <chrono>
#include "./cuda_runtime.hpp"

inline void checkErr(cudaError_t err, const char* filename, int lineno, const char* funcName)
{
    if (err != cudaSuccess)
    {
        const char* errName = cudaGetErrorName(err);
        const char* errStr  = cudaGetErrorString(err);
        fprintf(stderr, "CUDA Error at %s:%d. Function %s returned err %d: %s - %s\n", filename, lineno, funcName, err,
                errName, errStr);
        exit(EXIT_FAILURE);
    }
}

#define checkGpuErrors(errcode) checkErr((errcode), __FILE__, __LINE__, #errcode)

//! @brief time a generic unary function
template<class F>
float timeGpu(F&& f)
{
    cudaEvent_t start, stop;
    checkGpuErrors(cudaEventCreate(&start));
    checkGpuErrors(cudaEventCreate(&stop));

    checkGpuErrors(cudaEventRecord(start, cudaStreamDefault));

    f();

    checkGpuErrors(cudaEventRecord(stop, cudaStreamDefault));
    checkGpuErrors(cudaEventSynchronize(stop));

    float t0;
    checkGpuErrors(cudaEventElapsedTime(&t0, start, stop));

    checkGpuErrors(cudaEventDestroy(start));
    checkGpuErrors(cudaEventDestroy(stop));

    return t0;
}

template<class F>
float timeCpu(F&& f)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float>(t1 - t0).count();
}
