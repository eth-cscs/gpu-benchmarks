/*
 * GPU benchmarks
 *
 * Copyright (c) 2025 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Reduction performance test with memory tracking
 */

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "../common/timing.cuh"
#include "memory_tracking_allocator.cuh"

int main(int argc, char** argv)
{
    using ValueType = uint64_t;

    int power             = argc > 1 ? std::stoi(argv[1]) : 25;
    std::size_t numValues = 1lu << power;

    std::vector<ValueType> hostValues(numValues);
    {
        std::mt19937 gen;
        std::uniform_int_distribution<ValueType> dist(0, std::numeric_limits<uint32_t>::max());
        std::generate(hostValues.begin(), hostValues.end(), [&](){ return dist(gen); });
    }

    thrust::device_vector<ValueType> values = hostValues;

    tracking_mr memory_tracker;
    thrust::mr::allocator<ValueType, tracking_mr> alloc(&memory_tracker);

    auto reduceNormal = [&]()
    {
#ifdef __HIP__
        return thrust::reduce(thrust::hip::par, values.begin(), values.end());
#else
        return thrust::reduce(thrust::cuda::par, values.begin(), values.end());
#endif
    };

    auto reduceTracked = [&]()
    {
#ifdef __HIP__
        return thrust::reduce(thrust::hip::par(alloc), values.begin(), values.end());
#else
        return thrust::reduce(thrust::cuda::par(alloc), values.begin(), values.end());
#endif
    };

    auto deviceResult = reduceNormal();
    float timeReduce = timeGpu(reduceNormal);    // to compare with memory tracking time

    auto deviceResultTracked = reduceTracked();
    memory_tracker.reset();
    float timeReduceTracked = timeGpu(reduceTracked);

    memory_tracker.print_stats();
    std::size_t numBytesMoved = numValues * sizeof(ValueType);
    std::printf("reduction normal time for %zu values: %f s, bandwidth: %f MiB/s\n",
                numValues, timeReduce / 1000, float(numBytesMoved) / timeReduce / 1000);
    std::printf("reduction with memory tracking time for %zu values: %f s, bandwidth: %f MiB/s\n",
            numValues, timeReduceTracked / 1000, float(numBytesMoved) / timeReduceTracked / 1000);

    if (power <= 25)
    {
        auto hostResult = std::accumulate(hostValues.begin(), hostValues.end(), ValueType(0));
        std::printf("CPU matches GPU: %s\n", (deviceResult == hostResult ? "PASS" : "FAIL"));
        std::printf("GPU normal matches GPU with tracked memory: %s\n", (deviceResult == deviceResultTracked ? "PASS" : "FAIL"));
    }

    return 0;
}
