/*
 * GPU benchmarks
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Scan performance test with memory tracking
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/scan.h>

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
    thrust::device_vector<ValueType> scannedValues(numValues);

    tracking_mr memory_tracker;
    thrust::mr::allocator<ValueType, tracking_mr> alloc(&memory_tracker);

    auto scanNormal = [&]()
    {
#ifdef __HIP__
        thrust::exclusive_scan(thrust::hip::par, values.begin(), values.end(), scannedValues.begin());
#else
        thrust::exclusive_scan(thrust::cuda::par, values.begin(), values.end(), scannedValues.begin());
#endif
    };

    auto scanTracked = [&]()
    {
#ifdef __HIP__
        thrust::exclusive_scan(thrust::hip::par(alloc), values.begin(), values.end(), scannedValues.begin());
#else
        thrust::exclusive_scan(thrust::cuda::par(alloc), values.begin(), values.end(), scannedValues.begin());
#endif
    };

    scanNormal(); // warmup
    float timeScan = timeGpu(scanNormal);
    thrust::device_vector<ValueType> scannedValuesNormal = scannedValues;

    scanTracked(); // warmup
    memory_tracker.reset();
    float timeScanTracked = timeGpu(scanTracked);

    memory_tracker.print_stats();
    std::size_t numBytesMoved = 2lu * numValues * sizeof(ValueType);
    std::printf("exclusive scan normal time for %zu values: %f s, bandwidth: %f MiB/s\n",
                numValues, timeScan / 1000, float(numBytesMoved) / timeScan / 1000);
    std::printf("exclusive scan with memory tracking time for %zu values: %f s, bandwidth: %f MiB/s\n",
                numValues, timeScanTracked / 1000, float(numBytesMoved) / timeScanTracked / 1000);

    if (power <= 25)
    {
        std::vector<ValueType> hostScan(numValues);
        std::exclusive_scan(hostValues.begin(), hostValues.end(), hostScan.begin(), ValueType(0));
        std::printf("GPU matches CPU: %s\n", (hostScan.back() == scannedValues.back() ? "PASS" : "FAIL"));
        std::printf("GPU normal matches GPU with tracked memory: %s\n", (scannedValuesNormal.back() == scannedValues.back() ? "PASS" : "FAIL"));
    }

    return 0;
}
