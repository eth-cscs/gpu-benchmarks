/*
 * GPU benchmarks
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Radix-sort performance test with memory tracking
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "../common/timing.cuh"
#include "memory_tracking_allocator.cuh"

int main(int argc, char** argv)
{
    using KeyType = uint64_t;
    using ValueType = uint32_t;

    int power = argc > 1 ? std::stoi(argv[1]) : 25;
    std::size_t numKeys = 1lu << power;

    std::vector<KeyType> hostKeys(numKeys);
    {
        std::mt19937 gen;
        std::uniform_int_distribution<KeyType> dist(0, std::numeric_limits<KeyType>::max());
        std::generate(hostKeys.begin(), hostKeys.end(), [&]() { return dist(gen); });
    }

    tracking_mr mem_tracker;        // Input memory
    tracking_mr tmp_mem_tracker;    // Temporary memory

    tracking_mr::vector<KeyType> keys(
        hostKeys.begin(), hostKeys.end(), tracking_mr::allocator<KeyType>(&mem_tracker));
    tracking_mr::vector<ValueType> ordering(
        numKeys, 0, tracking_mr::allocator<ValueType>(&mem_tracker));
    thrust::sequence(ordering.begin(), ordering.end(), 0);

    tracking_mr::allocator<KeyType> tmp_alloc(&tmp_mem_tracker);

    auto radixSortNormal = [&]() {
#ifdef __HIP__
        thrust::sort_by_key(thrust::hip::par, keys.begin(), keys.end(), ordering.begin());
#else
        thrust::sort_by_key(thrust::cuda::par, keys.begin(), keys.end(), ordering.begin());
#endif
    };

    auto radixSortTracked = [&]() {
#ifdef __HIP__
        thrust::sort_by_key(
            thrust::hip::par(tmp_alloc), keys.begin(), keys.end(), ordering.begin());
#else
        thrust::sort_by_key(
            thrust::cuda::par(tmp_alloc), keys.begin(), keys.end(), ordering.begin());
#endif
    };

    radixSortNormal();                                 // warmup
    float timeRadixSort = timeGpu(radixSortNormal);    // to compare with memory tracking time

    // Re-initialize keys for tracked version
    thrust::copy(hostKeys.begin(), hostKeys.end(), keys.begin());
    thrust::sequence(ordering.begin(), ordering.end(), 0);

    radixSortTracked();    // warmup
    tmp_mem_tracker.reset();
    float timeRadixSortTracked = timeGpu(radixSortTracked);

    // time is measured in ms
    float time_s = timeRadixSort / 1000;
    mem_tracker.print_stats<false>();
    tmp_mem_tracker.print_stats<true>();
    std::size_t numBytesMoved = 2lu * numKeys * (sizeof(KeyType) + sizeof(ValueType));
    std::printf("radix sort normal time for %zu key-value pairs: %f s, bandwidth: %f MiB/s\n",
        numKeys, time_s, float(numBytesMoved) / time_s / (1024 * 1024));
    time_s = timeRadixSortTracked / 1000;
    std::printf(
        "radix sort with memory tracking time for %zu key-value pairs: %f s, bandwidth: %f MiB/s\n",
        numKeys, time_s, float(numBytesMoved) / time_s / (1024 * 1024));

    return 0;
}
