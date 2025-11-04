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
    using KeyType   = uint64_t;
    using ValueType = uint32_t;

    int power           = argc > 1 ? std::stoi(argv[1]) : 25;
    std::size_t numKeys = 1lu << power;

    std::vector<KeyType> hostKeys(numKeys);
    {
        std::mt19937 gen;
        std::uniform_int_distribution<KeyType> dist(0, std::numeric_limits<KeyType>::max());
        std::generate(hostKeys.begin(), hostKeys.end(), [&](){ return dist(gen); });
    }

    thrust::device_vector<KeyType> keys = hostKeys;
    thrust::device_vector<ValueType> ordering(numKeys);
    thrust::sequence(ordering.begin(), ordering.end(), 0);

    tracking_mr memory_tracker;
    thrust::mr::allocator<KeyType, tracking_mr> alloc(&memory_tracker);

    auto radixSortNormal = [&]()
    {
#ifdef __HIP__
        thrust::sort_by_key(thrust::hip::par, keys.begin(), keys.end(), ordering.begin());
#else
        thrust::sort_by_key(thrust::cuda::par, keys.begin(), keys.end(), ordering.begin());
#endif
    };

    auto radixSortTracked = [&]()
    {
#ifdef __HIP__
        thrust::sort_by_key(thrust::hip::par(alloc), keys.begin(), keys.end(), ordering.begin());
#else
        thrust::sort_by_key(thrust::cuda::par(alloc), keys.begin(), keys.end(), ordering.begin());
#endif
    };

    radixSortNormal(); // warmup
    float timeRadixSort = timeGpu(radixSortNormal);    // to compare with memory tracking time

    // Re-initialize keys for tracked version
    keys = hostKeys;
    thrust::sequence(ordering.begin(), ordering.end(), 0);

    radixSortTracked(); // warmup
    memory_tracker.reset();
    float timeRadixSortTracked = timeGpu(radixSortTracked);

    memory_tracker.print_stats();
    std::size_t numBytesMoved = 2lu * numKeys * (sizeof(KeyType) + sizeof(ValueType));
    std::printf("radix sort normal time for %zu key-value pairs: %f s, bandwidth: %f MiB/s\n",
                numKeys, timeRadixSort / 1000, float(numBytesMoved) / timeRadixSort / 1000);
    std::printf("radix sort with memory tracking time for %zu key-value pairs: %f s, bandwidth: %f MiB/s\n",
                numKeys, timeRadixSortTracked / 1000, float(numBytesMoved) / timeRadixSortTracked / 1000);

    return 0;
}
