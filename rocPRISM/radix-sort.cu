/*
 * AMD-GPU benchmarks
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Radix-sort performance test
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <iostream>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "timing.cuh"

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

    auto radixSort = [&]()
    {
        thrust::sort_by_key(keys.begin(), keys.end(), ordering.begin());
    };

    radixSort(); // warmup
    float t_radixSort = timeGpu(radixSort);

    std::size_t numBytesMoved = 2lu * numKeys * (sizeof(KeyType) + sizeof(ValueType));
    std::cout << "radix sort time for " << numKeys << " key-value pairs: " << t_radixSort / 1000 << " s"
        << ", bandwidth: " << float(numBytesMoved) / t_radixSort / 1000 << " MiB/s" << std::endl;

    return 0;
}
