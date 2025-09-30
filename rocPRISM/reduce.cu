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
#include <numeric>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "timing.cuh"

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

    thrust::device_vector<ValueType>   values = hostValues;

    auto reduce = [&]()
    {
        return thrust::reduce(values.begin(), values.end());
    };

    auto deviceResult = reduce(); // warmup
    float t_reduce    = timeGpu(reduce);

    std::size_t numBytesMoved = numValues * sizeof(ValueType);
    std::cout << "reduction time for " << numValues << " values: " << t_reduce / 1000 << " s"
        << ", bandwidth: " << float(numBytesMoved) / t_reduce / 1000 << " MiB/s" << std::endl;

    if (power <= 25)
    {
        auto hostResult = std::accumulate(hostValues.begin(), hostValues.end(), ValueType(0));
        std::cout << "GPU matches CPU: " << (deviceResult == hostResult ? "PASS" : "FAIL") << std::endl;
    }

    return 0;
}
