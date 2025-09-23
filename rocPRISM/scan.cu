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
#include <thrust/scan.h>

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
    thrust::device_vector<ValueType> scannedValues(numValues);

    auto scan = [&]()
    {
        thrust::exclusive_scan(values.begin(), values.end(), scannedValues.begin());
    };

    scan(); // warmup
    float t_scan = timeGpu(scan);

    std::size_t numBytesMoved = 2lu * numValues * sizeof(ValueType);
    std::cout << "exclusive scan time for " << numValues << " values: " << t_scan / 1000 << " s"
        << ", bandwidth: " << float(numBytesMoved) / t_scan / 1000 << " MiB/s" << std::endl;

    if (power <= 25)
    {
        std::vector<ValueType> hostScan(numValues);
        std::exclusive_scan(hostValues.begin(), hostValues.end(), hostScan.begin(), ValueType(0));
        std::cout << "GPU matches CPU: " << (hostScan.back() == scannedValues.back() ? "PASS" : "FAIL") << std::endl;
    }

    return 0;
}
