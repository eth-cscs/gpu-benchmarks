/*
 * GPU benchmarks
 *
 * Copyright (c) 2025 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

#pragma once

#include <cstdio>
#include <atomic>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/mr/allocator.h>

class tracking_mr final : public thrust::mr::memory_resource<>
{
    std::atomic<std::size_t> current_bytes{0};
    std::atomic<std::size_t> peak_bytes{0};
    std::atomic<std::size_t> total_bytes{0};
    std::atomic<std::size_t> num_allocs{0};

public:
    // default thrust alignment is used
    virtual void* do_allocate(std::size_t bytes, std::size_t) override
    {
        thrust::device_ptr<uint8_t> ptr = thrust::device_malloc<uint8_t>(bytes);
        if (!ptr.get()) throw std::bad_alloc();

        std::size_t current = current_bytes.fetch_add(bytes) + bytes;
        total_bytes.fetch_add(bytes);
        num_allocs.fetch_add(1);

        std::size_t peak = peak_bytes.load();
        while (current > peak && !peak_bytes.compare_exchange_weak(peak, current));

        return static_cast<void*>(ptr.get());
    }

    virtual void do_deallocate(void* ptr, std::size_t bytes, std::size_t) override
    {
        current_bytes.fetch_sub(bytes);
        thrust::device_free(thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(ptr)));
    }

    void reset()
    {
        current_bytes.store(0);
        peak_bytes.store(0);
        total_bytes.store(0);
        num_allocs.store(0);
    }

    void print_stats() const
    {
        std::printf("Memory statistics:\n");
        std::printf("  Total allocated: %f MiB\n", total_bytes.load() / (1024.0 * 1024.0));
        std::printf("  Peak allocated: %f MiB\n", peak_bytes.load() / (1024.0 * 1024.0));
        std::printf("  Number of allocations: %zu\n", num_allocs.load());
    }
};
