#pragma once

#if defined(FFT_BENCH_CUDA)
# include <cuda_runtime_api.h>
# define GPU_PREFIX(val) cuda##val

#else
# include <hip/hip_runtime_api.h>
# define GPU_PREFIX(val) hip##val
#endif

#include <stdexcept>
#include <utility>

namespace gpu { namespace api {

        using StatusType = GPU_PREFIX(Error_t);
        using StreamType = GPU_PREFIX(Stream_t);
        using EventType = GPU_PREFIX(Event_t);

        namespace status {
            // error / return values
            constexpr StatusType Success = GPU_PREFIX(Success);
            constexpr StatusType ErrorMemoryAllocation = GPU_PREFIX(ErrorMemoryAllocation);
            constexpr StatusType ErrorLaunchOutOfResources = GPU_PREFIX(ErrorLaunchOutOfResources);
            constexpr StatusType ErrorInvalidValue = GPU_PREFIX(ErrorInvalidValue);
            constexpr StatusType ErrorInvalidResourceHandle =
                GPU_PREFIX(ErrorInvalidResourceHandle);
            constexpr StatusType ErrorInvalidDevice = GPU_PREFIX(ErrorInvalidDevice);
            constexpr StatusType ErrorInvalidMemcpyDirection =
                GPU_PREFIX(ErrorInvalidMemcpyDirection);
            constexpr StatusType ErrorInvalidDevicePointer = GPU_PREFIX(ErrorInvalidDevicePointer);
            constexpr StatusType ErrorInitializationError = GPU_PREFIX(ErrorInitializationError);
            constexpr StatusType ErrorNoDevice = GPU_PREFIX(ErrorNoDevice);
            constexpr StatusType ErrorNotReady = GPU_PREFIX(ErrorNotReady);
            constexpr StatusType ErrorUnknown = GPU_PREFIX(ErrorUnknown);
            constexpr StatusType ErrorPeerAccessNotEnabled = GPU_PREFIX(ErrorPeerAccessNotEnabled);
            constexpr StatusType ErrorPeerAccessAlreadyEnabled =
                GPU_PREFIX(ErrorPeerAccessAlreadyEnabled);
            constexpr StatusType ErrorHostMemoryAlreadyRegistered =
                GPU_PREFIX(ErrorHostMemoryAlreadyRegistered);
            constexpr StatusType ErrorHostMemoryNotRegistered =
                GPU_PREFIX(ErrorHostMemoryNotRegistered);
            constexpr StatusType ErrorUnsupportedLimit = GPU_PREFIX(ErrorUnsupportedLimit);
        }    // namespace status

        // flags to pass to GPU API
        namespace flag {
            constexpr auto StreamDefault = GPU_PREFIX(StreamDefault);
            constexpr auto StreamNonBlocking = GPU_PREFIX(StreamNonBlocking);

            constexpr auto MemcpyDefault = GPU_PREFIX(MemcpyDefault);
            constexpr auto MemcpyHostToDevice = GPU_PREFIX(MemcpyHostToDevice);
            constexpr auto MemcpyDeviceToHost = GPU_PREFIX(MemcpyDeviceToHost);
            constexpr auto MemcpyDeviceToDevice = GPU_PREFIX(MemcpyDeviceToDevice);

            constexpr auto EventDefault = GPU_PREFIX(EventDefault);
            constexpr auto EventBlockingSync = GPU_PREFIX(EventBlockingSync);
            constexpr auto EventDisableTiming = GPU_PREFIX(EventDisableTiming);
            constexpr auto EventInterprocess = GPU_PREFIX(EventInterprocess);
        }    // namespace flag

        // ==================================
        // Error check functions
        // ==================================
        inline auto get_error_string(StatusType error) -> char const*
        {
            return GPU_PREFIX(GetErrorString)(error);
        }

        inline auto check_status(StatusType error) -> void
        {
            if (error != status::Success) { throw std::runtime_error(get_error_string(error)); }
        }

        // ===================================================
        // Forwarding functions of to GPU API with error check
        // ===================================================
        template <typename... ARGS>
        inline auto host_register(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(HostRegister)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto host_unregister(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(HostUnregister)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto stream_create_with_flags(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(StreamCreateWithFlags)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto stream_wait_event(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(StreamWaitEvent)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto event_create(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(EventCreate)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto event_create_with_flags(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(EventCreateWithFlags)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto event_record(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(EventRecord)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto event_synchronize(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(EventSynchronize)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto event_elapsed_time(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(EventElapsedTime)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto malloc(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(Malloc)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto memcpy(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(Memcpy)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto memcpy_2d(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(Memcpy2D)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto memcpy_async(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(MemcpyAsync)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto memcpy_2d_async(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(Memcpy2DAsync)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto get_device(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(GetDevice)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto set_device(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(SetDevice)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto get_device_count(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(GetDeviceCount)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto stream_synchronize(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(StreamSynchronize)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto memset_async(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(MemsetAsync)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto memset_2d_async(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(Memset2DAsync)(std::forward<ARGS>(args)...));
        }

        template <typename... ARGS>
        inline auto get_device_properties(ARGS&&... args) -> void
        {
            check_status(GPU_PREFIX(GetDeviceProperties)(std::forward<ARGS>(args)...));
        }

        inline auto device_synchronize() -> void { check_status(GPU_PREFIX(DeviceSynchronize)()); }

        inline auto check_last_error() -> void { check_status(GPU_PREFIX(GetLastError)()); }

        // =====================================================
        // Forwarding functions of to GPU API with status return
        // =====================================================

        inline auto get_last_error() -> StatusType { return GPU_PREFIX(GetLastError)(); }

        template <typename... ARGS>
        inline auto free(ARGS&&... args) -> StatusType
        {
            return GPU_PREFIX(Free)(std::forward<ARGS>(args)...);
        }

        template <typename... ARGS>
        inline auto stream_destroy(ARGS&&... args) -> StatusType
        {
            return GPU_PREFIX(StreamDestroy)(std::forward<ARGS>(args)...);
        }

        template <typename... ARGS>
        inline auto event_destroy(ARGS&&... args) -> StatusType
        {
            return GPU_PREFIX(EventDestroy)(std::forward<ARGS>(args)...);
        }

}}    // namespace gpu::api

#undef GPU_PREFIX
