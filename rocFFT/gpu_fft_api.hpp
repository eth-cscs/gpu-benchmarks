#pragma once

#if defined(FFT_BENCH_CUDA)
#include <cufft.h>
#define GPU_FFT_PREFIX(val) cufft##val

#else

#if __has_include(<hipfft/hipfft.h>)
#include <hipfft/hipfft.h>
#else
#include <hipfft.h>
#endif

#define GPU_FFT_PREFIX(val) hipfft##val
#endif

#include <stdexcept>
#include <utility>

namespace gpu {
namespace fft {

// ==================================
// Types
// ==================================
using ResultType = GPU_FFT_PREFIX(Result);
using HandleType = GPU_FFT_PREFIX(Handle);
using ComplexFloatType = GPU_FFT_PREFIX(Complex);
using ComplexDoubleType = GPU_FFT_PREFIX(DoubleComplex);

// Complex type selector
template <typename T> struct ComplexType;

template <> struct ComplexType<double> {
  using type = ComplexDoubleType;
};

template <> struct ComplexType<float> {
  using type = ComplexFloatType;
};

// ==================================
// Transform types
// ==================================
namespace TransformDirection {
#ifdef FFT_BENCH_CUDA
constexpr auto Forward = CUFFT_FORWARD;
constexpr auto Backward = CUFFT_INVERSE;
#else
constexpr auto Forward = HIPFFT_FORWARD;
constexpr auto Backward = HIPFFT_BACKWARD;
#endif
} // namespace TransformDirection

// ==================================
// Transform types
// ==================================
namespace TransformType {
#ifdef FFT_BENCH_CUDA
constexpr auto R2C = CUFFT_R2C;
constexpr auto C2R = CUFFT_C2R;
constexpr auto C2C = CUFFT_C2C;
constexpr auto D2Z = CUFFT_D2Z;
constexpr auto Z2D = CUFFT_Z2D;
constexpr auto Z2Z = CUFFT_Z2Z;
#else
constexpr auto R2C = HIPFFT_R2C;
constexpr auto C2R = HIPFFT_C2R;
constexpr auto C2C = HIPFFT_C2C;
constexpr auto D2Z = HIPFFT_D2Z;
constexpr auto Z2D = HIPFFT_Z2D;
constexpr auto Z2Z = HIPFFT_Z2Z;
#endif

// Transform type selector
template <typename T> struct ComplexToComplex;

template <> struct ComplexToComplex<double> {
  constexpr static auto value = Z2Z;
};

template <> struct ComplexToComplex<float> {
  constexpr static auto value = C2C;
};

// Transform type selector
template <typename T> struct RealToComplex;

template <> struct RealToComplex<double> {
  constexpr static auto value = D2Z;
};

template <> struct RealToComplex<float> {
  constexpr static auto value = R2C;
};

// Transform type selector
template <typename T> struct ComplexToReal;

template <> struct ComplexToReal<double> {
  constexpr static auto value = Z2D;
};

template <> struct ComplexToReal<float> {
  constexpr static auto value = C2R;
};
} // namespace TransformType

// ==================================
// Result values
// ==================================
namespace result {
#ifdef FFT_BENCH_CUDA
constexpr auto Success = CUFFT_SUCCESS;
#else
constexpr auto Success = HIPFFT_SUCCESS;
#endif
} // namespace result

// ==================================
// Error check functions
// ==================================
inline auto check_result(ResultType error) -> void {
  if (error != result::Success) {
    throw std::runtime_error("GPU FFT error");
  }
}

// ==================================
// Execution function overload
// ==================================
inline auto execute(HandleType &plan, const ComplexDoubleType *iData,
                    double *oData) -> void {
  check_result(GPU_FFT_PREFIX(ExecZ2D)(
      plan, const_cast<ComplexDoubleType *>(iData), oData));
}

inline auto execute(HandleType &plan, const ComplexFloatType *iData,
                    float *oData) -> void {
  check_result(GPU_FFT_PREFIX(ExecC2R)(
      plan, const_cast<ComplexFloatType *>(iData), oData));
}

inline auto execute(HandleType &plan, const double *iData,
                    ComplexDoubleType *oData) -> void {
  check_result(
      GPU_FFT_PREFIX(ExecD2Z)(plan, const_cast<double *>(iData), oData));
}

inline auto execute(HandleType &plan, const float *iData,
                    ComplexFloatType *oData) -> void {
  check_result(
      GPU_FFT_PREFIX(ExecR2C)(plan, const_cast<float *>(iData), oData));
}

inline auto execute(HandleType &plan, const ComplexDoubleType *iData,
                    ComplexDoubleType *oData, int direction) -> void {
  check_result(GPU_FFT_PREFIX(ExecZ2Z)(
      plan, const_cast<ComplexDoubleType *>(iData), oData, direction));
}

inline auto execute(HandleType &plan, const ComplexFloatType *iData,
                    ComplexFloatType *oData, int direction) -> void {
  check_result(GPU_FFT_PREFIX(ExecC2C)(
      plan, const_cast<ComplexFloatType *>(iData), oData, direction));
}

// ==================================
// Forwarding functions of to GPU API
// ==================================

template <typename... ARGS> inline auto create(ARGS &&...args) -> void {
  check_result(GPU_FFT_PREFIX(Create)(std::forward<ARGS>(args)...));
}

template <typename... ARGS> inline auto make_plan_many(ARGS &&...args) -> void {
  check_result(GPU_FFT_PREFIX(MakePlanMany)(std::forward<ARGS>(args)...));
}

template <typename... ARGS> inline auto plan_many(ARGS &&...args) -> void {
  check_result(GPU_FFT_PREFIX(PlanMany)(std::forward<ARGS>(args)...));
}

template <typename... ARGS> inline auto plan_1d(ARGS &&...args) -> void {
  check_result(GPU_FFT_PREFIX(Plan1d)(std::forward<ARGS>(args)...));
}

template <typename... ARGS> inline auto plan_2d(ARGS &&...args) -> void {
  check_result(GPU_FFT_PREFIX(Plan2d)(std::forward<ARGS>(args)...));
}

template <typename... ARGS> inline auto plan_3d(ARGS &&...args) -> void {
  check_result(GPU_FFT_PREFIX(Plan3d)(std::forward<ARGS>(args)...));
}

template <typename... ARGS> inline auto set_work_area(ARGS &&...args) -> void {
  check_result(GPU_FFT_PREFIX(SetWorkArea)(std::forward<ARGS>(args)...));
}

template <typename... ARGS> inline auto destroy(ARGS &&...args) -> ResultType {
  return GPU_FFT_PREFIX(Destroy)(std::forward<ARGS>(args)...);
}

template <typename... ARGS> inline auto set_stream(ARGS &&...args) -> void {
  check_result(GPU_FFT_PREFIX(SetStream)(std::forward<ARGS>(args)...));
}

template <typename... ARGS>
inline auto set_auto_allocation(ARGS &&...args) -> void {
  check_result(GPU_FFT_PREFIX(SetAutoAllocation)(std::forward<ARGS>(args)...));
}

} // namespace fft
} // namespace gpu

#undef GPU_FFT_PREFIX
