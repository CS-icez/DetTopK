// An external baseline in the benchmark: NVIDIA CUB's batched top-k (cub::DeviceBatchedTopK::MaxPairs),
// required to be deterministic with ties going to the smaller position
// (cuda::execution::determinism::gpu_to_gpu + tie_break::prefer_smaller_index), the same semantics as
// this project. The output is unordered (output_ordering::unsorted): the order of the positions it
// returns is unspecified, so fetch sorts them by position on the host before the driver hashes them,
// and that step is not timed. The time measured is therefore only "selecting which k", one thing less
// than this project does (which writes the positions out in order).
//
// Present only in build/bench_topk_cub from make bench-cub, and it needs a separate download of CCCL
// (the library CUB is part of): the CCCL 3.0 shipped with CUDA 13.0 does not have this algorithm; CCCL
// 3.5.0 has it, but only non-deterministic and for segments that fit in one thread block; determinism
// and long segments need the main branch as of 2026-11-13 (commit c2eb63f is used here), and run only
// on sm_90 and above (Hopper).
// This wrapper was checked on an sm_89 GPU built as non-deterministic (CUB_FLAGS=-DDETTOPK_CUB_NONDET): every row
// that differed from the serial implementation differed only in which of the tied positions was taken.
// Usage: see bench-cub in the Makefile.
//
// CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT: without it, even a build for sm_90 only fails at compile
// time with "the policy selected the single-block backend, which cannot be deterministic" (seen on main
// at c2eb63f); CUB's own tests and benchmarks define it. With it the check moves to run time: when the
// GPU cannot be deterministic the call returns cudaErrorNotSupported (reported here as a GPU error)
// rather than silently switching to a non-deterministic method.
#define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT
#include <cub/device/device_batched_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/std/__execution/env.h>
#include <cuda/stream>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <memory>

#include "impl.h"
#include "dettopk.h"

namespace bench {

namespace {

constexpr int32_t kMaxK = 2048;  // k is 2048 everywhere in the benchmark

class CubImpl : public Impl {
public:
    CubImpl() { cudaStreamCreate(&stream_); }
    ~CubImpl() override {
        cudaFree(scores_.ptr);
        cudaFree(len_.ptr);
        cudaFree(keys_out_.ptr);
        cudaFree(out_.ptr);
        cudaFree(temp_.ptr);
        cudaStreamDestroy(stream_);
    }

    void load(const Input& in) override {
        in_ = in;
        const size_t scores_bytes = static_cast<size_t>(in.rows * in.width) * sizeof(float);
        const size_t len_bytes = static_cast<size_t>(in.rows) * sizeof(int32_t);
        out_bytes_ = static_cast<size_t>(in.rows * in.k) * sizeof(int32_t);
        ok_ = in.k >= 1 && in.k <= kMaxK && in.width <= 163840 && grow(scores_, scores_bytes) &&
              grow(len_, len_bytes) && grow(keys_out_, static_cast<size_t>(in.rows * in.k) * sizeof(float)) &&
              grow(out_, out_bytes_) &&
              cudaMemcpy(scores_.ptr, in.scores, scores_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(len_.ptr, in.valid_len, len_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemset(out_.ptr, 0x80, out_bytes_) == cudaSuccess;
        // Ask first whether the temporary storage is large enough and reallocate if not (untimed).
        // It may ask for 0 bytes, but a null pointer is taken as "only query the size" and nothing
        // runs, so allocate at least 4 bytes.
        size_t bytes = 0;
        ok_ = ok_ && report("query temporary storage", call(nullptr, bytes)) && grow(temp_, bytes > 4 ? bytes : 4);
    }

    dettopk::Status run() override {
        if (!ok_) return dettopk::Status::kGpuError;
        size_t bytes = temp_.bytes;
        if (!report("run", call(temp_.ptr, bytes)) || !report("wait for the GPU", cudaStreamSynchronize(stream_))) {
            return dettopk::Status::kGpuError;
        }
        return dettopk::Status::kOk;
    }

    // Fetch the positions of each row and sort them in ascending order; rows with valid length below k
    // are padded with -1 (the output convention of this project).
    void fetch() override {
        if (!ok_) return;
        ok_ = cudaMemcpy(in_.out, out_.ptr, out_bytes_, cudaMemcpyDeviceToHost) == cudaSuccess &&
              cudaMemset(out_.ptr, 0x80, out_bytes_) == cudaSuccess;
        if (!ok_) return;
        for (int64_t r = 0; r < in_.rows; ++r) {
            int32_t* row = in_.out + r * in_.k;
            const int32_t n = std::min(std::max(in_.valid_len[r], 0), in_.k);
            std::sort(row, row + n);
            std::fill(row + n, row + in_.k, -1);
        }
    }

private:
    struct Buffer {
        void* ptr = nullptr;
        size_t bytes = 0;
    };

    // On error, say on standard error which step failed and how (operation not supported, for example,
    // when the GPU cannot be deterministic).
    static bool report(const char* what, cudaError_t e) {
        if (e == cudaSuccess) return true;
        std::fprintf(stderr, "  cub: %s failed: %s\n", what, cudaGetErrorString(e));
        return false;
    }

    static bool grow(Buffer& b, size_t bytes) {
        if (bytes <= b.bytes) return true;
        cudaFree(b.ptr);
        b.ptr = nullptr;
        b.bytes = 0;
        if (cudaMalloc(&b.ptr, bytes ? bytes : 4) != cudaSuccess) return false;
        b.bytes = bytes;
        return true;
    }

    // The upper bound of the segment length must be given at compile time, and the tighter the better
    // (it decides which kernel is used and how much temporary storage is needed), hence a few tiers by
    // row width.
    cudaError_t call(void* temp, size_t& bytes) {
        if (in_.width <= 8192) return call_bounded<8192>(temp, bytes);
        if (in_.width <= 16384) return call_bounded<16384>(temp, bytes);
        if (in_.width <= 32768) return call_bounded<32768>(temp, bytes);
        if (in_.width <= 65536) return call_bounded<65536>(temp, bytes);
        if (in_.width <= 131072) return call_bounded<131072>(temp, bytes);
        return call_bounded<163840>(temp, bytes);
    }

    template <int32_t MaxLen>
    cudaError_t call_bounded(void* temp, size_t& bytes) {
        const float* scores = static_cast<const float*>(scores_.ptr);
        // Segment r starts at scores + r x width; the values are positions within the segment (counted
        // from 0 in each segment); the output has k slots per segment.
        auto keys_in = cuda::make_strided_iterator(cuda::make_counting_iterator(scores), in_.width);
        auto keys_out = cuda::make_strided_iterator(
            cuda::make_counting_iterator(static_cast<float*>(keys_out_.ptr)), static_cast<int64_t>(in_.k));
        auto values_in = cuda::make_constant_iterator(cuda::make_counting_iterator(int32_t{0}));
        auto values_out = cuda::make_strided_iterator(
            cuda::make_counting_iterator(static_cast<int32_t*>(out_.ptr)), static_cast<int64_t>(in_.k));
        auto env = cuda::std::execution::env{
#ifdef DETTOPK_CUB_NONDET
            // Only for checking this wrapper on GPUs below sm_90: not deterministic,
            // ties are broken arbitrarily, and only rows that fit in one thread block can run.
            cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                     cuda::execution::tie_break::unspecified,
                                     cuda::execution::output_ordering::unsorted),
#else
            cuda::execution::require(cuda::execution::determinism::gpu_to_gpu,
                                     cuda::execution::tie_break::prefer_smaller_index,
                                     cuda::execution::output_ordering::unsorted),
#endif
            cuda::stream_ref{stream_}};
        return cub::DeviceBatchedTopK::MaxPairs(
            temp, bytes, keys_in, keys_out, values_in, values_out,
            cuda::args::deferred_sequence{static_cast<const int32_t*>(len_.ptr), cuda::args::bounds<0, MaxLen>()},
            cuda::args::immediate{in_.k, cuda::args::bounds<1, kMaxK>()},
            cuda::args::immediate{static_cast<int64_t>(in_.rows)}, env);
    }

    Input in_{};
    Buffer scores_, len_, keys_out_, out_, temp_;
    size_t out_bytes_ = 0;
    cudaStream_t stream_ = nullptr;
    bool ok_ = false;
};

}  // namespace

std::unique_ptr<Impl> make_cub_impl() { return std::make_unique<CubImpl>(); }

}  // namespace bench
