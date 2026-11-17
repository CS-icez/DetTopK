// DetTopK: deterministic batched top-k selection on the GPU. This is the public interface.
//
// All four functions below have the same signature and produce bit-identical output.
//
// Task: `scores` is a row-major matrix of `rows` x `width` scores. For each row r only the first
// valid_len[r] columns count. Select the k positions with the highest scores among them and write them
// to row r of `out` (`rows` x `k`).
//
// The output is uniquely determined by the input:
//   1. The selected positions are exactly the k highest-scoring ones.
//   2. When several positions tie at the k-th score and not all of them fit, smaller positions win.
//   3. Selected positions are written in ascending order of position. A row whose valid length is
//      below k selects all of its positions and pads the rest of its output row with -1.
//
// Special values:
//   - +0 and -0 compare equal; +inf and -inf are ordinary values.
//   - NaN: if a row with valid length greater than k has a NaN inside its valid range, the call returns
//     kNanFound and the contents of `out` are unspecified. Rows with valid length at most k are not
//     checked (all of their positions are selected anyway). Columns beyond the valid length are never read.
//
// Limitations of the GPU functions (topk, topk_host) in this version:
//   - They run on the current CUDA device and on the default stream; the caller cannot pass a stream.
//   - They are not thread-safe: the implementation keeps global state (launch parameters and buffers on
//     the GPU that are allocated at the first call and reused), so calls must not overlap.
//   - They are synchronous: the computation on the GPU has finished when a call returns.
//   - Scores are fp32 and positions and valid lengths are int32; the number of rows must fit in int32.
//   - The kernel is tuned for the NVIDIA H100 (sm_90) only.
#pragma once

#include <cstdint>

namespace dettopk {

enum class Status {
    kOk = 0,
    kNanFound,     // a NaN inside the valid range of some row
    kBadArgument,  // k < 1, or a valid length outside [0, width]
    kGpuError,     // a CUDA call failed, e.g. no usable GPU on the machine
};

// The GPU implementation (CUDA), one thread block per row. scores, valid_len and out are device
// pointers: all three arrays are already on the GPU. nan_row is a host pointer and may be null; when it
// is not and the call returns kNanFound, it receives the smallest index among the rows that contain a NaN.
// Available only when linked with the CUDA object (gpu/topk_gpu.cu).
Status topk(const float* scores, int64_t rows, int64_t width,
            const int32_t* valid_len, int32_t k, int32_t* out,
            int64_t* nan_row = nullptr);

// The same with the three arrays in host memory: copies the input to the GPU, calls topk and copies
// the output back. A convenience for tests and small inputs; the copies cost far more than the selection.
Status topk_host(const float* scores, int64_t rows, int64_t width,
                 const int32_t* valid_len, int32_t k, int32_t* out,
                 int64_t* nan_row = nullptr);

// Serial CPU implementation: the correctness reference and the baseline for speedup.
Status topk_serial(const float* scores, int64_t rows, int64_t width,
                   const int32_t* valid_len, int32_t k, int32_t* out,
                   int64_t* nan_row = nullptr);

// Multithreaded CPU implementation (OpenMP, parallel over rows): the multicore baseline. OpenMP decides
// the number of threads.
Status topk_omp(const float* scores, int64_t rows, int64_t width,
                const int32_t* valid_len, int32_t k, int32_t* out,
                int64_t* nan_row = nullptr);

}  // namespace dettopk
