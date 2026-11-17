// GPU implementation (CUDA): one thread block per row.
//
// How a thread block processes a row is all in block_row.h (the host emulation uses the same code).
// This file only holds what puts it on the GPU: the kernel shells, the launch, and fetching the status
// of the whole call.
//
// Memory: the input is read in place from global memory and the output is written to global memory. A
// thread block's working data (histogram, candidate buffer, the vote bits of collect, per-segment
// counts) all lives in shared memory. Besides the output, global memory holds only two integers for the
// status of the whole call plus, when there are few rows and a row is split across blocks, two counters
// and a candidate area per row. One error flag lives in page-locked host memory and is written directly
// from the GPU (zero-copy).
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <vector>

#include "block_row.h"
#include "dettopk_internal.h"

namespace dettopk {

namespace {

constexpr int kWarp = 32;  // threads per warp on the GPU

// What one thread of a thread block is on the GPU: CUDA provides its index and the thread count, the
// barrier is __syncthreads, the atomic add is atomicAdd, the atomic or is atomicOr, the warp vote is
// __ballot_sync and the warp shuffles are __shfl_up_sync and __shfl_sync.
struct CudaThread {
    __device__ int tid() const { return static_cast<int>(threadIdx.x); }
    __device__ int threads() const { return static_cast<int>(blockDim.x); }
    __device__ void sync() const { __syncthreads(); }
    __device__ uint32_t fetch_add(uint32_t* ptr, uint32_t v) const { return atomicAdd(ptr, v); }
    __device__ uint32_t fetch_or(uint32_t* ptr, uint32_t v) const { return atomicOr(ptr, v); }
    __device__ int group_size() const { return kWarp; }
    __device__ uint32_t ballot(bool pred) const { return __ballot_sync(0xFFFFFFFFu, pred); }
    __device__ uint32_t shfl_up(uint32_t v, int d) const {
        return __shfl_up_sync(0xFFFFFFFFu, v, static_cast<unsigned>(d));
    }
    __device__ uint32_t shfl(uint32_t v, int src) const { return __shfl_sync(0xFFFFFFFFu, v, src); }
    // For rows split across blocks, which hand candidates over through global memory (SplitCtx in
    // block_row.h): a memory fence, and a load that goes through L2 only.
    __device__ void fence() const { __threadfence(); }
    __device__ uint32_t load_l2(const uint32_t* ptr) const { return __ldcg(ptr); }
#ifdef DETTOPK_PROFILE
    __device__ uint64_t clock() const { return static_cast<uint64_t>(clock64()); }
#endif
};

// The status of the whole call is kept in two places:
//   - two integers on the GPU (g_status, in global memory), updated with atomic operations by thread
//     blocks that hit an error; both use 0 for "no error";
//   - one error flag (g_err_host) in page-locked memory mapped into the GPU: the host writes 0 before
//     launching the kernel, a thread block that hits an error writes 1, and the host reads it directly
//     after waiting for the GPU, with no copy (zero-copy).
// Normally only the flag is looked at and the call returns if it is 0; only when it is 1 is the status
// on the GPU copied back and zeroed again. So a call without errors is just "launch and wait", with no
// zeroing on the GPU and no copy back every time (which used to add about 4 microseconds to the floor
// of a call).
// The row index needs a minimum and therefore atomic operations, but on some platforms atomic
// operations on mapped memory are split into a separate read and write (CUDA programming guide,
// "Mapped Memory"), so the row index is not kept in mapped memory. The flag only ever takes the value
// 1, so several blocks writing it at once cannot corrupt it.
enum {
    kNanCode = 0,    // smallest row index with a NaN, stored as INT32_MAX - row and maximized; 0 means none
    kBadLength = 1,  // whether some row has an invalid valid length
    kStatusWords = 2,
};

// How a block's shared memory is laid out (the same for both kernels; topk_kernel has it expanded
// inline, see there): one contiguous array `words` whose size is given at launch. The first work_words
// entries are the work area, laid out in one of two ways depending on the path the row takes (see "Path
// choice" in block_row.h):
//   the other two paths: histogram and candidate buffer during selection, vote bits during collect
//     (work_words is at least histogram plus candidate buffer; the vote bits use at most bits_words
//     entries, no more than work_words, and 0 means votes are not recorded);
//   merged path: histogram, candidates (key and position interleaved, 2 x pair_capacity entries) and
//     vote bits (pair_bits_words entries) one after another, all in use at once (pair_capacity 0
//     disables the merged path). The sampled path uses these three areas as well (sample_min_len 0
//     disables it).
// After the work area come the two per-warp counts and one entry per thread.
__device__ inline gpu::BlockShared make_block_shared(uint32_t* words, gpu::BlockScalars* scalars, int32_t cand_capacity,
                                                     int32_t work_words, int32_t bits_words, int32_t pair_capacity,
                                                     int32_t pair_bits_words, int32_t sample_min_len, int32_t sample_r,
                                                     int32_t sample_rmax) {
    gpu::BlockShared sh;
    sh.hist = words;
    sh.cand = sh.hist + detail::kHighBuckets;
    sh.bits = words;
    sh.bits_capacity = static_cast<uint32_t>(bits_words);
    sh.pair = sh.hist + detail::kHighBuckets;
    sh.pair_bits = sh.pair + 2 * pair_capacity;
    sh.pair_capacity = static_cast<uint32_t>(pair_capacity);
    sh.pair_bits_capacity = static_cast<uint32_t>(pair_bits_words);
    sh.seg_g = words + work_words;
    sh.seg_e = sh.seg_g + blockDim.x / kWarp;
    sh.part = sh.seg_e + blockDim.x / kWarp;
    sh.s = scalars;
    sh.cand_capacity = static_cast<uint32_t>(cand_capacity);
    sh.sample_min_len = static_cast<uint32_t>(sample_min_len);
    sh.sample_r_req = static_cast<uint32_t>(sample_r);
    sh.sample_rmax_req = static_cast<uint32_t>(sample_rmax);
    return sh;
}

// The kernel: thread block r processes row r.
__global__ void topk_kernel(const float* scores, int64_t width, const int32_t* valid_len, int32_t k,
                            int32_t* out, int32_t cand_capacity, int32_t work_words, int32_t bits_words,
                            int32_t pair_capacity, int32_t pair_bits_words, int32_t sample_min_len,
                            int32_t sample_r, int32_t sample_rmax, bool select_only, int32_t* status,
                            volatile int32_t* err
#ifdef DETTOPK_PROFILE
                            , uint64_t* prof
#endif
) {
    // The shared memory layout is that of make_block_shared. It is expanded here rather than called: with
    // the call the compiler schedules instructions slightly differently, and this kernel's machine code
    // would no longer be what it was before the split kernel was added (making measurements
    // incomparable). Change both places together.
    extern __shared__ uint32_t words[];
    __shared__ gpu::BlockScalars scalars;
    gpu::BlockShared sh;
    sh.hist = words;
    sh.cand = sh.hist + detail::kHighBuckets;
    sh.bits = words;
    sh.bits_capacity = static_cast<uint32_t>(bits_words);
    sh.pair = sh.hist + detail::kHighBuckets;
    sh.pair_bits = sh.pair + 2 * pair_capacity;
    sh.pair_capacity = static_cast<uint32_t>(pair_capacity);
    sh.pair_bits_capacity = static_cast<uint32_t>(pair_bits_words);
    sh.seg_g = words + work_words;
    sh.seg_e = sh.seg_g + blockDim.x / kWarp;
    sh.part = sh.seg_e + blockDim.x / kWarp;
    sh.s = &scalars;
    sh.cand_capacity = static_cast<uint32_t>(cand_capacity);
    sh.sample_min_len = static_cast<uint32_t>(sample_min_len);
    sh.sample_r_req = static_cast<uint32_t>(sample_r);
    sh.sample_rmax_req = static_cast<uint32_t>(sample_rmax);

    const int64_t r = blockIdx.x;
    const CudaThread p{};
#ifdef DETTOPK_PROFILE
    const gpu::RowResult res = gpu::block_process_row(p, scores + r * width, valid_len[r], width, k, out + r * k, sh,
                                                      select_only, prof + gpu::kProfSlots * r);
#else
    const gpu::RowResult res =
        gpu::block_process_row(p, scores + r * width, valid_len[r], width, k, out + r * k, sh, select_only);
#endif

    // When several rows have a NaN, which thread block finishes first is not fixed; taking the minimum
    // makes the reported row index the same as the serial implementation's.
    if (threadIdx.x == 0) {
        if (res == gpu::RowResult::kNan) atomicMax(&status[kNanCode], INT32_MAX - static_cast<int32_t>(r));
        if (res == gpu::RowResult::kBadLength) atomicMax(&status[kBadLength], 1);
        if (res == gpu::RowResult::kNan || res == gpu::RowResult::kBadLength) *err = 1;
    }
}

// The kernel for rows split across `parts` blocks (SplitCtx in block_row.h): block i processes stretch
// i % parts of row i / parts, so consecutive blocks belong to the same row (CUDA places consecutive
// blocks on different SMs). The shared memory layout is the same as topk_kernel's.
// cand is the rows' candidate areas in global memory, cand_words words per row. counters holds two
// counters per row (counters[2r] and counters[2r + 1]), all 0 when the call starts and reset by the
// last block of the row (a row with a NaN has no last block, and the launcher zeroes them after the
// call; see topk).
// The algorithm is the split instantiation of block_process_row (Split is true), compiled separately
// from the one topk_kernel uses.
// The profiling build does not use this kernel (profiling has one set of slots per row, and with
// splitting several blocks would write to the same set).
__global__ void topk_split_kernel(const float* scores, int64_t width, const int32_t* valid_len, int32_t k,
                                  int32_t* out, int32_t cand_capacity, int32_t work_words, int32_t bits_words,
                                  int32_t pair_capacity, int32_t pair_bits_words, int32_t sample_min_len,
                                  int32_t sample_r, int32_t sample_rmax, bool select_only, int32_t* status,
                                  volatile int32_t* err, int32_t parts, uint32_t* cand, int64_t cand_words,
                                  uint32_t* counters) {
    extern __shared__ uint32_t words[];
    __shared__ gpu::BlockScalars scalars;
    const gpu::BlockShared sh = make_block_shared(words, &scalars, cand_capacity, work_words, bits_words, pair_capacity,
                                                  pair_bits_words, sample_min_len, sample_r, sample_rmax);

    const int64_t r = blockIdx.x / parts;
    const gpu::SplitCtx sp{static_cast<int>(blockIdx.x % parts), parts, cand + r * cand_words,
                           counters + 2 * r};
    const CudaThread p{};
    const gpu::RowResult res = gpu::block_process_row<CudaThread, true>(
        p, scores + r * width, valid_len[r], width, k, out + r * k, sh, select_only, nullptr, sp);
    if (threadIdx.x == 0) {
        if (res == gpu::RowResult::kNan) atomicMax(&status[kNanCode], INT32_MAX - static_cast<int32_t>(r));
        if (res == gpu::RowResult::kBadLength) atomicMax(&status[kBadLength], 1);
        if (res == gpu::RowResult::kNan || res == gpu::RowResult::kBadLength) *err = 1;
    }
}

// Read-only baseline (GpuConfig::read_passes, for benchmarking): thread block r reads the valid range
// of row r `passes` times, the way the three passes of selection read it (for_each_score), with a
// barrier between passes. Dynamic shared memory is unused but requested in the same amount as for the
// top-k kernel, so that the number of resident blocks per SM is the same. The bit patterns of the
// scores read are summed and written to slot 0, which keeps the compiler from eliminating the loads and
// gives the benchmark something to check.
__global__ void read_kernel(const float* scores, int64_t width, const int32_t* valid_len, int32_t k, int32_t* out,
                            int passes) {
    __shared__ uint32_t total;
    const int64_t r = blockIdx.x;
    const int32_t len = valid_len[r];
    const int tid = static_cast<int>(threadIdx.x);
    const int T = static_cast<int>(blockDim.x);
    int32_t* o = out + r * k;
    if (len < 0 || len > width) return;
    if (len <= k) {  // as in the top-k kernel: do not read the scores; write 0, 1, ... followed by -1 padding
        for (int32_t i = tid; i < k; i += T) o[i] = i < len ? i : -1;
        return;
    }
    if (tid == 0) total = 0;
    const float* row = scores + r * width;
    uint32_t acc = 0;
    for (int pass = 0; pass < passes; ++pass) {
        __syncthreads();
        gpu::for_each_score(tid, T, row, len, [&](float x) { acc += float_bits(x); });
    }
    atomicAdd(&total, acc);
    __syncthreads();
    if (tid == 0) o[0] = static_cast<int32_t>(total);
}

// Shared memory per thread block: the work_words entries used in turn by selection and collect, the two
// per-warp counts, and one entry per thread (in the order the kernel lays them out).
size_t shared_bytes(int threads, int work_words) {
    return (static_cast<size_t>(work_words) + 2 * static_cast<size_t>(threads / kWarp) +
            static_cast<size_t>(threads)) * sizeof(uint32_t);
}
int hist_cand_words(int cand_capacity) { return static_cast<int>(detail::kHighBuckets) + cand_capacity; }

// How many such thread blocks can be resident on one SM, as answered by CUDA's occupancy API: it
// considers this GPU's registers, threads, block slots and shared memory, and whichever runs out first
// sets the number. Returns -1 on error.
int blocks_per_sm(int threads, size_t bytes) {
    int n = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, topk_kernel, threads, bytes) != cudaSuccess) return -1;
    return n;
}

// How large the candidate buffer is when chosen automatically.
//   - The wanted capacity is 4096: on a standard-normal row 163840 wide, the picked bucket holds about
//     2700 keys, which 2048 cannot take.
//   - But it must not lower the number of resident thread blocks per SM: halve from 4096 until the
//     number of resident blocks equals that without a candidate buffer. On an H100 with 128 threads per
//     block an SM holds 16 blocks and 4096 would exhaust shared memory first, so it drops to 1024;
//     every other thread count gets 4096.
// The maximum work area (work_max) follows the same rule of not lowering the number of resident thread
// blocks per SM: ask CUDA how much shared memory a block may use with that many blocks resident,
// subtract the other areas, and all the rest is available (never less than histogram plus candidate
// buffer). This bounds the vote bits of collect, and the merged path's candidates come out of it too;
// each call gets what its row width needs (see plan_work).
// Computed once per thread count and remembered.
constexpr int kCandWanted = 4096;
GpuLaunchInfo g_auto[1024 / kWarp + 1];
bool g_auto_ready[1024 / kWarp + 1] = {};
GpuLaunchInfo g_manual[1024 / kWarp + 1];  // for manual capacities: one entry per (thread count, the two capacities)
int g_manual_key[1024 / kWarp + 1][2];
bool g_manual_ready[1024 / kWarp + 1] = {};
bool g_shared_raised = false;

// A kernel may use up to 48 KB of shared memory per block by default; more has to be declared to CUDA
// first. Declaring this GPU's maximum only permits it and takes no extra memory. The read-only baseline
// kernel is launched with the same amount of shared memory and needs the declaration as well.
template <class K>
bool raise_shared_limit_of(K kernel, int optin) {
    cudaFuncAttributes attr{};
    return cudaFuncGetAttributes(&attr, kernel) == cudaSuccess &&
           cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                optin - static_cast<int>(attr.sharedSizeBytes)) == cudaSuccess;
}

bool raise_shared_limit() {
    if (g_shared_raised) return true;
    int dev = 0, optin = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess ||
        !raise_shared_limit_of(topk_kernel, optin) || !raise_shared_limit_of(read_kernel, optin) ||
        !raise_shared_limit_of(topk_split_kernel, optin)) {
        return false;
    }
    g_shared_raised = true;
    return true;
}

bool plan_launch(int threads, int cand_request, int bits_request, GpuLaunchInfo* info) {
    const int slot = threads / kWarp;
    const bool automatic = cand_request < 0 && bits_request < 0;
    if (automatic && g_auto_ready[slot]) {
        *info = g_auto[slot];
        return true;
    }
    if (!automatic && g_manual_ready[slot] && g_manual_key[slot][0] == cand_request &&
        g_manual_key[slot][1] == bits_request) {
        *info = g_manual[slot];
        return true;
    }
    if (!raise_shared_limit()) return false;
    GpuLaunchInfo li{};
    li.blocks_per_sm_plain = blocks_per_sm(threads, shared_bytes(threads, hist_cand_words(0)));
    if (li.blocks_per_sm_plain <= 0) return false;
    if (cand_request >= 0) {
        li.cand_capacity = cand_request;
    } else {
        li.cand_capacity = kCandWanted;
        while (li.cand_capacity > 0 &&
               blocks_per_sm(threads, shared_bytes(threads, hist_cand_words(li.cand_capacity))) <
                   li.blocks_per_sm_plain) {
            li.cand_capacity /= 2;
        }
    }
    li.blocks_per_sm = blocks_per_sm(threads, shared_bytes(threads, hist_cand_words(li.cand_capacity)));
    if (li.blocks_per_sm <= 0) return false;
    {
        // The maximum work area: the dynamic shared memory a block may use with blocks_per_sm blocks
        // resident, minus the per-warp and per-thread areas.
        size_t avail = 0;
        if (cudaOccupancyAvailableDynamicSMemPerBlock(&avail, topk_kernel, li.blocks_per_sm, threads) !=
            cudaSuccess) {
            return false;
        }
        int words = static_cast<int>(avail / sizeof(uint32_t)) - 2 * (threads / kWarp) - threads;
        // Check again with the number of resident blocks and shrink until it is unchanged (never below
        // histogram plus candidate buffer).
        while (words > hist_cand_words(li.cand_capacity) &&
               blocks_per_sm(threads, shared_bytes(threads, words)) < li.blocks_per_sm) {
            words -= 256;
        }
        li.work_max = words > hist_cand_words(li.cand_capacity) ? words : hist_cand_words(li.cand_capacity);
    }
    li.bits_capacity = bits_request >= 0 ? bits_request : li.work_max;
    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, topk_kernel) != cudaSuccess) return false;
    li.regs = attr.numRegs;
    li.local_bytes = static_cast<int>(attr.localSizeBytes);
    if (automatic) {
        g_auto[slot] = li;
        g_auto_ready[slot] = true;
    } else {
        g_manual[slot] = li;
        g_manual_key[slot][0] = cand_request;
        g_manual_key[slot][1] = bits_request;
        g_manual_ready[slot] = true;
    }
    *info = li;
    return true;
}

// How the work area of this call is divided.
//   The other two paths: at least histogram plus candidate buffer, and as many vote bits as this row
//     width needs, up to the maximum. When the row width needs more than the maximum, only histogram
//     plus candidate buffer is given, and rows whose vote bits do not fit read the row once more at
//     write-out (rows with a short valid length, which do fit, still record their votes).
//   Merged path: the histogram, the candidates' keys and positions and the vote bits for this row
//     width must all fit at once. The candidate capacity is what was given manually, or otherwise the
//     maximum work area minus the histogram and the vote bits, all of it (not capped at 4096). The
//     merged path is not taken when votes are not recorded.
// The work area is the larger of the two. Without the merged path only what is needed is requested:
// the less shared memory is used, the more the SM has left for the L1 cache.
struct WorkPlan {
    int work_words;       // words of the work area
    int bits_words;       // maximum words of vote bits for the other two paths
    int pair_capacity;    // number of candidates the merged path holds; 0 disables the merged path
    int pair_bits_words;  // words of the merged path's vote bits
};
WorkPlan plan_work(const GpuLaunchInfo& li, int threads, int64_t width, int pair_request) {
    WorkPlan w{};
    const int base = hist_cand_words(li.cand_capacity);
    const int64_t need = gpu::collect_bits_words(width, threads, kWarp);
    w.work_words = need <= base || need > li.bits_capacity ? base : static_cast<int>(need);
    if (need <= li.bits_capacity) {
        const int64_t pairs = pair_request >= 0 ? pair_request
                                                : (li.work_max - static_cast<int64_t>(detail::kHighBuckets) - need) / 2;
        if (pairs > 0) {
            w.pair_capacity = static_cast<int>(pairs);
            w.pair_bits_words = static_cast<int>(need);
            const int64_t merged = detail::kHighBuckets + 2 * pairs + need;
            if (merged > w.work_words) w.work_words = static_cast<int>(merged);
        }
    }
    // Maximum words of vote bits: no more than the work area and no more than the given capacity (a
    // manual 0 records no votes even if the work area has room).
    w.bits_words = w.work_words < li.bits_capacity ? w.work_words : li.bits_capacity;
    return w;
}

GpuConfig g_config;
int32_t* g_status = nullptr;       // the status on the GPU, allocated at the first call and kept
// For rows split across blocks (split_parts): two counters per row (g_split_count), allocated once for
// kSplitMaxRows rows. The candidate area in global memory takes 2 x the sampled path's candidate
// capacity words per row; it is reallocated when too small and then kept.
constexpr int64_t kSplitMaxRows = 1024;
uint32_t* g_split_cand = nullptr;
int64_t g_split_cand_words = 0;
int g_sms = 0;  // number of SMs on the GPU, asked from CUDA at the first call
// The error flag (see kNanCode above): its address on the host and its address on the GPU (from
// cudaHostGetDevicePointer). The status on the GPU (g_status) and the split counters (g_split_count)
// are 0 at the start of every call: zeroed once at allocation and again after a call with an error.
volatile int32_t* g_err_host = nullptr;
int32_t* g_err_dev = nullptr;
uint32_t* g_split_count = nullptr;
#ifdef DETTOPK_PROFILE
// Profiling build: kProfSlots numbers per row (the cycles of each step plus the sampled-path outcome;
// see block_row.h), one array on the GPU, accumulated on the host.
static_assert(gpu::kProfSteps == kGpuProfSteps, "the step count in dettopk_internal.h must match ProfStep in block_row.h");
static_assert(gpu::kSampleFull + 1 == kGpuSampleOutcomes,
              "the outcome count in dettopk_internal.h must match SampleOutcome in block_row.h");
uint64_t* g_prof = nullptr;
int64_t g_prof_rows = 0;
GpuProfile g_prof_total{};
#endif

// How many blocks each row of this call is split across (1 means no splitting; see SplitCtx in
// block_row.h). Rows are split only when there are so few that one block per row cannot fill the GPU:
// min(maximum, number of SMs / rows) blocks per row, and no splitting below 2. An H100 has 132 SMs, so
// rows are split when there are at most 66. The automatic maximum is 16: on the decode-latency
// workload, splitting across 8, 16 or 32 blocks is equally fast (1 row drops from 55 to 38
// microseconds), and on the prefill sweep only 16 was measured.
// Splitting happens on the sampled path only, so there is none when the sampled path is off (merged
// path off, or minimum length 0), nor for the read-only baseline or the profiling build.
constexpr int kSplitMaxAuto = 16;
int split_parts(int64_t rows, const WorkPlan& wp, int32_t sample_min_len) {
#ifdef DETTOPK_PROFILE
    (void)rows, (void)wp, (void)sample_min_len, (void)kSplitMaxAuto;
    return 1;
#else
    const int max = g_config.split_max < 0 ? kSplitMaxAuto : g_config.split_max;
    if (max < 2 || g_config.read_passes > 0 || wp.pair_capacity == 0 || sample_min_len == 0 ||
        rows > kSplitMaxRows) {
        return 1;
    }
    const int64_t g = g_sms / rows;
    return static_cast<int>(g < max ? g : max);
#endif
}

// cudaMalloc does not guarantee that 0 bytes can be allocated; ask for at least 4.
template <class T>
bool device_alloc(T** ptr, size_t bytes) {
    return cudaMalloc(ptr, bytes ? bytes : 4) == cudaSuccess;
}

// How the thread count is chosen when none is given:
//   - with very few rows (at most 256) the GPU is not full and more threads make a single row faster:
//     use 1024;
//   - otherwise by row width: 256 up to 8192, 512 up to 24576, 1024 beyond.
// This rule was measured for the current code on an H100, not derived from first principles: over the
// calls of the prefill sweep the fastest thread count changes with the row width, and in a separate
// experiment that varied only the number of rows it did not depend on the number of rows at a given
// row width. When the bottleneck of the code moves, the best rule moves with it. Only row widths that
// are multiples of 8192 were measured.
constexpr int64_t kFewRows = 256;
int auto_threads(int64_t rows, int64_t width) {
    if (rows <= kFewRows) return 1024;
    if (width <= 8192) return 256;
    if (width <= 24576) return 512;
    return 1024;
}

}  // namespace

void set_gpu_config(const GpuConfig& config) { g_config = config; }

#ifdef DETTOPK_PROFILE
GpuProfile gpu_profile_take() {
    const GpuProfile p = g_prof_total;
    g_prof_total = {};
    return p;
}
#endif

Status gpu_launch_info(int threads, GpuLaunchInfo* info) {
    if (threads < kWarp || threads > 1024 || threads % kWarp != 0 || !info) return Status::kBadArgument;
    return plan_launch(threads, g_config.cand_capacity, g_config.bits_capacity, info) ? Status::kOk
                                                                                       : Status::kGpuError;
}

Status topk(const float* scores, int64_t rows, int64_t width, const int32_t* valid_len, int32_t k,
            int32_t* out, int64_t* nan_row) {
    const int threads = g_config.threads > 0 ? g_config.threads : auto_threads(rows, width);
    if (k < 1 || rows < 0 || width < 0 || rows > INT32_MAX) return Status::kBadArgument;
    // The thread count must be a multiple of 32: every warp is then full, as CUDA requires for warp votes.
    if (threads < kWarp || threads > 1024 || threads % kWarp != 0) return Status::kBadArgument;
    if (rows == 0) return Status::kOk;

    if (!g_status) {
        if (!device_alloc(&g_status, kStatusWords * sizeof(int32_t)) ||
            !device_alloc(&g_split_count, 2 * kSplitMaxRows * sizeof(uint32_t)) ||
            cudaMemset(g_status, 0, kStatusWords * sizeof(int32_t)) != cudaSuccess ||
            cudaMemset(g_split_count, 0, 2 * kSplitMaxRows * sizeof(uint32_t)) != cudaSuccess) {
            g_status = nullptr;
            return Status::kGpuError;
        }
    }
    if (!g_err_host) {
        int32_t* h = nullptr;
        if (cudaHostAlloc(&h, sizeof(int32_t), cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer(&g_err_dev, h, 0) != cudaSuccess) {
            return Status::kGpuError;
        }
        g_err_host = h;
    }
    GpuLaunchInfo li;
    if (!plan_launch(threads, g_config.cand_capacity, g_config.bits_capacity, &li)) return Status::kGpuError;
    // A manual capacity beyond the GPU's limit makes the kernel launch fail, which is reported below.
    const WorkPlan wp = plan_work(li, threads, width, g_config.pair_capacity);
    // Minimum valid length for the sampled path (0 disables it). Its candidates and vote bits live in the
    // merged path's two areas, so it is off whenever the merged path is (the launch below passes 0 then).
    const int32_t sample_min_len =
        g_config.sample_min_len < 0 ? kSampleMinLenAuto : static_cast<int32_t>(g_config.sample_min_len);
    const size_t bytes = shared_bytes(threads, wp.work_words);
    // Blocks per row. With splitting, the candidate area in global memory must hold every row of this call.
    if (g_sms == 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
            return Status::kGpuError;
        }
    }
    const int parts = split_parts(rows, wp, sample_min_len);
    const int64_t cand_words = 2 * static_cast<int64_t>(wp.pair_capacity);
    if (parts > 1 && rows * cand_words > g_split_cand_words) {
        cudaFree(g_split_cand);
        g_split_cand_words = 0;
        if (!device_alloc(&g_split_cand, static_cast<size_t>(rows * cand_words) * sizeof(uint32_t))) {
            g_split_cand = nullptr;
            return Status::kGpuError;
        }
        g_split_cand_words = rows * cand_words;
    }
    // Clear the error flag: the host writes it directly. The previous call waited for the GPU, so nothing
    // on the GPU is still writing it, and this write precedes the kernel launch, so the kernel sees it.
    *g_err_host = 0;

#ifdef DETTOPK_PROFILE
    if (rows > g_prof_rows) {
        cudaFree(g_prof);
        g_prof_rows = 0;
        if (!device_alloc(&g_prof, static_cast<size_t>(gpu::kProfSlots * rows) * sizeof(uint64_t))) return Status::kGpuError;
        g_prof_rows = rows;
    }
    if (cudaMemset(g_prof, 0, static_cast<size_t>(gpu::kProfSlots * rows) * sizeof(uint64_t)) != cudaSuccess) {
        return Status::kGpuError;
    }
#endif
    // The number of thread blocks equals the number of rows (times the blocks per row with splitting),
    // each with `threads` threads.
    if (g_config.read_passes > 0) {
        read_kernel<<<static_cast<unsigned>(rows), static_cast<unsigned>(threads), bytes>>>(
            scores, width, valid_len, k, out, g_config.read_passes);
    } else if (parts > 1) {
        topk_split_kernel<<<static_cast<unsigned>(rows * parts), static_cast<unsigned>(threads), bytes>>>(
            scores, width, valid_len, k, out, li.cand_capacity, wp.work_words, wp.bits_words, wp.pair_capacity,
            wp.pair_bits_words, sample_min_len, g_config.sample_r, g_config.sample_rmax, g_config.select_only,
            g_status, g_err_dev, parts, g_split_cand, cand_words, g_split_count);
    } else {
        topk_kernel<<<static_cast<unsigned>(rows), static_cast<unsigned>(threads), bytes>>>(
            scores, width, valid_len, k, out, li.cand_capacity, wp.work_words, wp.bits_words, wp.pair_capacity,
            wp.pair_bits_words, wp.pair_capacity > 0 ? sample_min_len : 0, g_config.sample_r, g_config.sample_rmax,
            g_config.select_only, g_status, g_err_dev
#ifdef DETTOPK_PROFILE
            , g_prof
#endif
        );
    }
    if (cudaGetLastError() != cudaSuccess) return Status::kGpuError;
#ifdef DETTOPK_PROFILE
    {
        std::vector<uint64_t> prof(static_cast<size_t>(gpu::kProfSlots * rows));
        if (cudaMemcpy(prof.data(), g_prof, prof.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return Status::kGpuError;
        }
        for (int64_t r = 0; r < rows; ++r) {
            const uint64_t* row = prof.data() + gpu::kProfSlots * r;
            // Rows that skipped selection (valid length at most k, or a NaN) have 0 in the slot of pass 1
            // (where the sampled path also records reading the row).
            if (row[gpu::kProfCount1] == 0) continue;
            for (int j = 0; j < gpu::kProfSteps; ++j) g_prof_total.step_cycles[j] += row[j];
            ++g_prof_total.rows;
            const uint64_t outcome = row[gpu::kProfOutcome];
            if (outcome < kGpuSampleOutcomes) ++g_prof_total.sample_rows[outcome];
        }
    }
#endif

    // Fetch the status: wait for the stream to finish, so that the computation on the GPU is over when
    // the function returns. After the wait the host can read the flag the GPU wrote to mapped memory
    // (CUDA programming guide, "Mapped Memory": host and device accesses to mapped memory must be
    // synchronized through streams or events; this is a stream synchronization).
    if (cudaStreamSynchronize(0) != cudaSuccess) return Status::kGpuError;
    if (*g_err_host == 0) return Status::kOk;
    // An error occurred: copy the status back from the GPU, then zero it and the split counters this
    // call used (a row with a NaN has no last block, so its counters were not reset). These are blocking
    // synchronous calls, made only on error.
    int32_t st[kStatusWords];
    if (cudaMemcpy(st, g_status, sizeof st, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemset(g_status, 0, sizeof st) != cudaSuccess ||
        (parts > 1 && cudaMemset(g_split_count, 0, static_cast<size_t>(2 * rows) * sizeof(uint32_t)) != cudaSuccess)) {
        return Status::kGpuError;
    }
    if (st[kBadLength]) return Status::kBadArgument;
    if (st[kNanCode] != 0) {
        if (nan_row) *nan_row = INT32_MAX - st[kNanCode];
        return Status::kNanFound;
    }
    return Status::kOk;
}

Status topk_host(const float* scores, int64_t rows, int64_t width, const int32_t* valid_len, int32_t k,
                 int32_t* out, int64_t* nan_row) {
    if (k < 1 || rows < 0 || width < 0) return Status::kBadArgument;
    if (rows == 0) return Status::kOk;
    const size_t scores_bytes = static_cast<size_t>(rows * width) * sizeof(float);
    const size_t len_bytes = static_cast<size_t>(rows) * sizeof(int32_t);
    const size_t out_bytes = static_cast<size_t>(rows * k) * sizeof(int32_t);

    float* d_scores = nullptr;
    int32_t* d_len = nullptr;
    int32_t* d_out = nullptr;
    Status st = Status::kGpuError;
    if (device_alloc(&d_scores, scores_bytes) && device_alloc(&d_len, len_bytes) &&
        device_alloc(&d_out, out_bytes) &&
        cudaMemcpy(d_scores, scores, scores_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(d_len, valid_len, len_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
        // With selection only, most slots are not written: first copy the caller's existing contents of
        // out to the GPU, so that the slots left unwritten come back unchanged.
        (!g_config.select_only || cudaMemcpy(d_out, out, out_bytes, cudaMemcpyHostToDevice) == cudaSuccess)) {
        st = topk(d_scores, rows, width, d_len, k, d_out, nan_row);
        if (st == Status::kOk && cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            st = Status::kGpuError;
        }
    }
    cudaFree(d_scores);
    cudaFree(d_len);
    cudaFree(d_out);
    return st;
}

}  // namespace dettopk
