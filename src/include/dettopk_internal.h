// Internal interface of DetTopK, for its own tests and benchmark: tuning parameters, launch information,
// the host emulation and profiling. It is not part of the public interface (dettopk.h) and may change.
#pragma once

#include <cstdint>

#include "dettopk.h"

namespace dettopk {

// ---- Parameters of the GPU implementation. Only available when built with CUDA (make gpu). ----
// They are global: set_gpu_config applies to the calls of topk and topk_host that follow.

struct GpuConfig {
    // Threads per thread block: a multiple of 32, at most 1024. 0 selects automatically: 1024 when there
    // are very few rows, otherwise by row width (the rule is in gpu/topk_gpu.cu).
    // Affects speed only, never the result.
    int threads = 0;
    // Number of keys the candidate buffer holds. Negative selects automatically (as close to 4096 as
    // possible without lowering the number of resident thread blocks per SM; the rule is in
    // gpu/topk_gpu.cu); 0 disables the candidate buffer. A row whose candidates do not fit reads the
    // whole row one more time. Affects speed only, never the result.
    int cand_capacity = -1;
    // Number of 32-bit words for the vote bits recorded during collect: two bits per position, so that
    // write-out does not have to read the row again. Negative selects automatically (as many as possible
    // without lowering the number of resident thread blocks per SM); 0 records nothing and write-out
    // reads the row again, as do rows whose vote bits do not fit. Affects speed only, never the result.
    int bits_capacity = -1;
    // Number of candidates the merged path holds (each takes two words: key and position). On the merged
    // path, pass 2 votes as it goes and collect does not read the row again; see gpu/block_row.h.
    // Negative selects automatically: take the largest work area that does not lower the number of
    // resident thread blocks per SM, subtract the histogram and the vote bits for this row width, and
    // give it all the rest. 0 disables the merged path. It is also disabled when vote bits are not
    // recorded (bits_capacity is 0, or the vote bits for this row width exceed it). Rows that do not fit
    // take one of the other two paths. Affects speed only, never the result.
    int pair_capacity = -1;
    // Sampled path (read 1/16 of the row to choose a cutoff, then read the whole row only once; see
    // gpu/block_row.h): rows whose valid length is at least this try the sampled path first. Negative
    // selects automatically (kSampleMinLenAuto); 0 disables it. The sampled path keeps its candidates
    // and vote bits in the merged path's two areas, so it is disabled whenever the merged path is
    // (pair_capacity is 0, or the vote bits do not fit). Rows that fail verification fall back to the
    // other three paths. Affects speed only, never the result.
    int sample_min_len = -1;
    // For tests: when nonzero, these two are used directly as the rank of the sample that becomes the
    // cutoff and as the maximum number of samples at or above the cutoff (to force the two fallbacks).
    // When 0, both are computed by sample_plan in block_row.h.
    int sample_r = 0;
    int sample_rmax = 0;
    // When a call has so few rows that one block per row cannot fill the GPU, the maximum number of
    // blocks one row is split across (on the sampled path each block reads one part of the row and the
    // last block to arrive finishes it; see SplitCtx in gpu/block_row.h). Negative selects automatically
    // (at most 16); 0 or 1 disables splitting. Affects speed only, never the result.
    int split_max = -1;
    // For timing the two phases separately: run selection only and skip collect. Time this and a full
    // run; the difference is the time of collect. The output is then incomplete: a row with valid length
    // greater than k has only its first two slots written (tau and the tie quota); see gpu/block_row.h.
    bool select_only = false;
    // Read-only baseline (for benchmarking; not a top-k): when positive, a kernel that only reads the
    // rows is launched instead, to see how fast this access pattern can read on the GPU. One thread
    // block per row, with the same thread count and dynamic shared memory as the top-k kernel (hence the
    // same number of resident blocks per SM). It reads the valid range read_passes times the way
    // selection reads a row, with a barrier between passes. A row with valid length greater than k
    // writes only slot 0: the sum of the bit patterns of the scores it read (accumulated over the
    // passes, wrapping at 32 bits). Rows with valid length at most k produce their normal output.
    int read_passes = 0;
};
// Minimum valid length at which the sampled path is tried when selected automatically. Measured on an
// H100 against 32768: 0.9% faster per layer on the prefill sweep, nearly all of it from the calls whose
// rows are 32768 wide (21% faster). Going down to 16384 gains nothing more: rows shorter than 24577 run
// in 512-thread blocks, whose smaller capacity makes sample_plan refuse the sampled path. In a
// simulation over full rows of real indexer scores from 21 layers, none of the 1344 rows newly admitted
// by lowering the value from 32768 fell back.
constexpr int kSampleMinLenAuto = 24576;
void set_gpu_config(const GpuConfig& config);

// Host emulation of the GPU implementation: `threads` real threads play one thread block and run the
// same algorithm code as the GPU. Needs no GPU; used only to test correctness (see gpu/topk_gpu_emu.cpp).
// group_size is the number of threads per warp (1 to 32; always 32 on the GPU); threads need not be a
// multiple of it. cand_capacity is the number of keys the candidate buffer holds (0 for none),
// bits_capacity the number of words for the vote bits of collect (0 to record none), pair_capacity the
// number of candidates the merged path holds (0 disables the merged path; its vote bits are likewise
// limited to bits_capacity words).
// When paths is not null, it counts which path each row took (accumulated, never reset), so that tests
// can confirm a case really reached the path it targets.
// sample holds the sampled-path parameters: GpuConfig's sample_min_len (here 0 disables it and there is
// no automatic value), sample_r and sample_rmax, plus the number of blocks a row is split across
// (split; no automatic value here) and the order in which those blocks run. With splitting, paths
// records only the path taken by the block that finished the row.
struct EmuPathCount {
    int64_t row = 0;     // row path: pass 3 reads the whole row
    int64_t cand = 0;    // candidate path: pass 3 reads the candidate buffer, which holds keys only
    int64_t merged = 0;  // merged path
    int64_t sampled = 0;       // sampled path, completed
    int64_t sample_many = 0;   // sampled path fell back: too many samples at or above the cutoff
    int64_t sample_high = 0;   // sampled path fell back: cutoff too high
    int64_t sample_full = 0;   // sampled path fell back: candidates do not fit
};
struct EmuSample {
    int min_len = 0;
    int r = 0;
    int rmax = 0;
    int split = 0;       // when above 1, a row is split across this many blocks (GpuConfig::split_max), run one after another
    int split_seed = 0;  // shuffles the order of the blocks, so that different blocks come last (see gpu/topk_gpu_emu.cpp)
};
Status topk_gpu_emu(const float* scores, int64_t rows, int64_t width,
                    const int32_t* valid_len, int32_t k, int32_t* out,
                    int64_t* nan_row, int threads, int group_size, int cand_capacity,
                    int bits_capacity, int pair_capacity, bool select_only = false,
                    EmuPathCount* paths = nullptr, EmuSample sample = {});

// How the kernel is launched for a given thread count: the candidate buffer capacity, the maximum vote
// bits for collect, the resident thread blocks per SM and the registers per thread. The benchmark writes
// it to the log, to confirm whether different thread counts have the same number of resident blocks.
struct GpuLaunchInfo {
    int cand_capacity;        // candidate buffer capacity actually used
    int bits_capacity;        // maximum words for collect's vote bits; a call gets what its row width needs (gpu/topk_gpu.cu)
    int work_max;             // maximum words of the work area; with the row width it gives the merged path's capacity
    int blocks_per_sm;        // resident blocks per SM with this capacity (from CUDA's occupancy API)
    int blocks_per_sm_plain;  // resident blocks per SM without a candidate buffer
    int regs;                 // registers per thread used by the kernel
    int local_bytes;          // bytes per thread spilled to local memory (when registers run out); should be 0
};
Status gpu_launch_info(int threads, GpuLaunchInfo* info);

#ifdef DETTOPK_PROFILE
// Profiling build only (make gpu-prof): the total cycles spent in each step by the rows that went
// through selection since the last take, and the number of those rows. The order of the steps is
// ProfStep in gpu/block_row.h. Taking resets the counters. Cycles are the SM clock read by thread 0 of
// each row. Copying the profile out is itself part of the time the benchmark measures, so times from
// the profiling build are not used as measurements.
// sample_rows counts those rows by sampled-path outcome (SampleOutcome in gpu/block_row.h: not tried,
// completed, and three kinds of fallback).
constexpr int kGpuProfSteps = 12;
constexpr int kGpuSampleOutcomes = 5;
struct GpuProfile {
    uint64_t step_cycles[kGpuProfSteps];
    uint64_t rows;
    uint64_t sample_rows[kGpuSampleOutcomes];
};
GpuProfile gpu_profile_take();
#endif

}  // namespace dettopk
