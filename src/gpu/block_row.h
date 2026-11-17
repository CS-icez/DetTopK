// How one thread block processes one row: the core of the GPU algorithm.
//
// How to read this code: block_process_row is code that every thread of the block executes from top to
// bottom. The threads run at the same time, and tid (which thread am I) decides which data each touches.
//   - A loop written for (i = tid; i < n; i += T) deals n items out round-robin over the T threads:
//     thread tid handles items tid, tid+T, tid+2T, ...
//   - if (tid == 0) means that only thread 0 does it and the others skip it.
//   - p.sync() is a barrier: all threads must reach it before any goes on. What was written to shared
//     memory before a barrier reads the same for every thread after it.
//   - p.fetch_add(ptr, v) is an atomic add: *ptr += v, returning the value before the add. Nothing is
//     lost when several threads do it at once.
//   - Warp: the threads of a block are grouped by index, p.group_size() per warp (32 on the GPU), and
//     the threads of a warp always execute the same instruction together.
//   - p.clock() reads a monotonic timer (the SM's cycle counter on the GPU). Used only by the profiling
//     build (compiled with DETTOPK_PROFILE).
//   - p.ballot(b) is a warp vote (__ballot_sync on the GPU): each thread of a warp answers yes or no and
//     every thread gets the same number back, whose bit j is the answer of thread j of the warp. The
//     threads of a warp must call it together. Used by collect and by pass 2 of the merged path.
//   - p.fetch_or(ptr, v) is an atomic or: *ptr |= v. Nothing is lost when several threads do it at once.
//     Used only by the vote fix-up.
//   - p.shfl_up(v, d) and p.shfl(v, src) are warp shuffles (__shfl_up_sync and __shfl_sync on the GPU):
//     each thread of a warp offers a number, and thread L of the warp gets back the number of thread
//     L - d (its own when L < d), or every thread gets the number of thread src. The threads of a warp
//     must call them together. Used by the pick and by the write-out of collect (prefix sums in a warp).
//   - p.fence() and p.load_l2(ptr) are used only when a row is split across blocks (see SplitCtx below).
//     fence is a memory fence (__threadfence on the GPU): whatever this thread wrote to global memory
//     before it is visible to other blocks by the time they observe an atomic operation this thread
//     performs after it. load_l2 reads global memory through L2 only (__ldcg on the GPU), bypassing a
//     possibly stale copy in L1.
//
// This file is used in two places: the kernel on the GPU (topk_gpu.cu) and the host emulation with real
// threads (topk_gpu_emu.cpp, for tests). The two differ only in who the threads are, how they wait for
// each other and how they add atomically. All of that is behind the template parameter P, and the
// algorithm itself is the same code.
//
// What lives in shared memory (one copy per thread block, invisible to other blocks):
//   hist    the histogram, 2048 entries, reused by the three passes
//   cand    the candidate buffer: pass 2 copies the keys that fall in the picked bucket here, and pass 3
//           reads only this instead of the row. The launcher sets its capacity (on the GPU, chosen so as
//           not to lower the number of resident thread blocks per SM; see topk_gpu.cu)
//   bits    the vote bits of collect: two bits per position, "key above tau" and "key equal to tau",
//           recorded during collect's first read of the row so that write-out does not read the row
//           again. It occupies the same memory as hist and cand: once selection is done, the histogram
//           and the candidate buffer are no longer needed. The launcher sets its capacity as well; a row
//           whose vote bits do not fit reads the row once more at write-out. See "Collect" below
//   pair, pair_bits   the two areas of the merged path, laid out after the histogram, in use at the same
//           time as hist and not overlapping it: the candidates (2 x pair_capacity entries; candidate i
//           has its key in entry 2i and its position in entry 2i + 1) and the vote bits
//           (pair_bits_capacity words). Key and position sit next to each other so that storing a
//           candidate in pass 2 needs a single base address and one register less (with separate areas
//           the compiler recomputes the base of the position area at every store). The sampled path
//           uses both areas too: pair first holds the keys of the samples (one per entry) and, once the
//           cutoff is chosen, the candidates. See "Path choice" and "Sampled path" below
//   seg_g, seg_e   one entry per warp: during collect they hold the counts of each segment (see
//           "Collect" below); the pick borrows seg_g for the per-warp sums; the count of "above" votes
//           from pass 2 of the merged path is parked in seg_e (which the pick does not borrow)
//   part    one entry per thread. It used to hold each thread's partial bucket sum during the pick; the
//           pick now works by warp shuffles and no longer needs it, and it only serves as the small
//           table of the sampled path's third pick. The layout is left unchanged, because changing it
//           would change the capacity of the candidate buffer
//   s       a few scalars: the picked bucket, the rank still to find and so on. Everyone reads them
//           after the barrier that follows their computation
#pragma once

#include <cmath>
#include <cstdint>

#include "key.h"
#include "radix.h"

// Profiling: present only when compiled with DETTOPK_PROFILE (make gpu-prof); a normal build contains none
// of these statements. A row is divided into the steps of ProfStep below, each ending in a barrier.
// Thread 0 reads the timer once after each barrier and adds the cycles the step took to prof[step]. It
// reads after the barrier, so what it records is the moment the whole block finished the step.
// The write-out step has no closing barrier of its own; the profiling build adds one at the end and the
// normal build has none.
#ifdef DETTOPK_PROFILE
#define DETTOPK_PROF(...) __VA_ARGS__
#else
#define DETTOPK_PROF(...)
#endif

namespace dettopk {
namespace gpu {

// The profiled steps, in the order a row performs them. With selection only, the last four are 0. The
// vote fix-up exists only on the merged and sampled paths and the vote only on the other two; a step
// that is not taken stays 0. The sampled path reuses the slots of later steps (see the comment of each
// step); when it falls back, the steps from pass 1 on keep accumulating into the same slots.
enum ProfStep {
    kProfSample,  // sampled path: clear the histogram, read and count the samples, clear the vote area
    kProfThresh,  // sampled path: choose the cutoff (one or two picks)
    kProfCount1,  // pass 1: clear the histogram, read and count the row. Sampled path: read the row, store candidates
    kProfPick1,   // the pick of pass 1 and the path choice. Sampled path: first pick over the candidates
    kProfCount2,  // pass 2: clear the histogram, read and count the row, copy candidates (the merged path also
                  // casts and records "above" votes). Sampled path: scan the candidates and count
    kProfPick2,
    kProfCount3,  // pass 3: read the candidate buffer (or the row, when the candidates did not fit) and count
    kProfPick3,
    kProfFix,     // merged and sampled paths: vote fix-up
    kProfVote,    // collect step (1): read the row, vote, record the votes (absent on the merged and sampled paths)
    kProfPrefix,  // collect step (2): prefix sum over the segments
    kProfWrite,   // collect step (3): write-out
    kProfSteps,
};
// Besides the cycles of each step, the profiling build records one more slot per row: the outcome of
// the sampled path (SampleOutcome).
constexpr int kProfOutcome = kProfSteps;
constexpr int kProfSlots = kProfSteps + 1;

// Which path a row takes; see "Path choice" in block_process_row. 0 means the row has not chosen one
// (valid length at most k, or a NaN).
enum RowPath : uint32_t {
    kPathNone = 0,
    kPathRow = 1,      // row path: the candidates do not fit, pass 3 reads the whole row
    kPathCand = 2,     // candidate path: pass 3 reads the candidate buffer, which holds keys only
    kPathMerged = 3,   // merged path: pass 2 votes as it goes, collect does not read the row again
    kPathSampled = 4,  // sampled path: choose a cutoff from samples first, read the whole row only once
};

// How the sampled path went. After any of the three fallbacks the row takes one of the other three paths.
enum SampleOutcome : uint32_t {
    kSampleNone = 0,  // sampled path not tried (row too short, or it does not fit)
    kSampleOk = 1,    // sampled path completed
    kSampleMany = 2,  // fell back: too many samples at or above the cutoff (mostly heavy ties); the row has not been read yet
    kSampleHigh = 3,  // fell back: cutoff too high, fewer than k keys at or above it
    kSampleFull = 4,  // fell back: more keys at or above the cutoff than the candidate capacity
};

struct BlockScalars {
    uint32_t nan_seen;  // how many threads saw a NaN in this row
    uint32_t need;      // the rank still to find inside the currently picked bucket; after the three passes, the tie quota
    uint32_t pick;      // the bucket picked by this pass
    uint32_t path;      // which path this row takes (RowPath)
    uint32_t cand_n;    // how many candidates have been stored
    uint32_t sample_r;      // sampled path: the rank of the sample taken as the cutoff (see sample_plan)
    uint32_t sample_rmax;   // sampled path: the maximum number of samples allowed at or above the cutoff
    uint32_t sample_outcome;  // how the sampled path went (SampleOutcome); only for tests and profiling
    uint32_t small_n;         // sampled path, third pick done by one warp: entries in the small table; then the count of keys equal to tau
};

struct BlockShared {
    uint32_t* hist;          // 2048 entries
    uint32_t* cand;          // cand_capacity entries
    uint32_t* bits;          // bits_capacity entries, sharing memory with hist and cand (used only after selection)
    uint32_t* pair;          // 2 x pair_capacity entries: the merged path's candidates, key i in entry 2i, position in entry 2i + 1
    uint32_t* pair_bits;     // pair_bits_capacity entries: the merged path's vote bits
    uint32_t* seg_g;         // one entry per warp
    uint32_t* seg_e;         // likewise
    uint32_t* part;          // one entry per thread; now only the small table of the sampled path's third pick
    BlockScalars* s;
    uint32_t cand_capacity;  // number of keys the candidate buffer holds; 0 means no candidate buffer
    uint32_t bits_capacity;  // 32-bit words for collect's vote bits; 0 records no votes and write-out reads the row again
    uint32_t pair_capacity;       // number of candidates the merged path holds; 0 disables the merged path
    uint32_t pair_bits_capacity;  // number of words for the merged path's vote bits
    uint32_t sample_min_len;      // rows at least this long try the sampled path (0: never); it uses the merged path's two areas
    uint32_t sample_r_req;        // when nonzero, used as given for the cutoff rank and the maximum count (tests force fallbacks);
    uint32_t sample_rmax_req;     // when 0, both come from the formulas in sample_plan
};

enum class RowResult {
    kOk,
    kNan,        // a NaN inside the valid range; the row has no output
    kBadLength,  // valid length outside [0, width]
    kSplitDone,  // row split across blocks: this block did its share, another block finishes the row (see SplitCtx)
};

// Number of 1 bits in x.
DETTOPK_HOST_DEVICE inline uint32_t popcount32(uint32_t x) {
#ifdef __CUDA_ARCH__
    return static_cast<uint32_t>(__popc(x));
#else
    return static_cast<uint32_t>(__builtin_popcount(x));
#endif
}

// Index of the lowest 1 bit of x. x must not be 0.
DETTOPK_HOST_DEVICE inline uint32_t lowest_bit(uint32_t x) {
#ifdef __CUDA_ARCH__
    return static_cast<uint32_t>(__ffs(static_cast<int>(x)) - 1);
#else
    return static_cast<uint32_t>(__builtin_ctz(x));
#endif
}

// Spreads the low 8 bits of x apart: bit l goes to bit 4l and all other bits are 0.
DETTOPK_HOST_DEVICE inline uint32_t spread_byte(uint32_t x) {
    x &= 0xFFu;
    x = (x | (x << 12)) & 0x000F000Fu;  // bits 0 to 3 stay, bits 4 to 7 move to bits 16 to 19
    x = (x | (x << 6)) & 0x03030303u;   // split each group of 4 bits into two halves 8 bits apart
    x = (x | (x << 3)) & 0x11111111u;   // split each pair of bits, 4 bits apart
    return x;
}

// Regroups the 4 vote words of one step by position and takes chunk m: bit 4l + j of the result is bit
// 8m + l of w_j (l = 0 to 7). Bit L of a vote word belongs to position j of thread L of the warp, i.e.
// position 4L + j of the step, so bit b of the result is position 32m + b of the step. See step (3) of
// collect below.
DETTOPK_HOST_DEVICE inline uint32_t regroup(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, int m) {
    const int s = 8 * m;
    return spread_byte(w0 >> s) | (spread_byte(w1 >> s) << 1) | (spread_byte(w2 >> s) << 2) |
           (spread_byte(w3 >> s) << 3);
}

// Segment length per warp in collect: the valid length divided evenly, rounded up to a multiple of 4
// (segment boundaries must fall on 16-byte alignment; see "Collect" below).
DETTOPK_HOST_DEVICE inline int64_t collect_seg(int64_t len, int groups) {
    const int64_t seg = (len + groups - 1) / groups;
    return (seg + 3) / 4 * 4;
}

// Number of words the vote bits of one row take in collect. Segments are cut per warp; at each step a
// warp looks at 4 x lanes positions and records 8 words (4 pairs of votes). Every warp gets the same
// number of words, sized for the largest step count: a segment spans at most segment length + 4
// positions (the steps of warp 0 start up to 3 positions before its segment), and the last warp may be
// partial (only in the host emulation), which gives it the shortest stride and the most steps.
// The launcher on the GPU (topk_gpu.cu) also uses this to size shared memory.
DETTOPK_HOST_DEVICE inline int64_t collect_bits_words(int64_t len, int threads, int group_size) {
    const int groups = (threads + group_size - 1) / group_size;
    const int last_lanes = threads - (groups - 1) * group_size;
    const int64_t span = collect_seg(len, groups) + 4;
    const int64_t stride = 4 * static_cast<int64_t>(last_lanes);
    return 8 * static_cast<int64_t>(groups) * ((span + stride - 1) / stride);
}

// Reads 4 floats at once starting at p, whose address must be a multiple of 16 bytes. On the GPU it
// compiles to one 128-bit load: fewer instructions than four 32-bit loads, and more loads in flight at
// a time. The host emulation simply reads 4 times.
DETTOPK_HOST_DEVICE inline void load4(const float* p, float v[4]) {
#ifdef __CUDA_ARCH__
    const float4 f = *reinterpret_cast<const float4*>(p);
    v[0] = f.x;
    v[1] = f.y;
    v[2] = f.z;
    v[3] = f.w;
#else
    v[0] = p[0];
    v[1] = p[1];
    v[2] = p[2];
    v[3] = p[3];
#endif
}

// How many elements at the start of a row are not 16-byte aligned: from that many onwards, groups of 4
// can be read with load4. At most len. A row starts at score number row x width, so when the width is
// not a multiple of 4 the rows start at different alignments.
DETTOPK_HOST_DEVICE inline int64_t unaligned_head(const float* row, int64_t len) {
    const int64_t head = static_cast<int64_t>(((16u - (reinterpret_cast<uintptr_t>(row) & 15u)) & 15u) / 4u);
    return head < len ? head : len;
}

// How every pass of selection reads the whole row: the threads of the block share out the first len
// scores of the row, reading each exactly once and calling f(score) for it. Each element goes to
// exactly one thread, exactly once; which thread gets which is unspecified (counting does not care).
// The row is cut into three stretches:
//   the first few elements that are not 16-byte aligned (at most 3), one per thread;
//   the middle, in quads (groups of 4) dealt round-robin to the threads and read one quad per load4, so
//     that neighbouring threads read neighbouring quads;
//   the last few elements that do not fill a quad (at most 3), one per thread.
// Indices never exceed len and are 32-bit (fewer registers).
template <class F>
DETTOPK_HOST_DEVICE inline void for_each_score(int tid, int T, const float* row, int32_t len, F&& f) {
    const int32_t head = static_cast<int32_t>(unaligned_head(row, len));
    const int32_t quads = (len - head) / 4;
    const int32_t tail = head + 4 * quads;
    for (int32_t i = tid; i < head; i += T) f(row[i]);
    // The middle stretch: thread tid handles quads tid, tid + T, tid + 2T, ... as it would reading one quad
    // per round, except that each round first issues the loads of two quads (q and q + T) back to back
    // and then processes both. A load from global memory takes a few hundred nanoseconds; with twice as
    // many loads in flight a thread block reads faster (most visibly when there are few rows and only a
    // few thread blocks on the GPU). In the last round the second quad may not exist.
    const float* body = row + head;
    for (int32_t q = tid; q < quads; q += 2 * T) {
        const bool second = q + T < quads;
        float v[4], u[4];
        load4(body + 4 * q, v);
        if (second) load4(body + 4 * (q + T), u);
        f(v[0]);
        f(v[1]);
        f(v[2]);
        f(v[3]);
        if (second) {
            f(u[0]);
            f(u[1]);
            f(u[2]);
            f(u[3]);
        }
    }
    for (int32_t i = tail + tid; i < len; i += T) f(row[i]);
}

// Same as for_each_score, except that it calls f(score, position): the sampled path has to remember the
// positions of the candidates when it reads the row. It is a separate copy rather than an extra
// parameter of for_each_score, because the other passes need no position and this keeps their register
// allocation from changing.
template <class F>
DETTOPK_HOST_DEVICE inline void for_each_score_pos(int tid, int T, const float* row, int32_t len, F&& f) {
    const int32_t head = static_cast<int32_t>(unaligned_head(row, len));
    const int32_t quads = (len - head) / 4;
    const int32_t tail = head + 4 * quads;
    for (int32_t i = tid; i < head; i += T) f(row[i], i);
    for (int32_t q = tid; q < quads; q += 2 * T) {
        const bool second = q + T < quads;
        float v[4], u[4];
        const int32_t i0 = head + 4 * q;
        load4(row + i0, v);
        if (second) load4(row + i0 + 4 * T, u);
        for (int j = 0; j < 4; ++j) f(v[j], i0 + j);
        if (second) {
            for (int j = 0; j < 4; ++j) f(u[j], i0 + 4 * T + j);
        }
    }
    for (int32_t i = tail + tid; i < len; i += T) f(row[i], i);
}

// When a row is split across `parts` blocks (see SplitCtx), block `part` reads only its own stretch.
// The cut lines up with for_each_score_pos: the `quads` quads of the middle are divided evenly by
// index, block `part` taking quads quads x part / parts to quads x (part + 1) / parts - 1; the
// unaligned elements at the start go to block 0, and the ones at the end that do not fill a quad go to
// the last block. Together the blocks cover the row with no gap and no overlap. How the threads of a
// block share these quads, and the two loads issued back to back per round, are as in
// for_each_score_pos. A quad index times a block index may exceed 2^31, so these two bounds are
// computed in 64 bits (once per block).
template <class F>
DETTOPK_HOST_DEVICE inline void for_each_score_pos_part(int tid, int T, const float* row, int32_t len, int part,
                                                     int parts, F&& f) {
    const int32_t head = static_cast<int32_t>(unaligned_head(row, len));
    const int32_t quads = (len - head) / 4;
    const int32_t tail = head + 4 * quads;
    const int32_t q0 = static_cast<int32_t>(static_cast<int64_t>(quads) * part / parts);
    const int32_t q1 = static_cast<int32_t>(static_cast<int64_t>(quads) * (part + 1) / parts);
    if (part == 0) {
        for (int32_t i = tid; i < head; i += T) f(row[i], i);
    }
    for (int32_t q = q0 + tid; q < q1; q += 2 * T) {
        const bool second = q + T < q1;
        float v[4], u[4];
        const int32_t i0 = head + 4 * q;
        load4(row + i0, v);
        if (second) load4(row + i0 + 4 * T, u);
        for (int j = 0; j < 4; ++j) f(v[j], i0 + j);
        if (second) {
            for (int j = 0; j < 4; ++j) f(u[j], i0 + 4 * T + j);
        }
    }
    if (part == parts - 1) {
        for (int32_t i = tail + tid; i < len; i += T) f(row[i], i);
    }
}

// ---- Row splitting: one row across several blocks ----
// When a call has only a few rows, only a few blocks are working on the GPU and the latency of a row is
// the time of the whole call. Yet one block takes 26 microseconds to process the elements of a whole
// row (convert to keys, compare with the cutoff, store candidates), because the SM cannot issue
// instructions any faster. In that case the launcher (topk_gpu.cu) gives a row to `parts` blocks, which
// divide the work on the sampled path only:
//   1. Sample and choose the cutoff: every block does this by itself. They read the same samples with
//      the same algorithm, so they necessarily arrive at the same cutoff t, without waiting for each other.
//   2. Read one stretch: block `part` reads only its own stretch (for_each_score_pos_part) and stores
//      the keys at or above t in the pair area of its own shared memory.
//   3. Hand over candidates: thread 0 atomically adds this block's count nb to the row's candidate
//      counter count[0] and gets the offset o back. The whole block copies its candidates to
//      [o, o + nb) of the row's candidate area cand in global memory, up to the capacity C (the counter
//      is still incremented in full, just as a single block reading the whole row stops storing at the
//      capacity but keeps counting).
//   4. Check in: every thread executes a fence and passes a barrier, then thread 0 atomically adds 1 to
//      the arrival counter count[1]. The block that finds parts - 1 there is the last block; the others
//      end here (returning kSplitDone). The fence guarantees that every block counted in the arrivals
//      the last block sees has already written its candidates.
//   5. Gather candidates: the last block reads the total n, subtracts both counters back to 0 (leaving
//      them ready for the next call, so the launcher need not zero them every time), reads min(n, C)
//      candidates into its own pair area through L2 only, and from "verify" on proceeds exactly as a
//      single block would.
// Example with the 12 keys 19 38 56 9 36 45 38 3 50 35 39 30 (positions 0 to 11) split across 2 blocks:
// t = 36. Block 0 (positions 0 to 5) keeps (38,1) (56,2) (36,4) (45,5) and block 1 (positions 6 to 11)
// keeps (38,6) (50,8) (39,10). Block 1 hands over first and gets slots 0 to 2; block 0 hands over
// later, gets slots 3 to 6, checks in last and finishes the row.
// Why the result is bit-identical to a single block's: the candidate set is "all keys of the row at or
// above t" and the total n is the same. Everything afterwards looks only at this set (verification
// looks at n, picks only count, and marking sets bits by position), so neither the order of the
// candidates nor which block comes last matters. Even the fallback taken is the same.
// No block ever stops to wait for another, so the blocks of a row need not be resident at the same
// time, and having more blocks than fit on the GPU cannot deadlock.
// Rows that cannot take the sampled path (valid length at most k, too short, or not fitting) and rows
// with "too many samples" are finished by block 0 the single-block way while the other blocks exit;
// "cutoff too high" and "does not fit" make the last block fall back to pass 1. A block that finds a
// NaN in its stretch does not check in, so the row has no last block and no output is written (the
// interface leaves the output of a row with a NaN unspecified), and that block reports the NaN. Nobody
// clears the counters of such a row; the launcher zeroes them after the call.
struct SplitCtx {
    int part;         // index of this block
    int parts;        // how many blocks the row is split across; 1 means no splitting
    uint32_t* cand;   // the row's candidate area in global memory: 2 x pair_capacity words, laid out like pair
    uint32_t* count;  // the row's two counters: [0] total candidates, [1] arrivals. 0 at the start of a call; reset in step 5
};

// ---- Samples of the sampled path: which positions are taken ----
// Starting at the aligned position (head = unaligned_head), every kSamplePeriod positions take
// kSampleRun consecutive ones: run u is [head + kSamplePeriod x u, head + kSamplePeriod x u + kSampleRun),
// taken only when the whole run lies within the valid length. That samples 1/16 of the row.
// The positions depend only on the valid length and on where the row starts, never on the scores. A run
// of 32 floats is 128 bytes, one whole cache line. Taking 8 out of every 128 positions (32-byte pieces)
// gives the same ratio, but the scattered small pieces read slowly; the current scheme is 4% faster
// overall. The longer the run, the more fallbacks when neighbouring positions have similar scores; this
// was measured on real scores. Run length and period are compile-time parameters, and changing them
// means recomputing the fallback probabilities.
constexpr int32_t kSampleRun = 32;
constexpr int32_t kSamplePeriod = 512;
static_assert(kSampleRun % 4 == 0 && kSamplePeriod % 4 == 0 && kSampleRun <= kSamplePeriod, "samples are read in quads");

// Number of quads in the samples (each read with one load4).
DETTOPK_HOST_DEVICE inline int32_t sample_quads(int32_t head, int32_t len) {
    const int32_t runs = len - head >= kSampleRun ? (len - head - kSampleRun) / kSamplePeriod + 1 : 0;
    return runs * (kSampleRun / 4);
}

// Position at which quad q of the samples starts.
DETTOPK_HOST_DEVICE inline int32_t sample_quad_pos(int32_t head, int32_t q) {
    constexpr int32_t per_run = kSampleRun / 4;
    return head + kSamplePeriod * (q / per_run) + 4 * (q % per_run);
}

// Whether this row takes the sampled path, the rank r of the sample taken as the cutoff, and the
// maximum number rmax of samples allowed at or above the cutoff. m is the number of samples and C the
// candidate capacity. The sampled path requires that the samples fit in the candidate area (m <= 2C),
// that C is at least k, and that 1 <= r <= rmax <= m. The cutoff need not be exactly the r-th sample:
// it is enough that the number c of samples at or above it lies in [r, rmax] (see "Choose the cutoff"
// in block_process_row). The probabilities of the two fallbacks depend on ranks only: "cutoff too high"
// is at most P(at least r of the top k-1 fall in the sample), and "does not fit" is at most
// P(at most rmax of the top C fall in the sample). The two means are E1 = (k-1) m / len and
// E2 = C m / len. Take
//   r = round(min(sqrt(E1 x E2), E1 + 9 sqrt(E1)))
//                                 which keeps the probability of "cutoff too high" below 1e-15. With
//                                 sqrt(E1 x E2) alone this side is far smaller (below 1e-23 at full
//                                 scale) and the candidates are 2.0 to 2.4 times k. At full scale
//                                 E1 + 9 sqrt(E1) is about 230, this side is about 2e-17 and the
//                                 candidates about 1.8 times k, so reading the row stores fewer
//                                 candidates, and finding tau and the vote fix-up shrink with them:
//                                 about 1% faster per layer on the prefill sweep. In test
//                                 configurations with a very small k, sqrt(E1 x E2) is smaller and is used;
//   rmax = floor(E2 - 7 sqrt(E2))  which leaves about seven standard deviations below E2, as room for
//                                 taking the cutoff at a bucket boundary and for ties.
// Nonzero r_req and rmax_req are used as given (tests force fallbacks with them). These two numbers
// only affect how often a fallback is taken, never the result.
DETTOPK_HOST_DEVICE inline bool sample_plan(int32_t len, int32_t k, int32_t head, uint32_t C, uint32_t r_req,
                                         uint32_t rmax_req, uint32_t& r, uint32_t& rmax) {
    const int32_t m = 4 * sample_quads(head, len);
    if (m <= 0 || static_cast<uint32_t>(m) > 2 * C || C < static_cast<uint32_t>(k)) return false;
    const float e1 = static_cast<float>(k - 1) * static_cast<float>(m) / static_cast<float>(len);
    const float e2 = static_cast<float>(C) * static_cast<float>(m) / static_cast<float>(len);
    const float r_geo = sqrtf(e1 * e2), r_tail = e1 + 9.0f * sqrtf(e1);
    const float rf = (r_geo < r_tail ? r_geo : r_tail) + 0.5f;
    const float rmaxf = e2 - 7.0f * sqrtf(e2);
    r = r_req ? r_req : (rf >= static_cast<float>(m) ? static_cast<uint32_t>(m) : static_cast<uint32_t>(rf));
    rmax = rmax_req ? rmax_req
                    : (rmaxf <= 0.0f ? 0u
                                     : (rmaxf >= static_cast<float>(m) ? static_cast<uint32_t>(m)
                                                                       : static_cast<uint32_t>(rmaxf)));
    if (r < 1) r = 1;
    if (r > static_cast<uint32_t>(m)) r = static_cast<uint32_t>(m);
    if (rmax > static_cast<uint32_t>(m)) rmax = static_cast<uint32_t>(m);
    return r <= rmax;
}

// ---- Segments of collect: pass 2 of the merged path, steps (1) and (3) of collect and the vote fix-up
// all cut the row this one way, defined here in one place ----
// Segments go by warp: warp q owns [begin, end). A warp walks right from `start` together, looking at
// `stride` consecutive positions per step; thread `lane` of the warp looks at positions 4 x lane to
// 4 x lane + 3 of the step, read with one load4. Why the row is cut this way is explained under
// "Collect" in block_process_row.
// Vote bits: one block of words per warp, starting at word bits0, 8 words per step (4 pairs of votes:
// pair j is "the j-th key I look at is above tau" and "is equal to tau"; bit L of a word belongs to
// thread L of the warp).
struct CollectSeg {
    int32_t begin, end;  // the segment
    int32_t start;       // where the steps start: a multiple of 4 away from head and not after begin
    int32_t stride;      // positions per step: 4 x the number of threads in this warp
    int32_t bits0;       // the word at which this warp's vote bits start
};

// The segment of warp q. T threads, S per warp; head is unaligned_head, seg is collect_seg and words is
// the number of words of the whole row's vote bits (collect_bits_words; 0 when votes are not recorded).
// Everything is computed in 32 bits: segment lengths and word counts are far below 2^31, and a 64-bit
// division on the GPU calls a subroutine and takes more registers.
DETTOPK_HOST_DEVICE inline CollectSeg collect_segment(int q, int T, int S, int32_t head, int32_t seg, int32_t len,
                                                   int32_t words) {
    const int groups = (T + S - 1) / S;
    const int lanes = (q + 1) * S <= T ? S : T - q * S;  // the last warp may be partial (only in the host emulation)
    const int32_t seg_end = head + (q + 1) * seg;  // at most head + groups x seg < len + 4 x groups + 3
    CollectSeg c;
    c.begin = q == 0 ? 0 : (seg_end - seg < len ? seg_end - seg : len);
    c.end = seg_end < len ? seg_end : len;
    c.start = c.begin - ((c.begin - head) & 3);
    c.stride = 4 * lanes;
    c.bits0 = words / groups * q;
    return c;
}

// The high 32 bits of the product of two 32-bit unsigned numbers. One instruction on the GPU.
DETTOPK_HOST_DEVICE inline uint32_t mul_hi32(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
    return __umulhi(a, b);
#else
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
#endif
}

// The vote fix-up replaces "divide by the segment length" with a multiplication. Once per row compute
// m = floor((2^32 - 1) / seg). For x, q0 = the high 32 bits of x * m is at most x / seg and at most 1
// below it (m > (2^32 - 1) / seg - 1, x < 2^31 and seg >= 4); add 1 when the remainder is at least seg.
// The result is bit-identical to the division.
struct SegDiv {
    uint32_t seg;  // segment length
    uint32_t m;    // floor((2^32 - 1) / seg)
};
DETTOPK_HOST_DEVICE inline SegDiv make_seg_div(int32_t seg) {
    return SegDiv{static_cast<uint32_t>(seg), 0xFFFFFFFFu / static_cast<uint32_t>(seg)};
}

// Where the vote bits of position i (0 <= i < len) are: returns the index of the word holding its
// "above" bit (the "equal" bit is in the next word); bit is the bit index and q the segment index.
// This is the inverse of the walk of collect (collect_segment, walk_segment): the position first gives
// the segment (segment 0 also covers the `head` elements at the start), then the distance from the
// start of the steps gives the step, the thread within the warp and which of its 4 positions it is.
// group_words is the number of words of vote bits per warp (words / groups in collect_segment); the
// other parameters are those of collect_segment.
// This runs once per candidate, so it avoids division. The segment index comes from a SegDiv
// multiplication. The segment length is a multiple of 4, so from segment 1 on the steps start at the
// segment start and the distance from it is the remainder; the steps of segment 0 start at head - 4
// (at 0 when head is 0). On the GPU every warp is full, the stride is the constant 128 and the division
// compiles to a shift; in the host emulation the last warp may be partial, and its actual thread count
// is used.
DETTOPK_HOST_DEVICE inline int32_t collect_slot(int32_t i, int T, int S, int32_t head, const SegDiv& d,
                                             int32_t group_words, uint32_t& bit, int& q) {
    int32_t o = i - head;  // positions before head are in segment 0 with a negative o (at least 1 after adding 4)
    q = 0;
    if (o >= 0) {
        uint32_t qq = mul_hi32(static_cast<uint32_t>(o), d.m);
        uint32_t r = static_cast<uint32_t>(o) - qq * d.seg;
        if (r >= d.seg) {
            ++qq;
            r -= d.seg;
        }
        q = static_cast<int>(qq);
        o = static_cast<int32_t>(r);
    }
    if (q == 0 && head != 0) o += 4;
#ifdef __CUDA_ARCH__
    (void)T;
    const uint32_t stride = 4u * static_cast<uint32_t>(S);
#else
    const uint32_t stride = 4u * static_cast<uint32_t>((q + 1) * S <= T ? S : T - q * S);
#endif
    const uint32_t uo = static_cast<uint32_t>(o);
    const uint32_t t = uo / stride;
    const uint32_t r = uo - t * stride;
    bit = r / 4;
    return group_words * q + static_cast<int32_t>(8 * t + 2 * (r & 3));
}

// The scores of the 4 positions from p0, read without conversion. Positions outside the segment are not
// read and count as 0.0f (the caller must check separately whether a position is in the segment).
DETTOPK_HOST_DEVICE inline void seg_quad_load(const float* row, const CollectSeg& c, int32_t p0, float v[4]) {
    if (p0 >= c.begin && p0 + 4 <= c.end) {
        load4(row + p0, v);
    } else {
        for (int j = 0; j < 4; ++j) {
            const int32_t i = p0 + j;
            v[j] = i >= c.begin && i < c.end ? row[i] : 0.0f;
        }
    }
}

// The keys of the 4 positions from p0. Positions outside the segment get the key of 0.0f (the caller
// must check separately whether a position is in the segment).
DETTOPK_HOST_DEVICE inline void seg_quad_keys(const float* row, const CollectSeg& c, int32_t p0, uint32_t key[4]) {
    float v[4];
    seg_quad_load(row, c, p0, v);
    for (int j = 0; j < 4; ++j) key[j] = score_to_key(v[j]);
}

// A warp walks its own segment reading scores: at each step, thread `lane` of the warp gets the 4 keys
// from p0 and calls f(p0, key, w), where w is the word at which this step's vote bits start. The
// threads of a warp take the same number of steps, so f may perform warp votes.
// As in for_each_score, each round first issues the loads of two steps (this one and the next) back to
// back and then processes both, doubling the loads in flight. What is read is kept as scores and only
// converted to keys when processed: conversion has to wait for the load to return, and placed between
// the two loads it would make the second load wait for the first.
// In the last round the second step may not exist (all threads of a warp see the same thing).
template <class F>
DETTOPK_HOST_DEVICE inline void walk_segment(const float* row, const CollectSeg& c, int lane, F&& f) {
    int32_t w = c.bits0;
    for (int32_t base = c.start; base < c.end; base += 2 * c.stride, w += 16) {
        const int32_t p0 = base + 4 * lane;
        const bool second = base + c.stride < c.end;
        float v[4], u[4];
        seg_quad_load(row, c, p0, v);
        if (second) seg_quad_load(row, c, p0 + c.stride, u);
        uint32_t key[4];
        for (int j = 0; j < 4; ++j) key[j] = score_to_key(v[j]);
        f(p0, key, w);
        if (second) {
            for (int j = 0; j < 4; ++j) key[j] = score_to_key(u[j]);
            f(p0 + c.stride, key, w + 8);
        }
    }
}

// Clears the histogram, the threads sharing the work.
template <class P>
DETTOPK_HOST_DEVICE void clear_hist(const P& p, uint32_t* hist, uint32_t buckets) {
    for (uint32_t b = static_cast<uint32_t>(p.tid()); b < buckets; b += static_cast<uint32_t>(p.threads())) {
        hist[b] = 0;
    }
}

// The pick: the threads of the block together find which bucket holds the s.need-th largest key
// (counting from the largest bucket down). The result goes to s.pick, and s.need becomes the rank still
// to find inside that bucket. The result equals that of pick_bucket in radix.h, where a single thread
// walks down from the largest bucket, more than two thousand steps over the three passes of a row.
// Here it takes four steps and only two barriers:
//   (1) The buckets, from the largest index down, are cut into T segments; thread t takes segment t
//       and sums it (keeping the sum in a register).
//   (2) A prefix sum inside the warp by doubling: shuffling within the warp, each thread obtains the
//       sum of "my warp up to and including my segment". The last thread of each warp writes the
//       warp's total to seg_g[warp] (seg_g has no other use before collect). Barrier.
//   (3) Every warp reads the totals of all warps and, by shuffling, adds up those of the warps before
//       its own. What precedes a thread's own segment (larger bucket indices) totals
//       before = that + the sum up to me - my segment's sum. Each thread checks whether the need-th key
//       falls in its segment: before < need <= before + my segment's sum. Exactly one thread qualifies.
//   (4) That thread walks down from the largest bucket of its segment the old way and finds the bucket.
//       Barrier.
// Step (2) used to be done by the first thread of each warp and then by thread 0 adding sequentially,
// each behind a barrier, four barriers in all; one barrier costs over a thousand cycles on a wide row.
// The sums are the same integer sums, so the result is bit-identical.
// Before the call s.need must already be visible to all threads (a barrier has been passed) and must
// not exceed the total of the histogram. On return a barrier has been passed.
// Every thread of the block must call it (it contains warp shuffles).
template <class P>
DETTOPK_HOST_DEVICE void block_pick_bucket(const P& p, const uint32_t* hist, uint32_t buckets, const BlockShared& sh) {
    const int tid = p.tid();
    const int T = p.threads();
    const int S = p.group_size();
    const int group = tid / S;
    const int lane = tid % S;
#ifdef __CUDA_ARCH__
    const int lanes = S;  // on the GPU every warp is full (threads are a multiple of 32); a constant lets the shuffles below unroll
#else
    const int lanes = (group + 1) * S <= T ? S : T - group * S;  // the last warp may be partial (only in the host emulation)
#endif
    BlockScalars& s = *sh.s;
    const uint32_t need = s.need;

    // (1) Segment tid is the buckets [lo, hi). With more threads than buckets, the later threads get empty
    // segments. At most 2048 buckets and 1024 threads, so 32-bit arithmetic suffices (fewer registers).
    const int seg = (static_cast<int>(buckets) + T - 1) / T;
    const int top = static_cast<int>(buckets) - tid * seg;
    const uint32_t hi = top > 0 ? static_cast<uint32_t>(top) : 0u;
    const uint32_t lo = top > seg ? hi - static_cast<uint32_t>(seg) : 0u;
    uint32_t sum = 0;
    for (uint32_t b = lo; b < hi; ++b) sum += hist[b];
    // (2) Doubling: in round r each thread adds the number held by the thread 2^r below it. Afterwards inc
    // is the sum of my warp up to and including my segment.
    uint32_t inc = sum;
    for (int d = 1; d < lanes; d <<= 1) {
        const uint32_t v = p.shfl_up(inc, d);
        if (lane >= d) inc += v;
    }
    if (lane == lanes - 1) sh.seg_g[group] = inc;
    p.sync();
    // (3) The sum of the warps before mine. The GPU has at most 32 warps, so one entry per thread (the
    // total of warp `lane`) is enough. In the host emulation there may be more warps than threads in a
    // warp, so thread `lane` adds every warp whose index is congruent to lane modulo lanes (striding by
    // the actual thread count of this warp: with a partial last warp, striding by S would skip entries).
    // Then another doubling sum inside the warp; the last thread holds the total and shuffles it to everyone.
    uint32_t acc = 0;
    for (int q = lane; q < group; q += lanes) acc += sh.seg_g[q];
    for (int d = 1; d < lanes; d <<= 1) {
        const uint32_t v = p.shfl_up(acc, d);
        if (lane >= d) acc += v;
    }
    const uint32_t before = p.shfl(acc, lanes - 1) + inc - sum;
    // (4) The thread whose segment holds the need-th key walks down from the largest bucket of its segment.
    if (before < need && need - before <= sum) {
        uint32_t n = need - before;
        uint32_t b = hi - 1;
        while (hist[b] < n) {
            n -= hist[b];
            --b;
        }
        s.pick = b;
        s.need = n;
    }
    p.sync();
}

// Step (3) of collect: a warp writes out its own segment. g and e are the numbers of keys above tau and
// equal to tau to the left of the segment start (the result of step (2)); bits is the row's vote bits
// (null when votes are not recorded, in which case the row is read and voted on again).
//
// Each thread of the warp owns one "regrouped word": the 4 vote words of a step are regrouped by
// position (regroup), bit b of regrouped word m being position 32m + b of the step, so that a regrouped
// word covers 32 consecutive positions. A full warp covers 128 positions per step, regrouped into 4 words.
// With recorded votes a round covers 8 steps, 1024 positions for a full warp: thread L owns regrouped
// word L % 4 of step L / 4 of the round, lower threads owning words further left. Without recorded
// votes a round covers 1 step: the row is read and 8 votes are cast, the first `chunks` threads own one
// regrouped word each and the rest own empty words.
// In each round:
//   (1) every thread counts the 1 bits in its two regrouped words (above, equal);
//   (2) a prefix sum inside the warp by doubling: in round r each thread adds the number held by the
//       thread 2^r below it. The two counts are packed into one number and added together (a round has
//       at most 1024 positions, so 16 bits each suffice). Minus my own, that is what lies to my left
//       within this round; plus the g and e from before the round, it gives the number gi of keys
//       above tau and the number ei of keys equal to tau to the left of my regrouped word;
//   (3) tie quota: a position equal to tau is selected only while the number of equals to its left is
//       below the quota. If ei has reached the quota, none in the word is selected; if ei plus the
//       count in the word does not exceed the quota, all are; otherwise only the lowest "quota - ei"
//       1 bits are kept. Only one regrouped word per row falls into the last case;
//   (4) selected = above, or equal and kept. Starting at slot gi + min(ei, quota), follow the selected
//       1 bits from low to high, writing one slot each and incrementing the slot. The slot formula is
//       "keys above tau to the left + min(keys equal to tau to the left, quota)": every selected
//       position earlier in the same word adds 1 to it, and unselected positions equal to tau occur
//       only once the quota is full, when the min no longer changes. So this is the formula unrolled
//       bit by bit;
//   (5) the prefix sum of the last thread of the warp is the total of the round; it is shuffled to
//       everyone and added to g and e.
// A regrouped word that is 0 writes nothing: the work of writing follows the number of selected
// positions, not the row width.
// In one round of write_segment each thread takes its two regrouped words (wg for above, we for equal;
// bit 0 of a word is position pos0) through (1) to (5). g and e come in as the counts to the left of
// the start of the round and go out with the round's total added. The threads of a warp must call it
// together.
template <class P>
DETTOPK_HOST_DEVICE inline void write_round(const P& p, int lane, int lanes, uint32_t wg, uint32_t we, int32_t pos0,
                                         uint32_t quota, uint32_t& g, uint32_t& e, int32_t* dst) {
    // (1) (2) prefix sum inside the warp: the count of "above" in the low 16 bits, of "equal" in the high 16
    const uint32_t ne = popcount32(we);
    const uint32_t mine = popcount32(wg) | (ne << 16);
    uint32_t inc = mine;  // the total of the regrouped words of the threads up to and including me in the warp
    for (int d = 1; d < lanes; d <<= 1) {
        const uint32_t v = p.shfl_up(inc, d);
        if (lane >= d) inc += v;
    }
    const uint32_t round_sum = p.shfl(inc, lanes - 1);
    const uint32_t left = inc - mine;
    const uint32_t gi = g + (left & 0xFFFFu);
    const uint32_t ei = e + (left >> 16);
    // (3) tie quota
    if (we != 0) {
        if (ei >= quota) {
            we = 0;
        } else if (ei + ne > quota) {
            uint32_t rest = we;
            for (uint32_t keep = quota - ei; keep > 0; --keep) rest &= rest - 1;  // drop the lowest `keep` 1 bits
            we ^= rest;                                                          // what remains are those `keep` bits
        }
    }
    // (4) write-out
    uint32_t sel = wg | we;
    uint32_t slot = gi + (ei < quota ? ei : quota);
    while (sel != 0) {
        dst[slot++] = pos0 + static_cast<int32_t>(lowest_bit(sel));
        sel &= sel - 1;
    }
    // (5)
    g += round_sum & 0xFFFFu;
    e += round_sum >> 16;
}

// dst is the output of this row (k slots); the slot is the index.
// Recorded and unrecorded votes are two separate loops: the loop for recorded votes (the usual one)
// need not keep alive the few values that only the other case uses, which saves registers.
template <class P>
DETTOPK_HOST_DEVICE void write_segment(const P& p, const float* row, const CollectSeg& c, int lane, const uint32_t* bits,
                                    uint32_t tau, uint32_t quota, uint32_t g, uint32_t e, int32_t* dst) {
#ifdef __CUDA_ARCH__
    // On the GPU every warp is full, so this is a constant: the warp shuffles in write_round unroll
    // completely and several values below become constants too. Computed from c.stride, the compiler
    // does not know it is 32 and compiles the shuffles into a small loop with branches, which is about
    // 2% slower per layer on the prefill sweep.
    const int lanes = 32;
#else
    const int lanes = c.stride / 4;      // number of threads in this warp (in the host emulation the last warp may be partial)
#endif
    const int chunks = (lanes + 7) / 8;  // regrouped words per step: each uses the votes of 8 threads
    if (bits) {
        // A round covers per_round steps; I own regrouped word my_m of step my_t of the round. A full
        // warp has 8 steps per round and thread L owns regrouped word L % 4 of step L / 4. When the
        // threads do not fill whole steps, the extra threads idle (only in the host emulation): their
        // my_t is set very large, so the test below whether the step is inside the segment fails.
        const int per_round = lanes / chunks;
        const int my_t = lane < per_round * chunks ? lane / chunks : 1 << 20;
        const int my_m = lane % chunks;
        const int32_t my_step = my_t * c.stride;  // how far my step is from the start of the round
        const uint32_t* w = bits + c.bits0 + 8 * my_t;  // the vote bits of my step (idle threads do not read them)
        for (int32_t base = c.start; base < c.end; base += per_round * c.stride, w += 8 * per_round) {
            uint32_t wg = 0, we = 0;  // my regrouped words: above, equal
            if (base + my_step < c.end) {
                wg = regroup(w[0], w[2], w[4], w[6], my_m);
                // The "equal" votes are almost all 0 (a row usually has only one or two keys equal to
                // tau). When they are 0 across the warp, the whole warp skips this statement on the GPU.
                if ((w[1] | w[3] | w[5] | w[7]) != 0) we = regroup(w[1], w[3], w[5], w[7], my_m);
            }
            write_round(p, lane, lanes, wg, we, base + my_step + 32 * my_m, quota, g, e, dst);
        }
    } else {
        // Unrecorded votes: 1 step per round. Read the row and cast 8 votes; the first `chunks` threads
        // own one regrouped word each and the rest own empty words.
        for (int32_t base = c.start; base < c.end; base += c.stride) {
            const int32_t p0 = base + 4 * lane;
            uint32_t key[4];
            seg_quad_keys(row, c, p0, key);
            uint32_t vg[4], ve[4];
            for (int j = 0; j < 4; ++j) {
                const bool in_seg = p0 + j >= c.begin && p0 + j < c.end;  // positions past either end of the segment vote "no"
                vg[j] = p.ballot(in_seg && key[j] > tau);
                ve[j] = p.ballot(in_seg && key[j] == tau);
            }
            uint32_t wg = 0, we = 0;
            if (lane < chunks) {
                wg = regroup(vg[0], vg[1], vg[2], vg[3], lane);
                we = regroup(ve[0], ve[1], ve[2], ve[3], lane);
            }
            write_round(p, lane, lanes, wg, we, base + 32 * lane, quota, g, e, dst);
        }
    }
}

// Processes one row. row is the row's scores, len its valid length and out its output (k slots).
// All threads of the block return the same value.
// select_only is for timing the two phases separately: return once selection is done, without collect
// (see the end of selection below).
// prof is used only by the profiling build (see DETTOPK_PROFILE at the top of the file); it has kProfSlots
// slots and is normally null.
// When Split is true the row is split across several blocks and sp says which block this is (see
// SplitCtx); a block that returns kSplitDone has only done its own share.
template <class P, bool Split = false>
DETTOPK_HOST_DEVICE RowResult block_process_row(const P& p, const float* row, int32_t len, int64_t width,
                                             int32_t k, int32_t* out, const BlockShared& sh,
                                             bool select_only = false, uint64_t* prof = nullptr,
                                             SplitCtx sp = SplitCtx{0, 1, nullptr, nullptr}) {
    using namespace detail;
    const int tid = p.tid();
    const int T = p.threads();
    const int S = p.group_size();
    BlockScalars& s = *sh.s;
    // With splitting (Split), which block finishes, the single-block way, a row that cannot take the
    // sampled path or falls back: block 0 at first, the last block after check-in. Without splitting it
    // is always this block. Split is a compile-time switch, and in the unsplit instantiation these tests
    // compile away (its machine code is the same as before the switch existed).
    bool owner = !Split || sp.part == 0;
    (void)sp;
    // The cycles of each step accumulate in prof[step]: when the sampled path falls back, the steps from
    // pass 1 on are recorded a second time.
    DETTOPK_PROF(uint64_t prof_last = p.clock();
              auto prof_step = [&](int step) {
                  if (tid == 0 && prof) {
                      const uint64_t now = p.clock();
                      prof[step] += now - prof_last;
                      prof_last = now;
                  }
              };)
    (void)prof;

    if (len < 0 || len > width) return RowResult::kBadLength;

    // Valid length at most k: everything is selected; write 0, 1, ... followed by -1 padding. The scores
    // are not read.
    if (len <= k) {
        if (!owner) return RowResult::kSplitDone;
        for (int64_t i = tid; i < k; i += T) out[i] = i < len ? static_cast<int32_t>(i) : -1;
        return RowResult::kOk;
    }
    // The sampled-path outcome is written only by thread 0 and only read by tests and profiling (see
    // SampleOutcome). Clear it for every row.
    if (tid == 0) s.sample_outcome = kSampleNone;

    uint32_t tau = 0, quota = 0;  // the result of selection: the k-th key and the tie quota
    uint32_t tau_count = 0;       // sampled path: candidates equal to tau (at least the quota); see "Rank write-out on the sampled path"
    bool selected = false;        // the sampled path has already found tau

    // ---- Sampled path: choose a cutoff t from samples first, then read the whole row only once ----
    // 1. Sample: read 1/16 of the row (positions per sample_quads), store the keys in pair and count
    //    them by the top 11 bits; clear the vote area along the way.
    // 2. Choose the cutoff: t is the smallest key of the bucket that holds the r-th sample, and the
    //    number c of samples at or above t must lie in [r, rmax]. If c is too large, refine one level
    //    by the middle 11 bits; if it is still too large, fall back (the row has not been read yet).
    // 3. Read the row: keys at or above t are stored in pair together with their positions (candidate i
    //    has its key in entry 2i and its position in entry 2i + 1), and their number n is counted.
    // 4. Verify: go on only if k <= n <= capacity, otherwise fall back. n >= k means that the k-th key
    //    tau is at or above t, so every key at or above tau (including every one equal to tau) is among
    //    the candidates; keys outside the candidates are below t, hence the k-th key of the candidates
    //    is the k-th key of the row.
    // 5. Scan the candidates counting by the top 11 bits and pick three times within the candidates to
    //    get tau and the tie quota. In collect, the vote fix-up sets the vote bits of every candidate
    //    at or above tau (the vote area was zeroed in step 1).
    // Falling back means taking the original route from "pass 1" below: the result is determined by the
    // data alone, and the output is the same whichever path is taken.
    // The sample positions are fixed and t depends only on the keys of the samples; the candidates are
    // "everything at or above t", uniquely determined by the data. Their order in pair may differ from
    // run to run, but picks only count and the vote fix-up computes where the vote bits are from the
    // position, so neither depends on the order.
    if (sh.sample_min_len != 0 && static_cast<uint32_t>(len) >= sh.sample_min_len) {
        const int32_t head = static_cast<int32_t>(unaligned_head(row, len));
        const int32_t quads = sample_quads(head, len);
        const int64_t words = collect_bits_words(len, T, S);  // number of words of this row's vote bits
        clear_hist(p, sh.hist, kHighBuckets);
        if (tid == 0) {
            uint32_t r = 0, rmax = 0;
            const bool ok = words <= static_cast<int64_t>(sh.pair_bits_capacity) &&
                            sample_plan(len, k, head, sh.pair_capacity, sh.sample_r_req, sh.sample_rmax_req, r, rmax);
            s.path = ok ? kPathSampled : kPathNone;
            s.sample_r = r;
            s.sample_rmax = rmax;
            s.need = r;
            s.nan_seen = 0;
        }
        p.sync();
        if (s.path == kPathSampled) {
            // 1. Sample. As when reading the row, each thread issues the loads of two quads back to
            // back per round and then processes both.
            bool nan = false;
            auto put = [&](const float v[4], int32_t q) {
                for (int j = 0; j < 4; ++j) {
                    const uint32_t bits = float_bits(v[j]);
                    if (is_nan_bits(bits)) {
                        nan = true;
                    } else {
                        const uint32_t key = bits_to_key(bits);
                        sh.pair[4 * q + j] = key;
                        p.fetch_add(&sh.hist[high_bits(key)], 1u);
                    }
                }
            };
            for (int32_t q = tid; q < quads; q += 2 * T) {
                const bool second = q + T < quads;
                float v[4], u[4];
                load4(row + sample_quad_pos(head, q), v);
                if (second) load4(row + sample_quad_pos(head, q + T), u);
                put(v, q);
                if (second) put(u, q + T);
            }
            for (int64_t i = tid; i < words; i += T) sh.pair_bits[i] = 0;
            if (nan) p.fetch_add(&s.nan_seen, 1u);
            p.sync();
            if (s.nan_seen != 0) return RowResult::kNan;  // a NaN in the samples is a NaN in the row
            DETTOPK_PROF(prof_step(kProfSample);)

            // 2. Choose the cutoff. A pick finds the bucket b holding the r-th sample; the number of
            // samples in buckets b and above is c = r - the rank still to find + the count of bucket b.
            block_pick_bucket(p, sh.hist, kHighBuckets, sh);
            const uint32_t b = s.pick;
            uint32_t t = b << 21;
            bool many = s.sample_r - s.need + sh.hist[b] > s.sample_rmax;
            if (many) {
                // Too many samples in bucket b: clear the histogram, count the samples whose top 11 bits
                // equal b by their middle 11 bits, pick once more and extend t to 22 bits.
                p.sync();  // everyone has finished reading the histogram
                clear_hist(p, sh.hist, kMidBuckets);
                p.sync();
                const int32_t m = 4 * quads;
                for (int32_t j = tid; j < m; j += T) {
                    const uint32_t key = sh.pair[j];
                    if (high_bits(key) == b) p.fetch_add(&sh.hist[mid_bits(key)], 1u);
                }
                p.sync();
                block_pick_bucket(p, sh.hist, kMidBuckets, sh);
                t |= s.pick << 10;
                many = s.sample_r - s.need + sh.hist[s.pick] > s.sample_rmax;
            }
            p.sync();  // everyone has finished reading the histogram and s
            DETTOPK_PROF(prof_step(kProfThresh);)
            if (many) {
                if (tid == 0) s.sample_outcome = kSampleMany;
            } else {
                // 3. Read the row. A candidate whose slot is at or beyond the capacity is not stored but is
                // still counted (n is the final value of the counter).
                const uint32_t C = sh.pair_capacity;
                clear_hist(p, sh.hist, kHighBuckets);
                if (tid == 0) {
                    s.need = static_cast<uint32_t>(k);
                    s.cand_n = 0;
                    s.small_n = 0;
                }
                p.sync();
                nan = false;
                auto keep = [&](float x, int32_t i) {
                    const uint32_t bits = float_bits(x);
                    if (is_nan_bits(bits)) {
                        nan = true;
                        return;
                    }
                    const uint32_t key = bits_to_key(bits);
                    if (key < t) return;
                    const uint32_t slot = p.fetch_add(&s.cand_n, 1u);
                    if (slot < C) {
                        sh.pair[2 * slot] = key;
                        sh.pair[2 * slot + 1] = static_cast<uint32_t>(i);
                    }
                };
                if constexpr (Split) {
                    for_each_score_pos_part(tid, T, row, len, sp.part, sp.parts, keep);
                } else {
                    for_each_score_pos(tid, T, row, len, keep);
                }
                if (nan) p.fetch_add(&s.nan_seen, 1u);
                p.sync();
                if (s.nan_seen != 0) return RowResult::kNan;
                if constexpr (Split) {
                    // With splitting: hand over candidates, check in, and the last block gathers them
                    // (steps 3 to 5 of SplitCtx). s.sample_r is borrowed for the offset in global memory
                    // and s.sample_rmax for "I am the last block"; neither is used again once the cutoff
                    // is chosen. No fields are added instead: a larger BlockScalars would change the
                    // static shared memory of the unsplit kernel, and with it possibly the work area and
                    // the capacities.
                    const uint32_t nb = s.cand_n;
                    if (tid == 0) s.sample_r = p.fetch_add(&sp.count[0], nb);
                    p.sync();
                    const uint32_t base = s.sample_r;
                    const uint32_t mine = nb < C ? nb : C;
                    for (uint32_t j = static_cast<uint32_t>(tid); j < mine && base + j < C;
                         j += static_cast<uint32_t>(T)) {
                        sp.cand[2 * (base + j)] = sh.pair[2 * j];
                        sp.cand[2 * (base + j) + 1] = sh.pair[2 * j + 1];
                    }
                    p.fence();
                    p.sync();
                    if (tid == 0) {
                        const uint32_t arrived = p.fetch_add(&sp.count[1], 1u);
                        s.sample_rmax = arrived + 1 == static_cast<uint32_t>(sp.parts) ? 1u : 0u;
                    }
                    p.sync();
                    if (s.sample_rmax == 0) return RowResult::kSplitDone;
                    owner = true;
                    // The last block gathers the candidates. The total is read atomically (adding 0) and
                    // the candidates are read through L2 only. s.cand_n becomes the total of the whole
                    // row, which verification and the picks below use as it is. After reading, both
                    // counters are subtracted back to 0: all the other blocks have checked in and will
                    // not touch them again. An atomic add (of minus the value read) is used rather than
                    // a separate store operation, so the host emulation works unchanged.
                    p.fence();
                    if (tid == 0) {
                        const uint32_t got = p.fetch_add(&sp.count[0], 0u);
                        s.cand_n = got;
                        p.fetch_add(&sp.count[0], 0u - got);
                        p.fetch_add(&sp.count[1], 0u - static_cast<uint32_t>(sp.parts));
                    }
                    p.sync();
                    const uint32_t total = s.cand_n;
                    const uint32_t m = total < C ? total : C;
                    for (uint32_t j = static_cast<uint32_t>(tid); j < 2 * m; j += static_cast<uint32_t>(T)) {
                        sh.pair[j] = p.load_l2(&sp.cand[j]);
                    }
                    p.sync();
                }
                DETTOPK_PROF(prof_step(kProfCount1);)
                // 4. Verify.
                const uint32_t n = s.cand_n;
                if (n < static_cast<uint32_t>(k) || n > C) {
                    p.sync();  // everyone has finished reading s.cand_n, which pass 1 below resets
                    if (tid == 0) s.sample_outcome = n < static_cast<uint32_t>(k) ? kSampleHigh : kSampleFull;
                } else {
                    // 5. Three picks within the candidates, like the three passes of the row path except
                    // that candidates are counted. The first count (top 11 bits) is not taken while
                    // reading the row; the candidates are scanned once more afterwards (the histogram
                    // was cleared before the row was read). While reading the row every candidate has to
                    // take a slot and store its key and position; an atomic add on the histogram on top
                    // of that lengthens the branch, and with it the time the whole warp spends going
                    // along. Measured separately, the counting alone costs almost nothing, yet placed
                    // after taking the slot and storing the candidate it cost wide rows 2.2 ms more per
                    // layer. Counting afterwards takes one more barrier and one scan of the candidates
                    // (four or five per thread) and is still faster overall, by 2.5%. Counting is
                    // integer addition and independent of order, so the result is unchanged.
                    for (uint32_t j = static_cast<uint32_t>(tid); j < n; j += static_cast<uint32_t>(T)) {
                        p.fetch_add(&sh.hist[high_bits(sh.pair[2 * j])], 1u);
                    }
                    p.sync();
                    block_pick_bucket(p, sh.hist, kHighBuckets, sh);
                    DETTOPK_PROF(prof_step(kProfPick1);)
                    const uint32_t b0 = s.pick;
                    clear_hist(p, sh.hist, kMidBuckets);
                    p.sync();
                    for (uint32_t j = static_cast<uint32_t>(tid); j < n; j += static_cast<uint32_t>(T)) {
                        const uint32_t key = sh.pair[2 * j];
                        if (high_bits(key) == b0) p.fetch_add(&sh.hist[mid_bits(key)], 1u);
                    }
                    p.sync();
                    DETTOPK_PROF(prof_step(kProfCount2);)
                    block_pick_bucket(p, sh.hist, kMidBuckets, sh);
                    DETTOPK_PROF(prof_step(kProfPick2);)
                    const uint32_t b01 = (b0 << 11) | s.pick;  // the top 22 bits
                    const uint32_t h = sh.hist[s.pick];         // how many candidates have top 22 bits equal to b01
#ifdef __CUDA_ARCH__
                    const int L = 32;
#else
                    const int L = S < T ? S : T;
#endif
                    // Third pick: only h candidates have top 22 bits equal to b01. When h is at most the
                    // number of threads L in a warp (32 on the GPU), which on wide rows is almost always
                    // the case (usually just a few), the pick is handed to warp 0, with no clearing of
                    // the histogram, no counting by the low 10 bits and no block-wide pick:
                    //   the threads of the block scan the candidates and put these h keys into a small
                    //   table, taking slots atomically (part, one entry per thread, not used elsewhere).
                    //   Barrier;
                    //   thread i of warp 0 takes entry i of the small table (none for i >= h) and, by
                    //   shuffling within the warp, counts how many entries are greater than its own
                    //   (greater) and how many are equal to it (equal). The need-th key is the one with
                    //   greater < need <= greater + equal, and the tie quota is need - greater,
                    //   bit-identical to counting by the low 10 bits and picking. Of the threads that
                    //   qualify, the lowest one writes to s. Barrier.
                    // Example: the small table is 39 38 45 38 (h = 4) and the 3rd key is wanted. 45 has 0
                    // keys before it; 39 has 1; both 38s have 2 before them and 2 equal, and 2 < 3 <= 4,
                    // so the 3rd key is 38 with tie quota 3 - 2 = 1.
                    // This replaces clear and count (one barrier each) plus a pick (two barriers) with
                    // two barriers, and is 1.4% faster per layer on the prefill sweep. The order of the
                    // small table may differ from run to run, but "how many are greater than or equal to
                    // mine" does not depend on order and the qualifying key is unique, so the result
                    // stays deterministic. When h exceeds L, the original procedure is used.
                    if (h <= static_cast<uint32_t>(L)) {
                        const uint32_t need = s.need;
                        for (uint32_t j = static_cast<uint32_t>(tid); j < n; j += static_cast<uint32_t>(T)) {
                            const uint32_t key = sh.pair[2 * j];
                            if ((key >> 10) == b01) sh.part[p.fetch_add(&s.small_n, 1u)] = key;
                        }
                        p.sync();
                        DETTOPK_PROF(prof_step(kProfCount3);)
                        if (tid < L) {
                            const bool valid = static_cast<uint32_t>(tid) < h;
                            const uint32_t v = valid ? sh.part[tid] : 0u;
                            uint32_t greater = 0, equal = 0;
                            for (int j = 0; j < L; ++j) {
                                const uint32_t u = p.shfl(v, j);
                                if (static_cast<uint32_t>(j) < h) {
                                    greater += u > v ? 1u : 0u;
                                    equal += u == v ? 1u : 0u;
                                }
                            }
                            const bool mine = valid && greater < need && need <= greater + equal;
                            const uint32_t who = p.ballot(mine);
                            if (mine && static_cast<uint32_t>(tid) == lowest_bit(who)) {
                                s.pick = low_bits(v);
                                s.need = need - greater;
                                s.small_n = equal;
                            }
                        }
                        p.sync();
                        DETTOPK_PROF(prof_step(kProfPick3);)
                        tau_count = s.small_n;
                    } else {
                        p.sync();  // everyone has finished reading h (the histogram is cleared next)
                        clear_hist(p, sh.hist, kLowBuckets);
                        p.sync();
                        for (uint32_t j = static_cast<uint32_t>(tid); j < n; j += static_cast<uint32_t>(T)) {
                            const uint32_t key = sh.pair[2 * j];
                            if ((key >> 10) == b01) p.fetch_add(&sh.hist[low_bits(key)], 1u);
                        }
                        p.sync();
                        DETTOPK_PROF(prof_step(kProfCount3);)
                        block_pick_bucket(p, sh.hist, kLowBuckets, sh);
                        DETTOPK_PROF(prof_step(kProfPick3);)
                        tau_count = sh.hist[s.pick];  // everything in this bucket of the low 10 bits equals tau
                    }
                    tau = (b01 << 10) | s.pick;
                    quota = s.need;
                    selected = true;
                    if (tid == 0) s.sample_outcome = kSampleOk;
                }
            }
        }
    }

    if (!selected && !owner) return RowResult::kSplitDone;  // with splitting, only one block finishes the row the single-block way
    if (!selected) {
        // ---- Selection, pass 1: count the whole row by the top 11 bits of the key, checking for NaN ----
        // How the row is read: see for_each_score.
        clear_hist(p, sh.hist, kHighBuckets);
        if (tid == 0) {
            s.nan_seen = 0;
            s.need = static_cast<uint32_t>(k);
        }
        p.sync();
        bool nan = false;
        for_each_score(tid, T, row, len, [&](float x) {
            const uint32_t bits = float_bits(x);
            if (is_nan_bits(bits)) {
                nan = true;
            } else {
                p.fetch_add(&sh.hist[high_bits(bits_to_key(bits))], 1u);
            }
        });
        if (nan) p.fetch_add(&s.nan_seen, 1u);
        p.sync();
        if (s.nan_seen != 0) return RowResult::kNan;
        DETTOPK_PROF(prof_step(kProfCount1);)
        // The pick; see block_pick_bucket. The number of candidates is known at this point: it is the
        // count of the picked bucket.
        block_pick_bucket(p, sh.hist, kHighBuckets, sh);
        const uint32_t b0 = s.pick;
        // ---- Path choice ----
        // Sampled path: see above. Rows with valid length at least sh.sample_min_len whose candidates and
        //   vote bits fit try it first and only arrive here after falling back.
        // Merged path: the candidates (key, position) and the row's vote bits all fit. Pass 2 reads the row
        //   by the segments of collect. A key whose top 11 bits are above b0 is certainly above tau, so its
        //   "above" vote is cast and recorded on the spot; keys that fall in bucket b0 are stored with
        //   their positions, and once tau is known their vote bits are set in shared memory (the vote
        //   fix-up), so collect does not read the row again. The row is read twice.
        // Candidate path: the candidate buffer, which holds keys only, is large enough. Pass 3 reads the
        //   candidate buffer and step (1) of collect reads the row. The row is read three times.
        // Row path: nothing fits. Pass 3 and step (1) of collect both read the row. The row is read four times.
        // All paths produce the same vote bits (see "Vote fix-up" below), hence the same result.
        // How collect's segments are cut (warps, segment length, words of vote bits) is recomputed at each
        // place below rather than kept across passes: the kernel is limited to 32 registers and keeping
        // them would spill.
        if (tid == 0) {
            const uint32_t c0 = sh.hist[b0];
            if (c0 <= sh.pair_capacity && collect_bits_words(len, T, S) <= static_cast<int64_t>(sh.pair_bits_capacity)) {
                s.path = kPathMerged;
            } else if (c0 <= sh.cand_capacity) {
                s.path = kPathCand;
            } else {
                s.path = kPathRow;
            }
            s.cand_n = 0;
        }
        p.sync();
        DETTOPK_PROF(prof_step(kProfPick1);)
        const uint32_t path = s.path;

        // ---- Pass 2: read the whole row again, look only at the keys in bucket b0 and count them by the
        // middle 11 bits ----
        // When they fit, these keys are copied to the candidate buffer along the way (with their positions
        // on the merged path). The slot is taken by an atomic add on a counter: who gets which slot may
        // differ from run to run, but candidates are only counted and used by the vote fix-up, both
        // independent of order, so the result stays deterministic.
        clear_hist(p, sh.hist, kMidBuckets);
        p.sync();
        if (path == kPathMerged) {
            // Walk by the segments of collect (see walk_segment), casting 4 "above" votes per step; the
            // "equal" words are recorded as 0 for now and or-ed in by the vote fix-up. The per-segment
            // counts of "above" so far include only keys whose top 11 bits are above b0, and are parked
            // in seg_e because the next two picks borrow seg_g. Positions outside the segment carry the
            // key of 0.0f and could be taken for candidates when that falls in bucket b0, so membership
            // in the segment is tested first.
            const int group = tid / S;
            const int lane = tid % S;
            const CollectSeg c = collect_segment(group, T, S, static_cast<int32_t>(unaligned_head(row, len)),
                                                 static_cast<int32_t>(collect_seg(len, (T + S - 1) / S)), len,
                                                 static_cast<int32_t>(collect_bits_words(len, T, S)));
            uint32_t g = 0;
            walk_segment(row, c, lane, [&](int32_t p0, const uint32_t key[4], int32_t w) {
                for (int j = 0; j < 4; ++j) {
                    const bool in_seg = p0 + j >= c.begin && p0 + j < c.end;
                    const uint32_t h = high_bits(key[j]);
                    const uint32_t vote_g = p.ballot(in_seg && h > b0);
                    if (lane == 0) {
                        sh.pair_bits[w + 2 * j] = vote_g;
                        sh.pair_bits[w + 2 * j + 1] = 0;
                    }
                    g += popcount32(vote_g);
                    if (in_seg && h == b0) {
                        const uint32_t slot = p.fetch_add(&s.cand_n, 1u);
                        sh.pair[2 * slot] = key[j];
                        sh.pair[2 * slot + 1] = static_cast<uint32_t>(p0 + j);
                        p.fetch_add(&sh.hist[mid_bits(key[j])], 1u);
                    }
                }
            });
            if (lane == 0) sh.seg_e[group] = g;
        } else {
            const bool use_cand = path == kPathCand;
            for_each_score(tid, T, row, len, [&](float x) {
                const uint32_t key = score_to_key(x);
                if (high_bits(key) != b0) return;
                if (use_cand) sh.cand[p.fetch_add(&s.cand_n, 1u)] = key;
                p.fetch_add(&sh.hist[mid_bits(key)], 1u);
            });
        }
        p.sync();
        DETTOPK_PROF(prof_step(kProfCount2);)
        block_pick_bucket(p, sh.hist, kMidBuckets, sh);
        DETTOPK_PROF(prof_step(kProfPick2);)
        const uint32_t b1 = s.pick;

        // ---- Pass 3: count the candidates whose middle 11 bits equal b1 by their low 10 bits ----
        // When the candidates were stored, only they are read (the merged path interleaves keys and
        // positions, so only the keys are read: every other entry). Rows whose candidates did not fit
        // (narrow-range or heavy-ties data, for example) have to read the row a third time.
        clear_hist(p, sh.hist, kLowBuckets);
        p.sync();
        if (path != kPathRow) {
            const uint32_t* keys = path == kPathMerged ? sh.pair : sh.cand;
            const int step = path == kPathMerged ? 2 : 1;
            const int64_t n = s.cand_n;
            for (int64_t j = tid; j < n; j += T) {
                const uint32_t key = keys[step * j];
                if (mid_bits(key) == b1) p.fetch_add(&sh.hist[low_bits(key)], 1u);
            }
        } else {
            for_each_score(tid, T, row, len, [&](float x) {
                const uint32_t key = score_to_key(x);
                if (high_bits(key) == b0 && mid_bits(key) == b1) p.fetch_add(&sh.hist[low_bits(key)], 1u);
            });
        }
        p.sync();
        DETTOPK_PROF(prof_step(kProfCount3);)
        block_pick_bucket(p, sh.hist, kLowBuckets, sh);
        DETTOPK_PROF(prof_step(kProfPick3);)
        // The bucket indices picked by the three passes, put together, are tau (the k-th key); need is
        // now the tie quota.
        tau = (b0 << 21) | (b1 << 10) | s.pick;
        quota = s.need;
    }
    DETTOPK_PROF(if (tid == 0 && prof) prof[kProfOutcome] = s.sample_outcome;)

    // With selection only, stop here. The two results of selection go to the first two output slots and
    // nothing else is written: this way selection can still be checked for correctness, and the time
    // measured really contains only the work of selection. On the merged path the "above" votes were
    // already cast in pass 2, so that part of collect's work is counted here.
    if (select_only) {
        if (tid == 0) {
            out[0] = static_cast<int32_t>(tau);
            if (k > 1) out[1] = static_cast<int32_t>(quota);
        }
        return RowResult::kOk;
    }

    // ---- Rank write-out on the sampled path ----
    // On the sampled path every selected position is among the candidates (selected keys are at or
    // above tau, and tau is at or above the cutoff). If the candidates equal to tau are exactly as many
    // as the tie quota (no surplus ties, which holds for the vast majority of rows), the selected ones
    // are "all candidates with key at or above tau", k in total. Then the segment-by-segment write-out
    // from the row's vote bits below is unnecessary and three steps suffice:
    //   (1) Mark: in a table B with one bit per position (borrowing pair_bits, zeroed when sampling),
    //       every selected candidate sets the bit of its position.
    //   (2) Prefix sum: B has W = ceil(len / 32) words. Thread tid takes a few consecutive words and
    //       counts the 1 bits in each; a block-wide prefix sum gives the number of 1 bits before each
    //       word, stored in the W words after B (Pf).
    //   (3) Write-out: for every selected candidate, slot = Pf[its word] + the number of 1 bits below
    //       it in that word, and its position is written to that slot.
    // Example with the 12 keys 19 38 56 9 36 45 38 3 50 35 39 30 (positions 0 to 11). For the top 5,
    // tau = 38 with tie quota 1: two candidates equal 38 (positions 1 and 6), more than the quota, so
    // this route is not taken. For the top 6, tau = 38 with tie quota 2: both 38s are selected and the
    // selected positions 1 2 5 6 8 10 are set in B; position 8 has four 1 bits before it and is written
    // to slot 4. The result is bit-identical to the route below: the selected set is the same, and the
    // slot is "the number of selected positions to the left" either way. Bits are set with an atomic
    // or, independent of order, and the prefix sum depends on B alone.
    // What it saves: computing where the vote bits are from the position in the fix-up (collect_slot),
    // the per-segment counts, and the round-by-round regrouping and counting of the row's vote bits in
    // write-out. It is 8.4% faster per layer on the prefill sweep.
    // B and Pf take 2W words; pair_bits has collect_bits_words(len) words (at least 8 words per 128
    // positions), which is enough.
    if (s.path == kPathSampled && tau_count == quota) {
        const uint32_t n = s.cand_n;
        const int32_t W = (len + 31) / 32;
        uint32_t* const B = sh.pair_bits;
        uint32_t* const Pf = B + W;
        // (1) Mark
        for (uint32_t j = static_cast<uint32_t>(tid); j < n; j += static_cast<uint32_t>(T)) {
            if (sh.pair[2 * j] < tau) continue;
            const uint32_t pos = sh.pair[2 * j + 1];
            p.fetch_or(&B[pos >> 5], 1u << (pos & 31u));
        }
        p.sync();
        DETTOPK_PROF(prof_step(kProfFix);)
        // (2) Prefix sum: count my own words, prefix-sum inside the warp by doubling and put each warp's
        //     total in seg_g (unused at this point); after the barrier, adding the totals of the warps
        //     before mine gives the number of 1 bits before my first word.
        const int32_t per = (W + T - 1) / T;
        const int32_t w0 = tid * per < W ? tid * per : W;
        const int32_t w1 = w0 + per < W ? w0 + per : W;
        uint32_t mine = 0;
        for (int32_t w = w0; w < w1; ++w) mine += popcount32(B[w]);
        const int group = tid / S, lane = tid % S;
#ifdef __CUDA_ARCH__
        const int lanes = 32;  // on the GPU every warp is full
#else
        const int lanes = (group + 1) * S <= T ? S : T - group * S;  // in the host emulation the last warp may be partial
#endif
        uint32_t inc = mine;
        for (int d = 1; d < lanes; d <<= 1) {
            const uint32_t v = p.shfl_up(inc, d);
            if (lane >= d) inc += v;
        }
        if (lane == lanes - 1) sh.seg_g[group] = inc;
        p.sync();
        uint32_t run = inc - mine;
        for (int q = 0; q < group; ++q) run += sh.seg_g[q];
        for (int32_t w = w0; w < w1; ++w) {
            Pf[w] = run;
            run += popcount32(B[w]);
        }
        p.sync();
        DETTOPK_PROF(prof_step(kProfPrefix);)
        // (3) Write-out
        for (uint32_t j = static_cast<uint32_t>(tid); j < n; j += static_cast<uint32_t>(T)) {
            if (sh.pair[2 * j] < tau) continue;
            const uint32_t pos = sh.pair[2 * j + 1];
            const uint32_t w = pos >> 5;
            out[Pf[w] + popcount32(B[w] & ((1u << (pos & 31u)) - 1u))] = static_cast<int32_t>(pos);
        }
        DETTOPK_PROF(p.sync(); prof_step(kProfWrite);)  // the closing barrier exists only in the profiling build; see the top of the file
        return RowResult::kOk;
    }

    // ---- Collect: every position works out by itself which output slot it goes to ----
    // This step cannot deal the row out round-robin to all threads: knowing "how many selected positions
    // are to my left" requires counting by contiguous segments.
    // Segments go by warp: the row is cut into as many segments as there are warps, and warp q owns
    // segment q, [begin, end). A warp walks right from the start of its segment together, looking at
    // 4 x lanes consecutive positions per step, 4 per thread (thread `lane` of the warp looks at
    // positions 4 x lane to 4 x lane + 3), read with one load4. So at each step the threads of a warp
    // read one small contiguous stretch (coalesced access).
    // At each step the warp casts 4 pairs of votes; pair j is "is the j-th key I look at above tau" and
    // "is it equal to tau". A vote is one number whose bit L is the answer of thread L of the warp.
    //
    // load4 needs 16-byte alignment, so segment boundaries are placed at aligned positions: the segment
    // length is a multiple of 4 and segment q (q >= 1) starts at head + q x segment length, where head
    // is unaligned_head (the unaligned elements at the start of the row). Segment 0 starts at 0 and
    // also covers those head elements; its steps start at head - 4 (at 0 when head is 0), and the few
    // positions this reaches back over are outside the segment, are not read and vote "no". This way
    // the 4 positions of every thread at every step start at an aligned address; the ends of a segment
    // that do not fill a quad are read one by one. The cut is defined in collect_segment.
    //
    // Votes are recorded in shared memory: one block of words per warp, 8 words (4 pairs of votes) per
    // step. On the merged and sampled paths the vote bits are in pair_bits (the merged path starts
    // recording in pass 2; the sampled path zeroes it when sampling). On the other two paths they are in
    // bits: if they fit they are recorded in step (1) and used directly at write-out; if not (the row is
    // too wide or the capacity small), write-out reads the row once more and votes again. All paths
    // produce the same vote bits, hence the same result.
    const int group = tid / S;                                     // which warp I am in
    const int lane = tid % S;                                      // which thread of the warp I am
    const int groups = (T + S - 1) / S;                            // number of warps
    const int32_t head = static_cast<int32_t>(unaligned_head(row, len));
    const int32_t seg = static_cast<int32_t>(collect_seg(len, groups));
    const int64_t words64 = collect_bits_words(len, T, S);
    const uint32_t path = s.path;
    const bool pairs = path == kPathMerged || path == kPathSampled;  // vote bits in pair_bits, candidates with positions in pair
    const bool use_bits = pairs || words64 <= static_cast<int64_t>(sh.bits_capacity);
    const int32_t words = use_bits ? static_cast<int32_t>(words64) : 0;  // used only when the vote bits fit, and then it is below 2^31
    uint32_t* const bits = pairs ? sh.pair_bits : sh.bits;
    const CollectSeg c = collect_segment(group, T, S, head, seg, len, words);
    uint32_t g = 0, e = 0;

    if (pairs) {
        // ---- Vote fix-up (merged and sampled paths): set the candidates' vote bits and the per-segment
        // counts ----
        // Merged path: when pass 2 voted, keys whose top 11 bits are above b0 already got their "above"
        // vote and keys below b0 have "no" for both votes, all of which is already right; only the
        // candidates (top 11 bits equal to b0) had both votes recorded as "no" for the time being.
        // Sampled path: all votes are "no" (zeroed when sampling) and the candidates are everything at or
        // above the cutoff.
        // Now that tau is known: a candidate whose key is above tau gets its "above" bit set and adds 1 to
        // its segment's count of above; one equal to tau gets its "equal" bit set and adds 1 to its
        // segment's count of equal. All keys equal to tau are among the candidates (merged path: their top
        // 11 bits are necessarily b0; sampled path: verification guarantees tau is at or above the cutoff).
        // The fixed-up vote bits are bit-identical to what the other two paths cast in step (1), and so are
        // the per-segment counts. Where the vote bits are is computed from the position (collect_slot), and
        // the results of atomic or and atomic add do not depend on the order of the candidates, so
        // everything stays deterministic.
        // First the per-segment counts are put in place: the merged path moves what pass 2 counted from
        // seg_e back to seg_g (the picks are done borrowing seg_g) and the sampled path starts from 0.
        // seg_e is cleared, and the fix-up starts after a barrier.
        const bool merged = path == kPathMerged;
        for (int q = tid; q < groups; q += T) {
            sh.seg_g[q] = merged ? sh.seg_e[q] : 0;
            sh.seg_e[q] = 0;
        }
        p.sync();
        const SegDiv d = make_seg_div(seg);
        const int32_t group_words = words / groups;
        const uint32_t n = s.cand_n;
        for (uint32_t j = static_cast<uint32_t>(tid); j < n; j += static_cast<uint32_t>(T)) {
            const uint32_t key = sh.pair[2 * j];
            if (key < tau) continue;
            uint32_t bit;
            int q;
            const int32_t wd = collect_slot(static_cast<int32_t>(sh.pair[2 * j + 1]), T, S, head, d, group_words, bit, q);
            if (key > tau) {
                p.fetch_or(&bits[wd], 1u << bit);
                p.fetch_add(&sh.seg_g[q], 1u);
            } else {
                p.fetch_or(&bits[wd + 1], 1u << bit);
                p.fetch_add(&sh.seg_e[q], 1u);
            }
        }
        p.sync();
        DETTOPK_PROF(prof_step(kProfFix);)
    } else {
        // (1) Read my own segment once, vote, count the keys above tau and equal to tau in the segment,
        //     and record the votes along the way if they fit. The threads of a warp get the same votes and
        //     the same counts; thread 0 of the warp writes them to the segment's entry.
        walk_segment(row, c, lane, [&](int32_t p0, const uint32_t key[4], int32_t w) {
            for (int j = 0; j < 4; ++j) {
                const bool in_seg = p0 + j >= c.begin && p0 + j < c.end;  // positions past either end of the segment vote "no"
                const uint32_t vote_g = p.ballot(in_seg && key[j] > tau);
                const uint32_t vote_e = p.ballot(in_seg && key[j] == tau);
                if (use_bits && lane == 0) {
                    bits[w + 2 * j] = vote_g;
                    bits[w + 2 * j + 1] = vote_e;
                }
                g += popcount32(vote_g);
                e += popcount32(vote_e);
            }
        });
        if (lane == 0) {
            sh.seg_g[group] = g;
            sh.seg_e[group] = e;
        }
        p.sync();
        DETTOPK_PROF(prof_step(kProfVote);)
    }
    // (2) Prefix sum: turn "counts per segment" into "counts to the left of each segment start". There
    //     are only a few dozen numbers, so one thread adds them up left to right.
    if (tid == 0) {
        uint32_t sum_g = 0, sum_e = 0;
        for (int q = 0; q < groups; ++q) {
            const uint32_t seg_g = sh.seg_g[q], seg_e = sh.seg_e[q];
            sh.seg_g[q] = sum_g;
            sh.seg_e[q] = sum_e;
            sum_g += seg_g;
            sum_e += seg_e;
        }
    }
    p.sync();
    DETTOPK_PROF(prof_step(kProfPrefix);)
    // (3) Write-out: every warp walks its segment from the start once more, using the recorded votes (or
    //     reading the row and voting again when none were recorded) to compute slots and write.
    //     Positions outside the segment have "no" votes and are never written. The walk is in write_segment.
    write_segment(p, row, c, lane, use_bits ? bits : nullptr, tau, quota, sh.seg_g[group], sh.seg_e[group], out);
    DETTOPK_PROF(p.sync(); prof_step(kProfWrite);)  // the closing barrier exists only in the profiling build; see the top of the file
    return RowResult::kOk;
}

}  // namespace gpu
}  // namespace dettopk
