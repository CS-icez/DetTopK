// Host emulation of the GPU implementation: real threads play one thread block and run exactly the same
// algorithm code as the GPU (block_row.h).
//
// Why it exists: the algorithm inside a thread block can join the differential tests on a machine with
// no GPU, and because the threads really run concurrently, errors such as a misplaced barrier or two
// threads writing the same location at the same time can be found (make tsan).
// It only cares about correctness; its speed means nothing.
//
// Differences from the GPU: there every row has its own thread block and the rows run at the same time;
// here there is a single "thread block" that processes one row after another.
// With a row split across blocks (EmuSample::split; see SplitCtx in block_row.h), this "thread block"
// plays the blocks of a row one after another, and the candidate area in global memory and the two
// counters are arrays on the host. The last block resets the two counters. Here they are checked to be
// 0 at the start of every row (they are cleared here only when the previous row had a NaN and hence no
// last block, which corresponds to the launcher zeroing them after a call with an error on the GPU),
// and kGpuError is returned if they are not.
// The order in which the blocks run is shuffled by row index and seed, so that different blocks come
// last. The blocks run one after another, so the memory ordering between blocks on the GPU (whether the
// fences are placed correctly) cannot be checked here.
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "block_row.h"
#include "dettopk_internal.h"

namespace dettopk {

namespace {

// What one warp needs for votes and shuffles: a barrier within the warp; two vote tables (one entry per
// thread) used alternately by consecutive votes; and two shuffle tables (one entry per thread) used
// alternately by consecutive shuffles.
struct EmuGroup {
    pthread_barrier_t barrier;
    uint8_t votes[2][32] = {};
    uint32_t values[2][32] = {};
};

// One thread of the thread block: barriers are pthread barriers; atomic add and atomic or are compiler
// builtins.
struct EmuThread {
    int tid_;
    int threads_;
    pthread_barrier_t* barrier_;
    int group_size_;
    EmuGroup* group_;
    mutable unsigned ballots_ = 0;  // number of votes this thread has cast
    mutable unsigned exchanges_ = 0;  // number of warp shuffles this thread has done

    int tid() const { return tid_; }
    int threads() const { return threads_; }
    void sync() const { pthread_barrier_wait(barrier_); }
    uint32_t fetch_add(uint32_t* ptr, uint32_t v) const {
        return __atomic_fetch_add(ptr, v, __ATOMIC_RELAXED);
    }
    uint32_t fetch_or(uint32_t* ptr, uint32_t v) const { return __atomic_fetch_or(ptr, v, __ATOMIC_RELAXED); }
    int group_size() const { return group_size_; }
    uint64_t clock() const {  // used only by the profiling build; time means nothing in the emulation
        return static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    // Warp vote: write my vote to my own entry, wait until the whole warp has written, then read the
    // whole table. The two tables alternate: the next vote writes to the other one, so nobody modifies
    // this one while everyone reads it; and by the time the vote after that returns to this table, all
    // threads of the warp have passed the barrier of the next vote and have therefore finished reading.
    uint32_t ballot(bool pred) const {
        uint8_t* votes = group_->votes[ballots_++ & 1];
        votes[tid_ % group_size_] = pred;
        pthread_barrier_wait(&group_->barrier);
        uint32_t mask = 0;
        for (int j = 0; j < group_size_; ++j) mask |= static_cast<uint32_t>(votes[j]) << j;
        return mask;
    }
    // Warp shuffle: write my number to my own entry, wait until the whole warp has written, then read
    // entry src. The two tables alternate for the same reason as with votes.
    uint32_t exchange(uint32_t v, int src) const {
        uint32_t* values = group_->values[exchanges_++ & 1];
        values[tid_ % group_size_] = v;
        pthread_barrier_wait(&group_->barrier);
        return values[src];
    }
    uint32_t shfl_up(uint32_t v, int d) const {
        const int lane = tid_ % group_size_;
        return exchange(v, lane >= d ? lane - d : lane);
    }
    uint32_t shfl(uint32_t v, int src) const { return exchange(v, src); }
    // Used when a row is split across blocks. In the emulation the blocks of a row run one after another
    // and each sees what the previous one wrote, so the fence does nothing and the load is a plain load.
    void fence() const {}
    uint32_t load_l2(const uint32_t* ptr) const { return *ptr; }
};

}  // namespace

Status topk_gpu_emu(const float* scores, int64_t rows, int64_t width, const int32_t* valid_len, int32_t k,
                    int32_t* out, int64_t* nan_row, int threads, int group_size, int cand_capacity,
                    int bits_capacity, int pair_capacity, bool select_only, EmuPathCount* paths, EmuSample sample) {
    if (k < 1 || rows < 0 || width < 0 || threads < 1 || cand_capacity < 0 || bits_capacity < 0 ||
        pair_capacity < 0 || sample.min_len < 0 || sample.r < 0 || sample.rmax < 0) {
        return Status::kBadArgument;
    }
    if (group_size < 1 || group_size > 32) return Status::kBadArgument;
    const int groups = (threads + group_size - 1) / group_size;

    // The shared memory of this "thread block", laid out as on the GPU (work is the work area):
    //   the other two paths: histogram and candidate buffer back to back, with the vote bits of collect
    //     sharing the same memory;
    //   merged and sampled paths: histogram, candidates (key and position interleaved, 2 x pair_capacity
    //     entries) and vote bits (bits_capacity entries) one after another.
    // work takes the largest of the three; the other areas are allocated separately at their exact size.
    // A write past the end of an area is reported by the memory check (make asan), except when it lands
    // elsewhere inside work, which is why some test configurations put the area under test right at the
    // end of work.
    const size_t hist = detail::kHighBuckets;
    const size_t hist_cand = hist + static_cast<size_t>(cand_capacity);
    const size_t merged = pair_capacity > 0 ? hist + 2 * static_cast<size_t>(pair_capacity) +
                                                  static_cast<size_t>(bits_capacity)
                                            : 0;
    std::vector<uint32_t> work(std::max({hist_cand, static_cast<size_t>(bits_capacity), merged}));
    std::vector<uint32_t> seg_g(static_cast<size_t>(groups)), seg_e(static_cast<size_t>(groups));
    std::vector<uint32_t> part(static_cast<size_t>(threads));
    gpu::BlockScalars scalars{};
    gpu::BlockShared sh{};
    sh.hist = work.data();
    sh.cand = work.data() + hist;
    sh.bits = work.data();
    sh.pair = work.data() + hist;
    sh.pair_bits = sh.pair + 2 * pair_capacity;
    sh.seg_g = seg_g.data();
    sh.seg_e = seg_e.data();
    sh.part = part.data();
    sh.s = &scalars;
    sh.cand_capacity = static_cast<uint32_t>(cand_capacity);
    sh.bits_capacity = static_cast<uint32_t>(bits_capacity);
    sh.pair_capacity = static_cast<uint32_t>(pair_capacity);
    sh.pair_bits_capacity = static_cast<uint32_t>(bits_capacity);
    // As on the GPU, the sampled path is off whenever the merged path is.
    sh.sample_min_len = pair_capacity > 0 ? static_cast<uint32_t>(sample.min_len) : 0;
    sh.sample_r_req = static_cast<uint32_t>(sample.r);
    sh.sample_rmax_req = static_cast<uint32_t>(sample.rmax);

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, static_cast<unsigned>(threads));
    // One barrier per warp, waiting for the number of threads the warp actually has (the last warp may
    // be partial).
    std::vector<EmuGroup> group_state(static_cast<size_t>(groups));
    for (int q = 0; q < groups; ++q) {
        const int lanes = (q + 1) * group_size <= threads ? group_size : threads - q * group_size;
        pthread_barrier_init(&group_state[static_cast<size_t>(q)].barrier, nullptr, static_cast<unsigned>(lanes));
    }

    // With splitting: this row's candidate area in global memory and its two counters (one set per row on
    // the GPU; here rows are processed one at a time, so one set is enough).
    // The candidate area is filled with 0xDEADBEEF at the start of every row, so a block that reads an
    // entry nobody wrote produces a wrong result. Its size is exactly 2 x capacity, so the memory check
    // reports writes past the end.
    // If the last block leaves the counters nonzero, split_left is set and the whole call returns kGpuError.
    std::vector<uint32_t> split_cand(2 * static_cast<size_t>(pair_capacity));
    uint32_t split_count[2] = {0, 0};
    bool row_nan = false;     // whether any block of this row reported a NaN (such a row has no last block to reset the counters)
    bool split_left = false;
    // The results of the whole call; only thread 0 writes them.
    int64_t first_nan = rows;
    bool bad_length = false;

    auto run_thread = [&](int tid) {
        const EmuThread p{tid, threads, &barrier, group_size, &group_state[static_cast<size_t>(tid / group_size)]};
        const int parts = sample.split > 1 ? sample.split : 1;
        for (int64_t r = 0; r < rows; ++r)
        for (int i = 0; i < parts; ++i) {
            gpu::RowResult res;
            if (parts > 1) {
                // Which block runs i-th: rotate by row index plus seed, and reverse when that sum is odd.
                // At the start of every row (before the first block runs) thread 0 checks the counters (see
                // split_left above) and fills the candidate area, then everyone passes a barrier and runs;
                // the last block of the previous row has already been waited for at the barrier below.
                const int rot = static_cast<int>((r + sample.split_seed) % parts);
                int b = (i + rot) % parts;
                if ((r + sample.split_seed) & 1) b = parts - 1 - b;
                if (i == 0 && tid == 0) {
                    if (row_nan) {
                        split_count[0] = split_count[1] = 0;
                    } else if (split_count[0] != 0 || split_count[1] != 0) {
                        split_left = true;
                    }
                    row_nan = false;
                    std::fill(split_cand.begin(), split_cand.end(), 0xDEADBEEFu);
                }
                p.sync();
                res = gpu::block_process_row<EmuThread, true>(p, scores + r * width, valid_len[r], width, k,
                                                              out + r * k, sh, select_only, nullptr,
                                                              gpu::SplitCtx{b, parts, split_cand.data(), split_count});
            } else {
                res = gpu::block_process_row(p, scores + r * width, valid_len[r], width, k, out + r * k, sh,
                                             select_only);
            }
            // With splitting, a block that does not own the row returns right after reading s.path, with no
            // barrier after it (on the GPU that block simply ends and nobody writes s again). In the
            // emulation thread 0 clears s.path below, so first wait until all threads are done with this block.
            if (parts > 1) p.sync();
            if (tid == 0) {
                if (res == gpu::RowResult::kNan && r < first_nan) first_nan = r;
                if (res == gpu::RowResult::kNan) row_nan = true;
                if (res == gpu::RowResult::kBadLength) bad_length = true;
                // Which path the row took; rows that never reached the path choice (valid length at most k,
                // or a NaN) have kPathNone. Clear it afterwards for the next row.
                if (paths && res == gpu::RowResult::kOk) {
                    if (scalars.path == gpu::kPathRow) ++paths->row;
                    if (scalars.path == gpu::kPathCand) ++paths->cand;
                    if (scalars.path == gpu::kPathMerged) ++paths->merged;
                    if (scalars.path == gpu::kPathSampled) ++paths->sampled;
                    if (scalars.sample_outcome == gpu::kSampleMany) ++paths->sample_many;
                    if (scalars.sample_outcome == gpu::kSampleHigh) ++paths->sample_high;
                    if (scalars.sample_outcome == gpu::kSampleFull) ++paths->sample_full;
                }
                scalars.path = gpu::kPathNone;
                scalars.sample_outcome = gpu::kSampleNone;
            }
            p.sync();  // the next row reuses this shared memory, so wait until every thread is done with this row
        }
    };
    std::vector<std::thread> others;
    for (int tid = 1; tid < threads; ++tid) others.emplace_back(run_thread, tid);
    run_thread(0);
    for (std::thread& t : others) t.join();
    pthread_barrier_destroy(&barrier);
    for (EmuGroup& g : group_state) pthread_barrier_destroy(&g.barrier);

    if (!row_nan && (split_count[0] != 0 || split_count[1] != 0)) split_left = true;  // the last row
    if (split_left) return Status::kGpuError;
    if (bad_length) return Status::kBadArgument;
    if (first_nan < rows) {
        if (nan_row) *nan_row = first_nan;
        return Status::kNanFound;
    }
    return Status::kOk;
}

}  // namespace dettopk
