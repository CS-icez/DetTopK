// The benchmark workload: which calls each of the three groups of shapes consists of, and how the data
// of each call is generated. Shapes and values follow the default configuration of vLLM and the
// config.json of DeepSeek-V3.2.
//
// The scenario is set by the chunk size C: a block (one chunk of the prompt) has C rows and block j has
// row width W = jC. A call holds at most max_elems elements, so a block is split into calls of
// R = min(C, max_elems / W) rows each.
// Row r of a block (counted from 0) has valid length W - C + r + 1.
// Full scale: C = 8192, k = 2048, max_elems = 2^27, and one cold prefill is 20 blocks.
// Shrinking s times divides C and k by s and max_elems by s^2, keeping all ratios; shrinking is only
// for checking the driver on a small machine.
//
// Data is generated row by row: the contents of a row depend only on (seed, distribution, block, row
// within the block), not on how the block is split into calls or how many threads generate it. So every
// run has exactly the same data, and a whole group never has to be in memory at once: the data of a
// call is generated when that call comes up.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace bench {

constexpr int kLayers = 61;  // number of layers of the model; each layer selects once

// The distribution of scores. The benchmark uses three: logistic is the main one and is fitted to real
// scores (see "Two distributions that resemble real scores" below); group 2 adds two adverse cases,
// narrow-range and heavy-ties. Standard normal and Gumbel were the main ones earlier; build_plan no
// longer lists them, but their generators are kept so that earlier data and hashes stay reproducible.
// New distributions are only appended: the index enters every row's seed, so the data and hashes of
// the existing ones never change.
enum class Dist {
    kNormal,    // standard normal
    kNarrow,    // narrow-range: all in [1, 1.25), the top 11 bits of every key equal, so pass 1 of selection cannot separate them
    kTies,      // heavy-ties: only the 16 integers 0 to 15, so the k-th is bound to fall in a large tie
    kGumbel,    // resembles real scores: upper tail decays exponentially, steep below
    kLogistic,  // resembles real scores: both tails decay exponentially; the main distribution
};

inline const char* dist_name(Dist d) {
    switch (d) {
        case Dist::kNormal: return "normal";
        case Dist::kNarrow: return "narrow";
        case Dist::kTies:   return "ties";
        case Dist::kGumbel:   return "gumbel";
        case Dist::kLogistic: return "logistic";
    }
    return "?";
}

// ---------- Two distributions that resemble real scores ----------
// Basis: scores produced by the indexers of two sparse-attention models on long inputs, of which only
// the top 4096 of each row (out of about 64K) were recorded. On a log scale the top 4096 are close to a
// straight line: the upper tail decays exponentially, much heavier than a normal distribution.
// Two distributions with an exponential upper tail were fitted row by row to ranks 64 to 4096, and both
// fit equally well (the recorded data cannot tell which one the bulk of a row follows):
//   Gumbel:   x = mu - beta * ln(-ln u), concentrated bulk, steep below;
//   logistic: x = mu + beta * ln(u / (1 - u)), symmetric.
// mu and beta vary a lot from row to row (for most rows the bulk lies well below zero, for a few the
// whole row is positive), and whether the bucket of the k-th key overflows the candidate buffer depends
// on exactly this variation. So instead of one set of median parameters, 32 rows per model were picked
// at equal spacing in mu / beta from the row-by-row fits and stored in the two tables below; every row
// takes one entry from the table by its own seed.
// bench/bench_torch.py has the same two tables; change both together.
struct RowParams {
    float mu;
    float beta;
};
constexpr int kRowParams = 64;
inline constexpr RowParams kGumbelParams[kRowParams] = {
    {-9.279f, 0.7176f},
    {-9.634f, 0.8122f},
    {-9.803f, 0.8900f},
    {-11.789f, 1.1247f},
    {-9.318f, 0.9271f},
    {-9.563f, 0.9860f},
    {-8.716f, 0.9293f},
    {-10.215f, 1.1212f},
    {-11.452f, 1.2899f},
    {-10.991f, 1.2692f},
    {-8.867f, 1.0443f},
    {-9.409f, 1.1295f},
    {-12.758f, 1.5575f},
    {-14.587f, 1.8129f},
    {-12.851f, 1.6235f},
    {-9.180f, 1.1770f},
    {-9.724f, 1.2637f},
    {-9.131f, 1.2029f},
    {-12.504f, 1.6679f},
    {-16.633f, 2.2470f},
    {-8.849f, 1.2131f},
    {-14.241f, 1.9839f},
    {-11.597f, 1.6416f},
    {-10.999f, 1.5862f},
    {-13.003f, 1.9149f},
    {-12.073f, 1.8177f},
    {-10.784f, 1.6995f},
    {-9.529f, 1.6231f},
    {0.999f, 0.4114f},
    {0.990f, 0.3341f},
    {1.371f, 0.4066f},
    {1.493f, 0.3709f},
    {-127.961f, 6.9603f},
    {-144.325f, 9.0001f},
    {-170.752f, 11.2831f},
    {-126.662f, 8.8113f},
    {-155.892f, 11.3598f},
    {-68.242f, 5.1530f},
    {-121.179f, 9.4464f},
    {-123.468f, 9.9183f},
    {-115.958f, 9.5745f},
    {-108.008f, 9.1937f},
    {-71.083f, 6.2432f},
    {-93.393f, 8.4333f},
    {-54.535f, 5.0600f},
    {-153.547f, 14.6007f},
    {-112.931f, 10.9730f},
    {-59.265f, 5.8807f},
    {-106.220f, 10.7444f},
    {-63.362f, 6.5261f},
    {-102.345f, 10.7275f},
    {-59.669f, 6.3858f},
    {-22.293f, 2.4345f},
    {-53.009f, 5.9461f},
    {-49.541f, 5.7263f},
    {-80.870f, 9.7985f},
    {-69.647f, 9.0943f},
    {-66.823f, 9.5458f},
    {-75.637f, 12.1575f},
    {-36.231f, 7.3758f},
    {-33.567f, 10.8114f},
    {5.052f, 1.2703f},
    {68.586f, 1.5458f},
    {96.131f, 1.5326f},
};
inline constexpr RowParams kLogisticParams[kRowParams] = {
    {-9.240f, 0.7101f},
    {-10.505f, 0.8808f},
    {-9.755f, 0.8810f},
    {-11.729f, 1.1134f},
    {-9.269f, 0.9179f},
    {-8.776f, 0.9005f},
    {-12.445f, 1.3209f},
    {-10.606f, 1.1589f},
    {-11.384f, 1.2772f},
    {-8.463f, 0.9732f},
    {-9.070f, 1.0639f},
    {-9.346f, 1.1175f},
    {-14.577f, 1.7728f},
    {-12.948f, 1.6032f},
    {-13.322f, 1.6771f},
    {-8.058f, 1.0294f},
    {-9.144f, 1.1843f},
    {-9.065f, 1.1903f},
    {-12.457f, 1.6565f},
    {-16.509f, 2.2234f},
    {-8.807f, 1.2039f},
    {-8.095f, 1.1246f},
    {-8.765f, 1.2377f},
    {-10.333f, 1.4863f},
    {-13.989f, 2.0554f},
    {-12.090f, 1.8165f},
    {-13.641f, 2.1459f},
    {-11.058f, 1.8816f},
    {1.020f, 0.4075f},
    {1.007f, 0.3310f},
    {1.392f, 0.4027f},
    {1.513f, 0.3673f},
    {-127.588f, 6.8899f},
    {-143.843f, 8.9092f},
    {-170.150f, 11.1698f},
    {-126.183f, 8.7201f},
    {-155.277f, 11.2433f},
    {-86.985f, 6.5277f},
    {-97.969f, 7.5915f},
    {-93.562f, 7.4720f},
    {-90.496f, 7.4280f},
    {-34.213f, 2.8957f},
    {-90.947f, 7.9426f},
    {-92.935f, 8.3463f},
    {-64.111f, 5.9171f},
    {-141.532f, 13.3844f},
    {-112.336f, 10.8600f},
    {-125.719f, 12.4138f},
    {-23.621f, 2.3775f},
    {-118.532f, 12.1493f},
    {-87.199f, 9.0973f},
    {-59.320f, 6.3192f},
    {-22.162f, 2.4096f},
    {-72.022f, 8.0440f},
    {-79.067f, 9.1017f},
    {-80.356f, 9.7023f},
    {-69.164f, 9.0032f},
    {-66.297f, 9.4451f},
    {-74.969f, 12.0299f},
    {-35.832f, 7.2999f},
    {-32.984f, 10.7011f},
    {5.117f, 1.2582f},
    {68.668f, 1.5303f},
    {96.213f, 1.5173f},
};

struct Scenario {
    int64_t chunk;      // C: rows per block, and also the step of the row width
    int32_t k;
    int64_t max_elems;  // maximum number of elements per call
    int blocks;         // number of blocks in one cold prefill
};

inline Scenario make_scenario(int scale) {
    return {8192 / scale, static_cast<int32_t>(2048 / scale),
            (int64_t{1} << 27) / scale / scale, 20};
}

// One top-k call.
struct Call {
    int group;          // which group (1, 2, 3)
    int config;         // the unit summaries add up: a block in group 1, a shape with a distribution in groups 2 and 3
    Dist dist;
    int block;          // which block of the cold prefill it comes from (counted from 1); group 3 belongs to no block and has 0
    int call;           // which call within the block (counted from 0)
    int64_t row0;       // row index within the block of the first row of this call
    int64_t rows;
    int64_t width;
    int32_t k;
    int32_t len_first;  // valid length of the first row
    int32_t len_step;   // difference in valid length between consecutive rows: 1 in prefill, 0 in decode
};

// The calls block `block` is split into.
inline std::vector<Call> block_calls(const Scenario& sc, int group, int config, Dist dist, int block) {
    const int64_t width = sc.chunk * block;
    const int64_t per_call = sc.max_elems / width < sc.chunk ? sc.max_elems / width : sc.chunk;
    std::vector<Call> calls;
    for (int64_t row0 = 0; row0 < sc.chunk; row0 += per_call) {
        const int64_t rows = sc.chunk - row0 < per_call ? sc.chunk - row0 : per_call;
        calls.push_back({group, config, dist, block, static_cast<int>(calls.size()), row0, rows, width,
                         sc.k, static_cast<int32_t>(width - sc.chunk + row0 + 1), 1});
    }
    return calls;
}

// Lists all the calls to make. groups holds the groups to run.
inline std::vector<Call> build_plan(const Scenario& sc, const std::vector<int>& groups) {
    std::vector<Call> plan;
    int config = 0;
    for (int g : groups) {
        if (g == 1) {
            // Group 1 (prefill sweep): all blocks of one cold prefill, every call of every block; the
            // total is the main metric.
            for (int j = 1; j <= sc.blocks; ++j) {
                for (const Call& c : block_calls(sc, 1, config, Dist::kLogistic, j)) plan.push_back(c);
                ++config;
            }
        } else if (g == 2) {
            // Group 2 (shapes and distributions): two representative shapes, each the first call of
            // its block, measured with the main distribution and the two adverse ones. Block 2 is where
            // the two limits meet (any wider and the element limit cuts the number of rows); the last
            // block is the widest with the fewest rows. At full scale they are 8192x16384 and 819x163840.
            for (int j : {2, sc.blocks}) {
                for (Dist d : {Dist::kLogistic, Dist::kNarrow, Dist::kTies}) {
                    plan.push_back(block_calls(sc, 2, config++, d, j).front());
                }
            }
        } else if (g == 4) {
            // Group 4: for tuning, not part of the reported results, and run only when named. The row
            // width is fixed and only the number of rows changes, to see whether the thread count
            // should follow the number of rows or the row width. Standard normal, every row read in full.
            // The row counts are 132 x 2048 / T (T = 1024, 512, 256, 128): on an H100 with T threads
            // per block, exactly the number of rows that fills all 132 SMs with resident blocks. The
            // row widths are 16384 and 32768.
            // 792, 1584 and 3168 were added later; they are 1.5 times the three tiers above and off the
            // boundaries, so the last batch fills only half the GPU. They check whether "the smallest
            // thread block that fills the GPU" still holds (most calls of group 1 are off the boundaries).
            for (int64_t width : {2 * sc.chunk, 4 * sc.chunk}) {
                for (int64_t full : {264, 528, 792, 1056, 1584, 2112, 3168}) {
                    const int64_t rows = full / (8192 / sc.chunk) > 0 ? full / (8192 / sc.chunk) : 1;
                    plan.push_back({4, config++, Dist::kNormal, 0, 0, 0, rows, width, sc.k,
                                    static_cast<int32_t>(width), 0});
                }
            }
        } else if (g == 3) {
            // Group 3 (decode latency): the maximum row width, every row read in full, 1 and 64 rows.
            const int64_t width = sc.chunk * sc.blocks;
            for (int64_t rows : {1, 64}) {
                plan.push_back({3, config++, Dist::kLogistic, 0, 0, 0, rows, width, sc.k,
                                static_cast<int32_t>(width), 0});
            }
        }
    }
    return plan;
}

// ---------- Generating data ----------

// splitmix64: the state advances by a constant and the output is a scrambled copy of it. Fast, and its
// statistical quality is sufficient for a benchmark.
inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// The seed of one row.
inline uint64_t row_seed(uint64_t seed, Dist dist, int block, int64_t row) {
    uint64_t s = seed;
    s = splitmix64(s) ^ static_cast<uint64_t>(dist);
    s = splitmix64(s) ^ static_cast<uint64_t>(block);
    s = splitmix64(s) ^ static_cast<uint64_t>(row);
    return splitmix64(s);
}

// Fills one row: the first len elements follow the distribution and the rest are NaN.
// Nothing beyond the valid length should ever be read; NaN makes an implementation that reads it fail
// at once, rather than quietly measure a wrong time.
inline void fill_row(float* row, int64_t width, int32_t len, Dist dist, uint64_t seed) {
    uint64_t state = seed;
    switch (dist) {
        case Dist::kNormal:
            // Box-Muller: two uniform numbers become two independent standard normal numbers. u1 has 24
            // bits, so the largest magnitude is about 5.8; the k-th is near 2 or lower, and the clipped
            // tail does not affect selection.
            for (int32_t i = 0; i < len; i += 2) {
                const uint64_t x = splitmix64(state);
                const float u1 = static_cast<float>((x >> 40) + 1) * 0x1p-24f;  // (0, 1]
                const float u2 = static_cast<float>(x & 0xFFFFFF) * 0x1p-24f;   // [0, 1)
                const float radius = std::sqrt(-2.0f * std::log(u1));
                const float angle = 6.2831853f * u2;
                row[i] = radius * std::cos(angle);
                if (i + 1 < len) row[i + 1] = radius * std::sin(angle);
            }
            break;
        case Dist::kNarrow:
            // The bit pattern of 1.0 is 0x3F800000; only the low 21 bits are random, so the value is in
            // [1, 1.25) and the top 11 bits never change.
            for (int32_t i = 0; i < len; ++i) {
                const uint32_t bits = 0x3F800000u | static_cast<uint32_t>(splitmix64(state) >> 43);
                float v;
                std::memcpy(&v, &bits, sizeof v);
                row[i] = v;
            }
            break;
        case Dist::kTies:
            for (int32_t i = 0; i < len; ++i) row[i] = static_cast<float>(splitmix64(state) >> 60);
            break;
        case Dist::kGumbel:
        case Dist::kLogistic: {
            // First take this row's parameters by its seed, then generate element by element. u has 53
            // bits and lies in (0, 1); double precision avoids an infinite logarithm when u is very close to 1.
            const bool gumbel = dist == Dist::kGumbel;
            const RowParams prm = (gumbel ? kGumbelParams : kLogisticParams)[splitmix64(state) % kRowParams];
            for (int32_t i = 0; i < len; ++i) {
                const double u = (static_cast<double>(splitmix64(state) >> 11) + 0.5) * 0x1p-53;
                const double t = gumbel ? -std::log(-std::log(u)) : std::log(u / (1.0 - u));
                row[i] = static_cast<float>(prm.mu + prm.beta * t);
            }
            break;
        }
    }
    for (int64_t i = len; i < width; ++i) row[i] = std::numeric_limits<float>::quiet_NaN();
}

// Generates the input of one call: scores is rows x width and lens holds the valid length of each row.
// Rows are generated by `threads` threads.
inline void fill_call(const Call& c, uint64_t seed, float* scores, int32_t* lens, int threads) {
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < c.rows; ++i) {
        const int32_t len = c.len_first + static_cast<int32_t>(i) * c.len_step;
        lens[i] = len;
        fill_row(scores + i * c.width, c.width, len, c.dist, row_seed(seed, c.dist, c.block, c.row0 + i));
    }
}

}  // namespace bench
