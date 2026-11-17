// Differential tests: compare the output of the implementation under test, bit for bit, with a slow
// reference implementation that is obviously correct.
//
// The reference implementation (oracle) uses neither the key mapping nor radix select: it stable-sorts
// by floating-point value (higher scores first, equal scores keeping their original order), takes the
// first k and sorts them by position. It shares no logic with the implementations under test, so a
// mistake in the key mapping or in the bucketing is caught.
//
// The cases cover boundaries rather than pile up volume: ties, +0 versus -0, infinities, negatives,
// subnormals, keys that differ only in low bits, all kinds of valid lengths, NaN. The random seed is
// fixed, so every run has exactly the same cases.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

#include "key.h"
#include "dettopk_internal.h"
// Sample positions of the sampled path (sample_quads, sample_quad_pos, unaligned_head): the boundary
// checks recompute the cutoff on the host.
#include "../gpu/block_row.h"

namespace {

using dettopk::Status;

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

float from_bits(uint32_t u) {
    float x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}

// Signature of an implementation under test. Every implementation runs the same set of cases.
using TopkFn = std::function<Status(const float*, int64_t, int64_t, const int32_t*, int32_t,
                                    int32_t*, int64_t*)>;

struct Case {
    std::string name;
    int64_t rows = 0;
    int64_t width = 0;
    int32_t k = 0;
    std::vector<float> scores;   // rows * width
    std::vector<int32_t> lens;   // rows
};

// ---------- Reference implementation ----------

void oracle_row(const float* row, int32_t len, int32_t k, int32_t* out) {
    const int32_t n = std::min(len, k);
    std::vector<int32_t> idx(static_cast<size_t>(len));
    std::iota(idx.begin(), idx.end(), 0);
    if (len > k) {
        std::stable_sort(idx.begin(), idx.end(),
                         [row](int32_t a, int32_t b) { return row[a] > row[b]; });
        idx.resize(static_cast<size_t>(n));
        std::sort(idx.begin(), idx.end());
    }
    for (int32_t i = 0; i < n; ++i) out[i] = idx[static_cast<size_t>(i)];
    for (int32_t i = n; i < k; ++i) out[i] = -1;
}

std::vector<int32_t> oracle(const Case& c) {
    std::vector<int32_t> out(static_cast<size_t>(c.rows * c.k));
    for (int64_t r = 0; r < c.rows; ++r) {
        oracle_row(c.scores.data() + r * c.width, c.lens[static_cast<size_t>(r)], c.k,
                   out.data() + r * c.k);
    }
    return out;
}

// ---------- Building cases ----------

// Each element is given by gen(row, column); an empty lens means every row uses the full width.
// Columns beyond the valid length are filled with NaN and +inf: an implementation that reads them is
// certain to get a wrong result.
Case make_case(std::string name, int64_t rows, int64_t width, int32_t k,
               const std::function<float(int64_t, int64_t)>& gen,
               std::vector<int32_t> lens = {}) {
    Case c;
    c.name = std::move(name);
    c.rows = rows;
    c.width = width;
    c.k = k;
    if (lens.empty()) lens.assign(static_cast<size_t>(rows), static_cast<int32_t>(width));
    c.lens = std::move(lens);
    c.scores.resize(static_cast<size_t>(rows * width));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < width; ++i) {
            float v;
            if (i < c.lens[static_cast<size_t>(r)]) {
                v = gen(r, i);
            } else {
                v = (i % 2 == 0) ? kNaN : kInf;
            }
            c.scores[static_cast<size_t>(r * width + i)] = v;
        }
    }
    return c;
}

std::vector<Case> build_cases() {
    std::vector<Case> cases;
    std::mt19937 rng(20261111);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto gauss = [&](int64_t, int64_t) { return normal(rng); };

    // 1. Random scores, several values of k
    cases.push_back(make_case("random 8x1000 k=50", 8, 1000, 50, gauss));
    cases.push_back(make_case("random 3x5000 k=2048", 3, 5000, 2048, gauss));
    cases.push_back(make_case("random k=1", 6, 777, 1, gauss));
    cases.push_back(make_case("random k=width-1", 6, 300, 299, gauss));

    // 2. Valid lengths: 0, 1, k-1, k, k+1, full
    cases.push_back(make_case("valid length boundaries", 9, 64, 16, gauss, {0, 1, 15, 16, 17, 18, 32, 63, 64}));
    // Causal shape: each row sees one more column than the previous one
    {
        const int64_t rows = 40, width = 200;
        std::vector<int32_t> lens;
        for (int64_t r = 0; r < rows; ++r) lens.push_back(static_cast<int32_t>(width - rows + r + 1));
        cases.push_back(make_case("causal valid lengths", rows, width, 64, gauss, lens));
    }
    cases.push_back(make_case("k above width", 4, 10, 16, gauss, {0, 3, 9, 10}));

    // 3. Heavy ties: scores take only the four values 0, 1, 2, 3, so the k-th is bound to fall in a tie
    for (int32_t k : {1, 499, 500, 501, 1999}) {
        std::uniform_int_distribution<int> pick(0, 3);
        cases.push_back(make_case("four-value ties k=" + std::to_string(k), 4, 2000, k,
                                  [&](int64_t, int64_t) { return static_cast<float>(pick(rng)); }));
    }
    // A whole row of one value
    cases.push_back(make_case("constant row 1.5", 2, 500, 100, [](int64_t, int64_t) { return 1.5f; }));
    cases.push_back(make_case("constant row -0", 2, 500, 100, [](int64_t, int64_t) { return -0.0f; }));
    cases.push_back(make_case("constant row -inf", 2, 500, 100, [](int64_t, int64_t) { return -kInf; }));

    // 4. +0 and -0 mixed with the k-th falling on zero: the two zeros must count as tied and be taken by position
    {
        std::uniform_int_distribution<int> pick(0, 9);
        auto zeros = [&](int64_t, int64_t) {
            const int t = pick(rng);
            if (t == 0) return 1.0f;
            if (t <= 3) return 0.0f;
            if (t <= 6) return -0.0f;
            return -1.0f;
        };
        cases.push_back(make_case("signed-zero ties k=400", 5, 1000, 400, zeros));
        cases.push_back(make_case("signed-zero ties k=150", 5, 1000, 150, zeros));
    }

    // 5. Infinities: more +inf than k, fewer than k; fewer than k finite values, so the k-th falls on -inf
    {
        std::uniform_int_distribution<int> pick(0, 9);
        auto infs = [&](int64_t, int64_t) {
            const int t = pick(rng);
            if (t <= 1) return kInf;
            if (t <= 5) return -kInf;
            return normal(rng);
        };
        cases.push_back(make_case("infinities k=100 (lands on +inf)", 4, 1000, 100, infs));
        cases.push_back(make_case("infinities k=400 (lands on a finite value)", 4, 1000, 400, infs));
        cases.push_back(make_case("infinities k=800 (lands on -inf)", 4, 1000, 800, infs));
    }

    // 6. Negatives: all negative; the k-th near zero (half positive, half negative)
    cases.push_back(make_case("all negative", 4, 1500, 200,
                              [&](int64_t, int64_t) { return normal(rng) - 10.0f; }));
    cases.push_back(make_case("k-th near zero", 4, 1500, 750, gauss));

    // 7. Subnormals (extremely small numbers) and zero
    {
        std::uniform_int_distribution<uint32_t> tiny(0, 300);
        std::uniform_int_distribution<int> sign(0, 1);
        cases.push_back(make_case("subnormals", 4, 1200, 300, [&](int64_t, int64_t) {
            const uint32_t mag = tiny(rng);
            return from_bits(mag | (sign(rng) ? 0x80000000u : 0u));
        }));
    }

    // 8. Keys that differ only in low bits: in the early passes all elements crowd into one bucket, and
    // only the later passes separate them
    {
        std::uniform_int_distribution<uint32_t> low8(0, 255), low16(0, 65535);
        cases.push_back(make_case("only the low 8 bits differ (positive)", 3, 3000, 700,
                                  [&](int64_t, int64_t) { return from_bits(0x3F800000u + low8(rng)); }));
        cases.push_back(make_case("only the low 8 bits differ (negative)", 3, 3000, 700,
                                  [&](int64_t, int64_t) { return from_bits(0xBF800000u + low8(rng)); }));
        cases.push_back(make_case("only the low 16 bits differ", 3, 3000, 700,
                                  [&](int64_t, int64_t) { return from_bits(0x3F800000u + low16(rng)); }));
        // The three passes of selection look at 11, 11 and 10 bits: make the keys differ only in the
        // bits one pass looks at, checking pass by pass
        std::uniform_int_distribution<uint32_t> low10(0, 1023), mid11(0, 2047);
        cases.push_back(make_case("only the low 10 bits differ", 3, 3000, 700,
                                  [&](int64_t, int64_t) { return from_bits(0x3F800000u + low10(rng)); }));
        cases.push_back(make_case("only the middle 11 bits differ", 3, 3000, 700, [&](int64_t, int64_t) {
            return from_bits(0x3F800000u + (mid11(rng) << 10));
        }));
        cases.push_back(make_case("only the top 11 bits differ", 3, 3000, 700, [&](int64_t, int64_t) {
            uint32_t u = mid11(rng) << 21;
            if ((u & 0x7FFFFFFFu) > 0x7F800000u) u &= 0xFF800000u;  // avoid NaN
            return from_bits(u);
        }));
    }

    // 9. Random bit patterns: covers all exponents and both signs (skipping NaN)
    {
        std::uniform_int_distribution<uint32_t> any;
        cases.push_back(make_case("random bit patterns", 20, 4000, 100, [&](int64_t, int64_t) {
            uint32_t u = any(rng);
            if ((u & 0x7FFFFFFFu) > 0x7F800000u) u &= 0xFF800000u;  // turn a NaN into the infinity of the same sign
            return from_bits(u);
        }));
    }

    // 10. Many rows of varying length (including short rows of at most k): for the multithreaded
    //     implementations. There must be enough rows for every thread to get work, and rows wide enough
    //     that the threads use their work areas at the same time
    {
        const int64_t rows = 300, width = 3000;
        std::vector<int32_t> lens;
        std::uniform_int_distribution<int32_t> any_len(0, static_cast<int32_t>(width));
        for (int64_t r = 0; r < rows; ++r) lens.push_back(r % 7 == 0 ? any_len(rng) % 80 : any_len(rng));
        cases.push_back(make_case("many rows, varying lengths", rows, width, 64, gauss, lens));
    }

    // 11. Row width not a multiple of 4: rows start 0 to 3 elements off alignment, and when pass 1 of the
    //     GPU implementation reads 4 floats at a time, the few at the start and end that do not fill a
    //     quad are read separately. The first and last 3 of each row are made the highest scores, so
    //     missing one gives a wrong result. rng is not used, so the data of the earlier cases is unchanged.
    cases.push_back(make_case("width not a multiple of 4, largest at both ends", 8, 1003, 100, [](int64_t r, int64_t i) {
        if (i < 3 || i >= 1000) return 50.0f + static_cast<float>(r) + static_cast<float>(i) * 0.001f;
        const uint32_t h = static_cast<uint32_t>(i) * 2654435761u + static_cast<uint32_t>(r) * 97u;
        return static_cast<float>(h % 100000u) / 1000.0f - 50.0f;
    }));

    return cases;
}

// ---------- Comparison ----------

int g_failed = 0;   // accumulated over all implementations
int g_checked = 0;  // number of checks the current implementation has run
int g_failed_here = 0;
bool g_verbose = false;

void report(bool ok, const std::string& name, const std::string& detail = "") {
    ++g_checked;
    if (!ok) {
        ++g_failed;
        ++g_failed_here;
    }
    if (ok && !g_verbose) return;
    std::printf("  %s  %s%s%s\n", ok ? "pass" : "FAIL", name.c_str(), detail.empty() ? "" : ": ",
                detail.c_str());
}

const int32_t kUntouched = -12345;  // the value the output area is filled with before a run

// Runs one case and compares the output with want slot by slot.
void run_case(const TopkFn& fn, const Case& c, const std::vector<int32_t>& want) {
    std::vector<int32_t> got(want.size(), kUntouched);
    int64_t nan_row = -1;
    const Status st = fn(c.scores.data(), c.rows, c.width, c.lens.data(), c.k, got.data(), &nan_row);
    if (st != Status::kOk) {
        report(false, c.name, "returned status is not kOk");
        return;
    }
    for (size_t j = 0; j < want.size(); ++j) {
        if (want[j] != got[j]) {
            char buf[160];
            std::snprintf(buf, sizeof buf, "row %lld slot %lld should be %d, got %d",
                          static_cast<long long>(j / static_cast<size_t>(c.k)),
                          static_cast<long long>(j % static_cast<size_t>(c.k)), want[j], got[j]);
            report(false, c.name, buf);
            return;
        }
    }
    report(true, c.name);
}

void run_case(const TopkFn& fn, const Case& c) { run_case(fn, c, oracle(c)); }

// What a selection-only run (for timing the two phases separately; see gpu/block_row.h) should produce:
// rows with valid length at most k have their normal output; for the other rows the first two slots are
// tau and the tie quota and the remaining slots are not written.
// tau and the quota are derived from the reference output: tau is the smallest key among the selected
// positions, and the quota is the number of selected positions whose key equals tau.
std::vector<int32_t> oracle_select_only(const Case& c) {
    std::vector<int32_t> want = oracle(c);
    for (int64_t r = 0; r < c.rows; ++r) {
        if (c.lens[static_cast<size_t>(r)] <= c.k) continue;
        int32_t* row = want.data() + r * c.k;
        uint32_t tau = UINT32_MAX, quota = 0;
        for (int32_t j = 0; j < c.k; ++j) {
            const uint32_t key = dettopk::score_to_key(c.scores[static_cast<size_t>(r * c.width + row[j])]);
            if (key < tau) {
                tau = key;
                quota = 0;
            }
            if (key == tau) ++quota;
        }
        std::fill(row, row + c.k, kUntouched);
        row[0] = static_cast<int32_t>(tau);
        if (c.k > 1) row[1] = static_cast<int32_t>(quota);
    }
    return want;
}

// The three NaN situations, and argument checks.
void run_nan_and_argument_checks(const TopkFn& fn) {
    const int64_t rows = 4, width = 100;
    const int32_t k = 10;
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> base(static_cast<size_t>(rows * width));
    for (float& v : base) v = normal(rng);
    const std::vector<int32_t> full(static_cast<size_t>(rows), static_cast<int32_t>(width));
    std::vector<int32_t> out(static_cast<size_t>(rows * k));

    // NaN inside the valid range: all three bit patterns (ordinary, sign bit set, only the lowest
    // mantissa bit set) must be detected
    for (uint32_t bits : {0x7FC00000u, 0xFFC00000u, 0x7F800001u}) {
        std::vector<float> s = base;
        s[static_cast<size_t>(2 * width + 57)] = from_bits(bits);
        int64_t nan_row = -1;
        const Status st = fn(s.data(), rows, width, full.data(), k, out.data(), &nan_row);
        char name[64];
        std::snprintf(name, sizeof name, "NaN bit pattern 0x%08X detected", bits);
        report(st == Status::kNanFound && nan_row == 2, name);
    }

    // After a call with a NaN, the same rows (without the NaN) must work as usual: after an error the GPU
    // implementation has to zero the error status on the GPU and the split counters.
    // Slot 57 above is not among the samples of the sampled path; with splitting, the block that reads it
    // does not check in, and that row has no last block to reset its counters.
    run_case(fn, Case{"normal after a call with a NaN", rows, width, k, base, full});

    // NaN in slot 3 of a row: the sampled path of the GPU implementation hits it while reading the
    // samples (which start at the first aligned position of the row)
    {
        std::vector<float> s = base;
        s[static_cast<size_t>(2 * width + 3)] = kNaN;
        int64_t nan_row = -1;
        const Status st = fn(s.data(), rows, width, full.data(), k, out.data(), &nan_row);
        report(st == Status::kNanFound && nan_row == 2, "NaN at the start of a row (among the samples) detected");
    }

    // Several rows with a NaN: the smallest row index is reported (a multithreaded implementation must
    // not report whichever it hits first)
    {
        const int64_t many = 64;
        std::vector<float> s(static_cast<size_t>(many * width));
        for (float& v : s) v = normal(rng);
        for (int64_t r : {50, 9, 33, 61}) s[static_cast<size_t>(r * width + 20)] = kNaN;
        const std::vector<int32_t> lens(static_cast<size_t>(many), static_cast<int32_t>(width));
        std::vector<int32_t> o(static_cast<size_t>(many * k));
        int64_t nan_row = -1;
        const Status st = fn(s.data(), many, width, lens.data(), k, o.data(), &nan_row);
        report(st == Status::kNanFound && nan_row == 9, "smallest row index reported when several rows have a NaN");
    }

    // NaN beyond the valid length: must not be read
    {
        std::vector<float> s = base;
        std::vector<int32_t> lens = full;
        lens[2] = 50;
        s[static_cast<size_t>(2 * width + 57)] = kNaN;
        const Status st = fn(s.data(), rows, width, lens.data(), k, out.data(), nullptr);
        report(st == Status::kOk, "NaN beyond the valid length not reported");
    }

    // Rows with valid length at most k are not checked for NaN and output 0..len-1 as usual
    {
        std::vector<float> s = base;
        std::vector<int32_t> lens = full;
        lens[1] = 10;
        s[static_cast<size_t>(1 * width + 3)] = kNaN;
        const Status st = fn(s.data(), rows, width, lens.data(), k, out.data(), nullptr);
        bool ok = st == Status::kOk;
        for (int32_t i = 0; i < k; ++i) ok = ok && out[static_cast<size_t>(1 * k + i)] == i;
        report(ok, "NaN in a short row not checked");
    }

    // Argument checks
    {
        std::vector<int32_t> lens = full;
        report(fn(base.data(), rows, width, lens.data(), 0, out.data(), nullptr) == Status::kBadArgument,
               "k=0 rejected");
        lens[3] = static_cast<int32_t>(width) + 1;
        report(fn(base.data(), rows, width, lens.data(), k, out.data(), nullptr) == Status::kBadArgument,
               "valid length above width rejected");
        lens[3] = -1;
        report(fn(base.data(), rows, width, lens.data(), k, out.data(), nullptr) == Status::kBadArgument,
               "negative valid length rejected");
    }
}

void run_all(const std::string& impl_name, const TopkFn& fn, const std::vector<Case>& cases) {
    g_checked = 0;
    g_failed_here = 0;
    if (g_verbose) std::printf("== %s ==\n", impl_name.c_str());
    for (const Case& c : cases) run_case(fn, c);
    run_nan_and_argument_checks(fn);
    if (g_failed_here) {
        std::printf("%s: %d checks, %d failed\n", impl_name.c_str(), g_checked, g_failed_here);
    } else {
        std::printf("%s: all %d checks passed\n", impl_name.c_str(), g_checked);
    }
}

void run_all_select_only(const std::string& impl_name, const TopkFn& fn, const std::vector<Case>& cases) {
    g_checked = 0;
    g_failed_here = 0;
    if (g_verbose) std::printf("== %s ==\n", impl_name.c_str());
    for (const Case& c : cases) run_case(fn, c, oracle_select_only(c));
    if (g_failed_here) {
        std::printf("%s: %d checks, %d failed\n", impl_name.c_str(), g_checked, g_failed_here);
    } else {
        std::printf("%s: all %d checks passed\n", impl_name.c_str(), g_checked);
    }
}

// The merged path's capacity exactly at c0, the count of the bucket picked by pass 1: with capacity c0
// the row takes the merged path, with c0 - 1 it does not (the candidate buffer gets c0, so it takes the
// candidate path). Both results must be right and so must the path taken: a capacity test that is off
// by one in either direction is caught.
// From each case the first row with valid length above k is taken and made a one-row case, run under
// three thread configurations.
void run_pair_boundary(const std::vector<Case>& cases) {
    g_checked = 0;
    g_failed_here = 0;
    struct Threads {
        int threads, group_size;
    };
    for (const Case& c : cases) {
        int64_t r = 0;
        while (r < c.rows && c.lens[static_cast<size_t>(r)] <= c.k) ++r;
        if (r == c.rows) continue;
        Case one;
        one.name = c.name + " (row " + std::to_string(r) + ")";
        one.rows = 1;
        one.width = c.width;
        one.k = c.k;
        one.lens = {c.lens[static_cast<size_t>(r)]};
        one.scores.assign(c.scores.begin() + r * c.width, c.scores.begin() + (r + 1) * c.width);
        // c0: the top 11 bits of tau, the k-th key, are the picked bucket; count how many keys of this
        // row fall in that bucket.
        const std::vector<int32_t> want = oracle(one);
        uint32_t tau = UINT32_MAX;
        for (int32_t pos : want) tau = std::min(tau, dettopk::score_to_key(one.scores[static_cast<size_t>(pos)]));
        int c0 = 0;
        for (int32_t i = 0; i < one.lens[0]; ++i) {
            c0 += (dettopk::score_to_key(one.scores[static_cast<size_t>(i)]) >> 21) == (tau >> 21);
        }
        for (const Threads t : {Threads{1, 32}, Threads{40, 32}, Threads{7, 2}}) {
            for (const int pairs : {c0, c0 - 1}) {
                dettopk::EmuPathCount paths;
                const TopkFn fn = [&](const float* s, int64_t rows, int64_t width, const int32_t* lens, int32_t k,
                                      int32_t* out, int64_t* nan_row) {
                    return dettopk::topk_gpu_emu(s, rows, width, lens, k, out, nan_row, t.threads, t.group_size, c0,
                                              1 << 20, pairs, false, &paths);
                };
                run_case(fn, one, want);
                const bool merged = pairs == c0;
                report(paths.merged == (merged ? 1 : 0) && paths.cand == (merged ? 0 : 1),
                       one.name + ", " + std::to_string(t.threads) + " threads, merged capacity " + std::to_string(pairs) +
                           (merged ? " = c0, merged path" : " = c0 - 1, candidate path"));
            }
        }
    }
    if (g_failed_here) {
        std::printf("merged capacity at c0: %d checks, %d failed\n", g_checked, g_failed_here);
    } else {
        std::printf("merged capacity at c0: all %d checks passed\n", g_checked);
    }
}

// The cutoff of the sampled path, recomputed on the host following "Choose the cutoff" in
// gpu/block_row.h: the smallest key of the top-11-bit bucket that holds the r-th sample; if more than
// rmax samples are at or above it, it is extended to 22 bits. many returns whether rmax is still
// exceeded after refining (in which case the sampled path falls back before reading the row).
// The sample positions come from the functions in block_row.h (there is little to get wrong in the
// positions; what is being checked is the cutoff and the verification).
uint32_t sample_threshold(const float* row, int32_t len, uint32_t r, uint32_t rmax, bool& many) {
    const int32_t head = static_cast<int32_t>(dettopk::gpu::unaligned_head(row, len));
    std::vector<uint32_t> keys;
    for (int32_t q = 0; q < dettopk::gpu::sample_quads(head, len); ++q) {
        for (int j = 0; j < 4; ++j) keys.push_back(dettopk::score_to_key(row[dettopk::gpu::sample_quad_pos(head, q) + j]));
    }
    std::vector<uint32_t> sorted = keys;
    std::sort(sorted.begin(), sorted.end(), std::greater<uint32_t>());
    r = std::min<uint32_t>(r, static_cast<uint32_t>(sorted.size()));
    rmax = std::min<uint32_t>(rmax, static_cast<uint32_t>(sorted.size()));
    const uint32_t sr = sorted[r - 1];
    auto at_least = [&](uint32_t t) {
        return static_cast<uint32_t>(std::count_if(keys.begin(), keys.end(), [t](uint32_t x) { return x >= t; }));
    };
    uint32_t t = sr & ~((1u << 21) - 1);
    many = false;
    if (at_least(t) > rmax) {
        t = sr & ~((1u << 10) - 1);
        many = at_least(t) > rmax;
    }
    return t;
}

// The verification of the sampled path exactly at its bounds: the cutoff is fixed (r is half the number
// of samples, rmax the number of samples, and the cutoff uses the top 11 bits only) and n keys are at or
// above it. With candidate capacity n the sampled path completes; with n - 1 the candidates do not fit
// and it falls back. With k = n it completes; with k = n + 1 the cutoff is too high and it falls back.
// All results must be right and so must the path taken: an off-by-one in either inequality of the
// verification is caught. From each case the first row with valid length above k is taken.
void run_sample_boundary(const std::vector<Case>& cases) {
    g_checked = 0;
    g_failed_here = 0;
    struct Threads {
        int threads, group_size;
    };
    for (const Case& c : cases) {
        int64_t r = 0;
        while (r < c.rows && c.lens[static_cast<size_t>(r)] <= c.k) ++r;
        if (r == c.rows) continue;
        Case one;
        one.name = c.name + " (row " + std::to_string(r) + ")";
        one.rows = 1;
        one.width = c.width;
        one.lens = {c.lens[static_cast<size_t>(r)]};
        one.scores.assign(c.scores.begin() + r * c.width, c.scores.begin() + (r + 1) * c.width);
        const int32_t len = one.lens[0];
        const int32_t head = static_cast<int32_t>(dettopk::gpu::unaligned_head(one.scores.data(), len));
        const int32_t m = 4 * dettopk::gpu::sample_quads(head, len);
        if (m < 2) continue;
        const dettopk::EmuSample sample{1, m / 2, m};
        bool many = false;
        const uint32_t t = sample_threshold(one.scores.data(), len, static_cast<uint32_t>(sample.r),
                                            static_cast<uint32_t>(sample.rmax), many);
        int32_t n = 0;
        for (int32_t i = 0; i < len; ++i) n += dettopk::score_to_key(one.scores[static_cast<size_t>(i)]) >= t;
        struct Try {
            int32_t k, pairs;
            bool ok;  // the sampled path completes; otherwise it falls back
            const char* what;
        };
        std::vector<Try> tries;
        // Capacity: n completes, n - 1 does not fit. The capacity must be at least k and the samples must
        // fit in the candidate area (2 x capacity >= m).
        if (n - 1 >= c.k && 2 * (n - 1) >= m) {
            tries.push_back({c.k, n, true, "capacity = n"});
            tries.push_back({c.k, n - 1, false, "capacity = n - 1, does not fit"});
        }
        // k: n completes, n + 1 makes the cutoff too high. The valid length must exceed k for the row to
        // go through selection.
        if (n + 1 < len) {
            tries.push_back({n, 1 << 16, true, "k = n"});
            tries.push_back({n + 1, 1 << 16, false, "k = n + 1, cutoff too high"});
        }
        for (const Try& tr : tries) {
            one.k = tr.k;
            const std::vector<int32_t> want = oracle(one);
            for (const Threads th : {Threads{1, 32}, Threads{40, 32}, Threads{7, 2}}) {
                dettopk::EmuPathCount paths;
                const TopkFn fn = [&](const float* s, int64_t rows, int64_t width, const int32_t* lens, int32_t k,
                                      int32_t* out, int64_t* nan_row) {
                    return dettopk::topk_gpu_emu(s, rows, width, lens, k, out, nan_row, th.threads, th.group_size,
                                              4096, 1 << 20, tr.pairs, false, &paths, sample);
                };
                run_case(fn, one, want);
                const bool ok = tr.ok ? paths.sampled == 1
                                      : paths.sampled == 0 && (tr.pairs < n ? paths.sample_full : paths.sample_high) == 1;
                report(ok, one.name + ", " + std::to_string(th.threads) + " threads, n = " + std::to_string(n) + ", " +
                               tr.what);
            }
        }
    }
    if (g_failed_here) {
        std::printf("sampled-path verification at its bounds: %d checks, %d failed\n", g_checked, g_failed_here);
    } else {
        std::printf("sampled-path verification at its bounds: all %d checks passed\n", g_checked);
    }
}

}  // namespace

// Usage: test_topk [-v] [emu | gpu]. By default only failed checks and a summary per implementation are
// printed; -v prints every check.
// emu runs only the host-emulation rounds of the GPU implementation (for make tsan: the thread sanitizer
// reports false positives on the OpenMP implementation).
// gpu runs only the real GPU implementation (for compute-sanitizer: the host emulation and the CPU
// implementations are much slower under it, and it finds nothing in them).
int main(int argc, char** argv) {
    bool only_emu = false, only_gpu = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-v") g_verbose = true;
        if (std::string(argv[i]) == "emu") only_emu = true;
        if (std::string(argv[i]) == "gpu") only_gpu = true;
    }
    const std::vector<Case> cases = build_cases();

    if (!only_emu && !only_gpu) {
        run_all("CPU serial", dettopk::topk_serial, cases);

        // The OpenMP implementation runs with several thread counts: 1 thread, one that does not divide
        // the row count, and one above the core count. Every run is compared bit for bit with the
        // reference, which also verifies that the thread count does not affect the result.
        const int max_threads = omp_get_max_threads();
        for (int threads : {1, 3, max_threads, 2 * max_threads + 1}) {
            omp_set_num_threads(threads);
            run_all("CPU OpenMP, " + std::to_string(threads) + " threads", dettopk::topk_omp, cases);
        }
        omp_set_num_threads(max_threads);
    }

    // Host emulation of the GPU implementation. Skipped with gpu (compute-sanitizer only checks the real
    // kernel, and the emulation is very slow under it).
    if (!only_gpu) {
        // The GPU algorithm, with real threads on the host emulating one thread block. The thread count,
        // the warp size and the capacities of the candidate buffer, vote bits and merged path vary:
        // 1 thread; a single, partial warp; warps of 1 thread; several warps with a partial last one
        // (7 threads at 2 per warp, 40 threads at 32 per warp); all warps full (as on the GPU).
        //
        // Which of the three paths a row takes (see "Path choice" in gpu/block_row.h) is decided by the
        // number of keys in the bucket picked by pass 1 (c0 below) and by the capacities: c0 within the
        // merged capacity with the vote bits fitting gives the merged path; otherwise c0 within the
        // candidate buffer capacity gives the candidate path; otherwise the row path.
        // Each configuration states which paths must be reached, and this is checked after the run (the
        // emulation counts the path of every row).
        //
        // The first 9 configurations have merged capacity 0 and test the other two paths, as before the
        // merged path existed. Candidate buffer: with 4096 most rows fit; with 16 and 300 some rows do
        // not fit and pass 3 reads the row instead; with 0 every row reads the row.
        // Vote bits (the comparison results recorded during collect, in words): when very large all are
        // recorded; with 0 none are and write-out reads the row again; 2048, 2348 and 6144 are exactly
        // the size of the histogram or of histogram plus candidate buffer, so the vote bits can only go
        // where selection worked. With warps of 1 or 2 threads every position needs almost two words,
        // the vote bits are squeezed to the top of the candidate buffer, and the longer rows do not fit
        // and take the read-again route; in the other configurations a position needs less than a word.
        // 10006 sits on the boundary: with 1 thread a row of valid length 5000 needs exactly 10008 words
        // (collect_bits_words), two too many to fit, while all shorter rows (4000 at most) fit. The vote
        // area is then larger than the histogram and exactly 10006 words, so a wrong fit test is
        // reported by the memory check.
        //
        // The rows of the cases are not wide and c0 is at most a little over two thousand, so with
        // candidate buffer 4096 no row takes the row path.
        // The next configurations take the merged path: with a very large capacity every row does; with
        // capacity 64 and candidate buffer 300 all three paths occur; with candidate buffer 4096 there
        // is no row path. The last of them has 10006 words of vote bits at the end of the work area: a
        // row of valid length 5000 cannot fit its vote bits and skips the merged path, and a wrong test
        // is reported by the memory check. The merged capacity exactly at c0 has its own check
        // (run_pair_boundary).
        // The last two have 20 threads per warp (52 threads as 20, 20, 12): at write-out a step regroups
        // into 3 words (2 for the last warp), 20 threads cover 6 steps per round and 2 threads idle (see
        // write_segment in block_row.h). One records votes and the other does not.
        //
        // The configurations after that try the sampled path (sample is the minimum valid length, the
        // rank taken as the cutoff and the maximum count; see "Sampled path" in block_row.h). A
        // completed sampled path counts as kSampled and each of the three fallbacks has its own bit;
        // rows that fall back go on to one of the other three paths. The rows of the cases are narrow
        // and have few samples (a few dozen to three hundred per row), so the automatic r is often the
        // last sample: most rows complete and a few have the cutoff too high (with few samples it is
        // too close to k). With r = 1 the cutoff is the bucket of the largest sample and is too high for
        // most rows. With r = rmax = 1, rows with many ties have too many samples at or above the
        // cutoff. With merged capacity 64 and a very large r the cutoff is very low: most rows with k up
        // to 64 do not fit and a few short rows complete (rows with a larger k have a capacity below k
        // and do not try the sampled path). This configuration does not require "cutoff too high": with
        // run length 8 some row happened to hit it, and with run length 32 the sample positions changed
        // and none does. "Cutoff too high" is covered by other configurations.
        // The verification exactly at its bounds has its own check (run_sample_boundary).
        enum : int {
            kRow = 1,
            kCand = 2,
            kMerged = 4,
            kSampled = 8,
            kSampleMany = 16,
            kSampleHigh = 32,
            kSampleFull = 64,
        };  // paths that must be reached
        struct EmuConfig {
            int threads;
            int group_size;
            int cand_capacity;
            int bits_capacity;
            int pair_capacity;
            int expect;  // which paths must be reached
            dettopk::EmuSample sample = {};
        };
        constexpr int kAllBits = 1 << 20;
        constexpr int kAllPairs = 1 << 16;
        for (const EmuConfig cfg : {EmuConfig{1, 32, 4096, kAllBits, 0, kCand},
                                    EmuConfig{3, 32, 4096, 0, 0, kCand},
                                    EmuConfig{5, 1, 16, kAllBits, 0, kRow | kCand},
                                    EmuConfig{7, 2, 0, 2048, 0, kRow},
                                    EmuConfig{24, 8, 300, 2348, 0, kRow | kCand},
                                    EmuConfig{40, 32, 4096, 6144, 0, kCand},
                                    EmuConfig{64, 32, 16, kAllBits, 0, kRow | kCand},
                                    EmuConfig{5, 1, 4096, 6144, 0, kCand},
                                    EmuConfig{1, 32, 0, 10006, 0, kRow},
                                    EmuConfig{1, 32, 4096, kAllBits, kAllPairs, kMerged},
                                    EmuConfig{3, 32, 0, kAllBits, kAllPairs, kMerged},
                                    EmuConfig{5, 1, 0, kAllBits, kAllPairs, kMerged},
                                    EmuConfig{7, 2, 300, kAllBits, 64, kRow | kCand | kMerged},
                                    EmuConfig{40, 32, 4096, kAllBits, 64, kCand | kMerged},
                                    EmuConfig{64, 32, 0, kAllBits, kAllPairs, kMerged},
                                    EmuConfig{24, 8, 300, kAllBits, 64, kRow | kCand | kMerged},
                                    EmuConfig{1, 32, 0, 10006, 8192, kRow | kMerged},
                                    EmuConfig{52, 20, 300, kAllBits, 64, kRow | kCand | kMerged},
                                    EmuConfig{52, 20, 4096, 0, 0, kCand},
                                    EmuConfig{40, 32, 4096, kAllBits, kAllPairs, kSampled | kMerged | kSampleHigh, {1, 0, 0}},
                                    EmuConfig{1, 32, 0, kAllBits, kAllPairs, kSampled | kMerged | kSampleHigh, {1, 0, 0}},
                                    EmuConfig{7, 2, 300, kAllBits, kAllPairs, kSampled | kMerged | kSampleHigh, {1, 1, 1 << 20}},
                                    EmuConfig{5, 1, 0, kAllBits, kAllPairs, kSampled | kMerged | kSampleMany | kSampleHigh,
                                              {1, 1, 1}},
                                    EmuConfig{24, 8, 300, kAllBits, 64,
                                              kRow | kCand | kMerged | kSampled | kSampleFull,
                                              {1, 1 << 20, 1 << 20}},
                                    EmuConfig{64, 32, 0, kAllBits, kAllPairs, kSampled | kMerged, {1000, 0, 0}},
                                    EmuConfig{52, 20, 300, kAllBits, kAllPairs, kSampled | kMerged | kSampleHigh, {1, 1, 1 << 20}}}) {
            dettopk::EmuPathCount paths;
            const TopkFn fn = [cfg, &paths](const float* s, int64_t rows, int64_t width, const int32_t* lens, int32_t k,
                                            int32_t* out, int64_t* nan_row) {
                return dettopk::topk_gpu_emu(s, rows, width, lens, k, out, nan_row, cfg.threads, cfg.group_size,
                                          cfg.cand_capacity, cfg.bits_capacity, cfg.pair_capacity, false, &paths,
                                          cfg.sample);
            };
            std::string name = "GPU host emulation, " + std::to_string(cfg.threads) + " threads, " +
                               std::to_string(cfg.group_size) + " per warp, candidate buffer " +
                               std::to_string(cfg.cand_capacity) + ", vote bits " + std::to_string(cfg.bits_capacity) +
                               ", merged " + std::to_string(cfg.pair_capacity);
            if (cfg.sample.min_len > 0) {
                name += ", sampled " + std::to_string(cfg.sample.min_len) + "/" + std::to_string(cfg.sample.r) + "/" +
                        std::to_string(cfg.sample.rmax);
            }
            run_all(name, fn, cases);
            // Check which paths were reached: every required one is (by more than 0 rows), and no row took
            // one that is not required.
            const int got = (paths.row > 0 ? kRow : 0) | (paths.cand > 0 ? kCand : 0) | (paths.merged > 0 ? kMerged : 0) |
                            (paths.sampled > 0 ? kSampled : 0) | (paths.sample_many > 0 ? kSampleMany : 0) |
                            (paths.sample_high > 0 ? kSampleHigh : 0) | (paths.sample_full > 0 ? kSampleFull : 0);
            char buf[256];
            std::snprintf(buf, sizeof buf,
                          "row path %lld rows, candidate path %lld, merged path %lld, sampled path %lld; "
                          "fallbacks: too many %lld, too high %lld, does not fit %lld",
                          static_cast<long long>(paths.row), static_cast<long long>(paths.cand),
                          static_cast<long long>(paths.merged), static_cast<long long>(paths.sampled),
                          static_cast<long long>(paths.sample_many), static_cast<long long>(paths.sample_high),
                          static_cast<long long>(paths.sample_full));
            report(got == cfg.expect, name + ", paths reached", buf);
            if (g_verbose || got != cfg.expect) std::printf("  %s\n", buf);
        }
        // Row splitting (SplitCtx in block_row.h): take the sampled-path configurations above, split
        // rows across 2, 3 and 5 blocks, and run again with several block orders (a different seed makes
        // a different block the last one). The results must be right, and the path of every row
        // (completed, the three fallbacks, and the path taken after falling back) must also be item for
        // item the same as without splitting: splitting does not change the candidate set, hence not the
        // outcome of verification either. Rows that cannot take the sampled path (valid length at most
        // k, samples not fitting) are finished by block 0, as are rows with too many samples; "cutoff
        // too high" and "does not fit" make the last block fall back. The configurations force each of
        // these.
        // The NaN checks are in run_all: when the NaN is not at a sample position only one block meets
        // it, that block does not check in, and the row has no last block.
        // The cases include row widths that are not a multiple of 4 (rows start unaligned, and the
        // elements at the start go to block 0) and stretches of less than one quad (short rows, many blocks).
        {
            struct SplitConfig {
                EmuConfig cfg;
                int split, seed;
            };
            const dettopk::EmuSample sample_auto{1, 0, 0}, sample_high{1, 1, 1 << 20}, sample_many{1, 1, 1},
                sample_full{1, 1 << 20, 1 << 20};
            for (const SplitConfig sc : {SplitConfig{{40, 32, 4096, kAllBits, kAllPairs, 0, sample_auto}, 2, 0},
                                         SplitConfig{{40, 32, 4096, kAllBits, kAllPairs, 0, sample_auto}, 5, 1},
                                         SplitConfig{{1, 32, 0, kAllBits, kAllPairs, 0, sample_auto}, 3, 2},
                                         SplitConfig{{7, 2, 300, kAllBits, kAllPairs, 0, sample_high}, 3, 0},
                                         SplitConfig{{5, 1, 0, kAllBits, kAllPairs, 0, sample_many}, 2, 1},
                                         SplitConfig{{24, 8, 300, kAllBits, 64, 0, sample_full}, 5, 2},
                                         SplitConfig{{52, 20, 300, kAllBits, kAllPairs, 0, sample_high}, 2, 3}}) {
                const EmuConfig& cfg = sc.cfg;
                // Emulation split across `parts` blocks (1 means no splitting); records the paths when
                // paths is not null
                auto emu = [&cfg, &sc](int parts, dettopk::EmuPathCount* paths) -> TopkFn {
                    dettopk::EmuSample sample = cfg.sample;
                    sample.split = parts;
                    sample.split_seed = sc.seed;
                    return [&cfg, paths, sample](const float* s, int64_t rows, int64_t width, const int32_t* lens,
                                                 int32_t k, int32_t* out, int64_t* nan_row) {
                        return dettopk::topk_gpu_emu(s, rows, width, lens, k, out, nan_row, cfg.threads, cfg.group_size,
                                                  cfg.cand_capacity, cfg.bits_capacity, cfg.pair_capacity, false, paths,
                                                  sample);
                    };
                };
                // Results: the cases plus the NaN and argument checks, all compared with the reference
                run_all("GPU host emulation, row split across " + std::to_string(sc.split) + " blocks (order " +
                            std::to_string(sc.seed) +
                            "), " + std::to_string(cfg.threads) + " threads, " + std::to_string(cfg.group_size) +
                            " per warp, merged " + std::to_string(cfg.pair_capacity) + ", sampled " +
                            std::to_string(cfg.sample.min_len) + "/" + std::to_string(cfg.sample.r) + "/" +
                            std::to_string(cfg.sample.rmax),
                        emu(sc.split, nullptr), cases);
                // Paths: run with and without splitting over all cases (the results were checked above)
                // and count the path of every row
                dettopk::EmuPathCount whole, split;
                for (const Case& c : cases) {
                    std::vector<int32_t> out(static_cast<size_t>(c.rows * c.k));
                    emu(1, &whole)(c.scores.data(), c.rows, c.width, c.lens.data(), c.k, out.data(), nullptr);
                    emu(sc.split, &split)(c.scores.data(), c.rows, c.width, c.lens.data(), c.k, out.data(), nullptr);
                }
                char buf[256];
                std::snprintf(buf, sizeof buf,
                              "sampled path completed %lld/%lld rows, fallbacks: too many %lld/%lld, "
                              "too high %lld/%lld, does not fit %lld/%lld",
                              static_cast<long long>(split.sampled), static_cast<long long>(whole.sampled),
                              static_cast<long long>(split.sample_many), static_cast<long long>(whole.sample_many),
                              static_cast<long long>(split.sample_high), static_cast<long long>(whole.sample_high),
                              static_cast<long long>(split.sample_full), static_cast<long long>(whole.sample_full));
                const bool same = split.row == whole.row && split.cand == whole.cand && split.merged == whole.merged &&
                                  split.sampled == whole.sampled && split.sample_many == whole.sample_many &&
                                  split.sample_high == whole.sample_high && split.sample_full == whole.sample_full;
                report(same && whole.sampled > 0,
                       "row split across " + std::to_string(sc.split) + " blocks, same paths as without splitting", buf);
                if (g_verbose || !same) std::printf("  %s\n", buf);
            }
            // Selection only is split as well: tau and the quota are written by the last block
            const TopkFn fn = [](const float* s, int64_t rows, int64_t width, const int32_t* lens, int32_t k,
                                 int32_t* out, int64_t* nan_row) {
                return dettopk::topk_gpu_emu(s, rows, width, lens, k, out, nan_row, 40, 32, 0, kAllBits, kAllPairs, true,
                                          nullptr, dettopk::EmuSample{1, 0, 0, 3, 1});
            };
            run_all_select_only("GPU host emulation, selection only, sampled path, row split across 3 blocks", fn, cases);
        }
        run_pair_boundary(cases);
        run_sample_boundary(cases);
        // The selection-only switch: whether tau and the quota are right. One configuration each for
        // candidates fitting and not fitting, the merged path and the sampled path.
        for (const EmuConfig cfg : {EmuConfig{3, 32, 4096, 0, 0, 0}, EmuConfig{32, 32, 0, 0, 0, 0},
                                    EmuConfig{40, 32, 0, kAllBits, kAllPairs, 0},
                                    EmuConfig{40, 32, 0, kAllBits, kAllPairs, 0, {1, 0, 0}}}) {
            const TopkFn fn = [cfg](const float* s, int64_t rows, int64_t width, const int32_t* lens, int32_t k,
                                    int32_t* out, int64_t* nan_row) {
                return dettopk::topk_gpu_emu(s, rows, width, lens, k, out, nan_row, cfg.threads, cfg.group_size,
                                          cfg.cand_capacity, cfg.bits_capacity, cfg.pair_capacity, true, nullptr,
                                          cfg.sample);
            };
            run_all_select_only("GPU host emulation, selection only, " + std::to_string(cfg.threads) +
                                    " threads, candidate buffer " +
                                    std::to_string(cfg.cand_capacity) + ", merged " +
                                    std::to_string(cfg.pair_capacity) +
                                    (cfg.sample.min_len > 0 ? ", sampled path" : ""),
                                fn, cases);
        }
    }

#ifdef DETTOPK_WITH_GPU
    // The real GPU implementation: default parameters (automatic thread count), then the smallest and the
    // largest thread block.
    if (!only_emu) {
        run_all("GPU, default parameters", dettopk::topk_host, cases);
        for (const int threads : {64, 1024}) {
            dettopk::GpuConfig fixed;
            fixed.threads = threads;
            dettopk::set_gpu_config(fixed);
            run_all("GPU, " + std::to_string(threads) + " threads", dettopk::topk_host, cases);
        }
        // With the merged path off, test the other two paths: one configuration each with the usual
        // candidate buffer, a very small one (most rows do not fit and pass 3 reads the row) and none.
        for (const int cand : {-1, 16, 0}) {
            dettopk::GpuConfig fixed;
            fixed.cand_capacity = cand;
            fixed.pair_capacity = 0;
            dettopk::set_gpu_config(fixed);
            run_all("GPU, merged path off, candidate buffer " + (cand < 0 ? std::string("auto") : std::to_string(cand)),
                    dettopk::topk_host, cases);
        }
        // A very small merged capacity: some rows take the merged path and the rest the candidate path.
        {
            dettopk::GpuConfig fixed;
            fixed.pair_capacity = 64;
            dettopk::set_gpu_config(fixed);
            run_all("GPU, merged 64", dettopk::topk_host, cases);
        }
        // Collect records no votes and write-out reads the row again (the merged path is off too then).
        {
            dettopk::GpuConfig fixed;
            fixed.bits_capacity = 0;
            dettopk::set_gpu_config(fixed);
            run_all("GPU, no vote bits", dettopk::topk_host, cases);
        }
        // Sampled path: the rows of the cases are narrower than the automatic minimum length, so none of
        // the configurations above takes it. Minimum length 1 with the automatic cutoff; the cutoff at
        // the 1st sample (too high for most rows, which fall back); merged capacity 64 with a very low
        // cutoff (rows with k up to 64 do not fit and fall back).
        for (const int which : {0, 1, 2}) {
            dettopk::GpuConfig fixed;
            fixed.sample_min_len = 1;
            if (which == 1) {
                fixed.sample_r = 1;
                fixed.sample_rmax = 1 << 20;
            }
            if (which == 2) {
                fixed.pair_capacity = 64;
                fixed.sample_r = 1 << 20;
                fixed.sample_rmax = 1 << 20;
            }
            // Blocks per row: automatic (cases with at most half as many rows as SMs are split, across up
            // to 16 blocks), none, and at most 3 (longer stretches per block).
            for (const int split : {-1, 0, 3}) {
                fixed.split_max = split;
                dettopk::set_gpu_config(fixed);
                run_all(std::string("GPU, sampled path, ") +
                            (which == 0 ? "automatic cutoff" : which == 1 ? "cutoff too high" : "does not fit") +
                            (split < 0 ? "" : split == 0 ? ", no splitting" : ", at most 3 blocks per row"),
                        dettopk::topk_host, cases);
            }
        }
        dettopk::GpuConfig select_only;
        select_only.select_only = true;
        dettopk::set_gpu_config(select_only);
        run_all_select_only("GPU, selection only", dettopk::topk_host, cases);
        select_only.sample_min_len = 1;
        dettopk::set_gpu_config(select_only);
        run_all_select_only("GPU, selection only, sampled path", dettopk::topk_host, cases);
        dettopk::set_gpu_config({});
    }
#endif

    if (g_failed) {
        std::printf("\n%d checks failed\n", g_failed);
        return 1;
    }
    std::printf("\nall passed\n");
    return 0;
}
