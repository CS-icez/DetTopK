// A minimal example of calling DetTopK. It includes only the public header dettopk.h.
//
// Three rows of eight scores, top 4 of each:
//   row 0  four scores tie at 2.0 for the last three places; the smaller positions win
//   row 1  no ties
//   row 2  valid length 3 < k: all three positions are selected and the row is padded with -1. The
//          columns beyond the valid length hold NaN, which is fine: they are never read.
//
// Build and run (from src/):
//   make example       the CPU implementations only; runs on any machine
//   make example-gpu   also the GPU implementation (topk_host); needs the CUDA toolkit and a GPU
// The output is checked against the expected positions, and every implementation must produce the same
// bytes; the exit status is 0 only if all of them do.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "dettopk.h"

namespace {

constexpr int64_t kRows = 3;
constexpr int64_t kWidth = 8;
constexpr int32_t kK = 4;

const float kScores[kRows * kWidth] = {
    0.5f, 2.0f, -1.0f, 3.5f, 2.0f, 0.1f, 2.0f, 2.0f,  // row 0
    1.0f, 9.0f, 3.0f, 8.0f, 2.0f, 7.0f, 4.0f, 6.0f,   // row 1
    5.0f, 1.0f, 4.0f, NAN, NAN, NAN, NAN, NAN,        // row 2: only the first 3 columns count
};
const int32_t kValidLen[kRows] = {8, 8, 3};
const int32_t kExpected[kRows * kK] = {
    1, 3, 4, 6,   // 3.5 at 3, then three of the four 2.0s: positions 1, 4, 6 (7 loses the tie)
    1, 3, 5, 7,   // 9, 8, 7, 6
    0, 1, 2, -1,  // all valid positions, padded with -1
};

using TopkFn = dettopk::Status (*)(const float*, int64_t, int64_t, const int32_t*, int32_t, int32_t*, int64_t*);

bool run(const char* name, TopkFn topk) {
    int32_t out[kRows * kK];
    dettopk::Status status = topk(kScores, kRows, kWidth, kValidLen, kK, out, nullptr);
    if (status != dettopk::Status::kOk) {
        std::printf("%-8s failed with status %d\n", name, static_cast<int>(status));
        return false;
    }
    for (int64_t r = 0; r < kRows; ++r) {
        std::printf("%-8s row %lld:", name, static_cast<long long>(r));
        for (int32_t j = 0; j < kK; ++j) std::printf(" %d", out[r * kK + j]);
        std::printf("\n");
    }
    bool ok = std::memcmp(out, kExpected, sizeof(out)) == 0;
    if (!ok) std::printf("%-8s output differs from the expected positions\n", name);
    return ok;
}

}  // namespace

int main() {
    bool ok = run("serial", dettopk::topk_serial);
    ok = run("openmp", dettopk::topk_omp) && ok;
#ifdef DETTOPK_WITH_GPU
    // topk_host takes host arrays and copies them to the GPU and back; dettopk::topk takes device pointers
    // and has the same signature.
    ok = run("gpu", dettopk::topk_host) && ok;
#endif
    std::printf(ok ? "all outputs as expected\n" : "FAILED\n");
    return ok ? 0 : 1;
}
