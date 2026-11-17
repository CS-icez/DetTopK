// Multithreaded CPU implementation (OpenMP): parallel over rows. What happens inside a row is in
// select_row.h, the same code the serial implementation uses.
//
// Why rows can simply be processed in parallel with output bit-identical to the serial implementation:
//   - Rows are independent: row r reads only row r of the scores and writes only row r of the output.
//   - The only area that is written repeatedly, the work area cand, is private to each thread.
//   - So threads share no writable data, and which thread processes which row, and in what order,
//     cannot change the result of any row.
//
// OpenMP decides the number of threads (the environment variable OMP_NUM_THREADS, or a call to
// omp_set_num_threads by the caller beforehand).
#include <vector>

#include "select_row.h"
#include "dettopk.h"

namespace dettopk {

Status topk_omp(const float* scores, int64_t rows, int64_t width,
                const int32_t* valid_len, int32_t k, int32_t* out,
                int64_t* nan_row) {
    if (!detail::arguments_ok(rows, width, valid_len, k)) return Status::kBadArgument;

    // The smallest index among the rows that contain a NaN; equal to rows when there is none.
    // Several threads may each hit a row with a NaN. Merging them by taking the minimum makes the
    // reported row index the same as the serial implementation's.
    int64_t first_nan = rows;

#pragma omp parallel
    {
        std::vector<uint32_t> cand(static_cast<size_t>(width));  // each thread's own work area

        // dynamic: a thread finishes the rows it holds and then fetches the next batch. Rows of one call
        // cost about the same, but cores may differ in speed or be busy with other programs, so handing
        // out work on demand is safer than dividing it evenly in advance.
#pragma omp for schedule(dynamic, 8) reduction(min : first_nan)
        for (int64_t r = 0; r < rows; ++r) {
            if (!detail::process_row(scores + r * width, valid_len[r], k, out + r * k, cand.data())) {
                if (r < first_nan) first_nan = r;
            }
        }
    }

    if (first_nan < rows) {
        if (nan_row) *nan_row = first_nan;
        return Status::kNanFound;
    }
    return Status::kOk;
}

}  // namespace dettopk
