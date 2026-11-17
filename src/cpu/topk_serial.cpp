// Serial CPU implementation: one row after another. What happens inside a row is in select_row.h.
#include <vector>

#include "select_row.h"
#include "dettopk.h"

namespace dettopk {

Status topk_serial(const float* scores, int64_t rows, int64_t width,
                   const int32_t* valid_len, int32_t k, int32_t* out,
                   int64_t* nan_row) {
    if (!detail::arguments_ok(rows, width, valid_len, k)) return Status::kBadArgument;

    std::vector<uint32_t> cand(static_cast<size_t>(width));  // work area for selection, reused by every row

    for (int64_t r = 0; r < rows; ++r) {
        if (!detail::process_row(scores + r * width, valid_len[r], k, out + r * k, cand.data())) {
            if (nan_row) *nan_row = r;
            return Status::kNanFound;
        }
    }
    return Status::kOk;
}

}  // namespace dettopk
