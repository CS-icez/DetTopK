// Processing one row: shared by the serial CPU implementation and the OpenMP one. The two differ only
// in who processes which row. What happens inside a row is identical, so their outputs are bit-identical.
//
// Each row takes two phases:
//
//   Selection: find the k-th largest key, called tau. Keys above tau are certainly selected and keys
//              below tau certainly are not.
//   Collect:   scan left to right and write out the selected positions in order (a stream compaction).
//
// Selection is a radix select: it only counts and never sorts. Keys have 32 bits and are examined from
// the top in three passes of 11, 11 and 10 bits, so the first two passes have 2048 buckets each and the
// last has 1024. Each pass counts the elements of every bucket and then accumulates the counts from the
// largest bucket downwards. The first bucket at which the running total reaches the rank still being
// looked for is the one that holds the k-th key (this step is the pick). All larger buckets are selected
// whole, and the next pass looks only at the elements of this one bucket.
// A bucket is just a counter and stores no elements: finding out which elements are in a bucket means
// reading the data again.
//
// Each pass takes 11 bits rather than fewer so that few candidates remain after pass 1: the finer the
// buckets, the fewer elements in the picked one.
//
// Small example (4-bit keys, 2 bits per pass, top 4): the keys are 9 14 6 11 9 3 12 9.
//   Pass 1 buckets by the high 2 bits: bucket 11 has 2 keys (14, 12), bucket 10 has 4 (9, 11, 9, 9).
//     2 is short of 4 and 2 + 4 reaches it, so the 4th key is in bucket 10, as the 2nd inside that bucket.
//   Pass 2 looks only at the low 2 bits of the 4 keys in bucket 10: bucket 11 has 1 key (11), bucket 01
//     has 3 (the three 9s). 1 is short of 2 and 1 + 0 + 3 reaches it, so the key is in bucket 01.
//   Result: tau = 10 01 = 9. Of the three tied 9s one more gets in, and collect takes the leftmost.
#pragma once

#include <cassert>
#include <cstdint>

#include "key.h"
#include "radix.h"
#include "dettopk.h"

namespace dettopk {
namespace detail {

// Whether the arguments are valid: k is at least 1 and every valid length lies in [0, width].
inline bool arguments_ok(int64_t rows, int64_t width, const int32_t* valid_len, int32_t k) {
    if (k < 1 || rows < 0 || width < 0) return false;
    for (int64_t r = 0; r < rows; ++r) {
        if (valid_len[r] < 0 || valid_len[r] > width) return false;
    }
    return true;
}

// How a key is split into three digits (high_bits, mid_bits, low_bits) and how a bucket is picked
// (pick_bucket) are in include/radix.h.

// Processes one row whose valid length len is greater than k. cand is a work area of at least len entries.
// Returns false when the row contains a NaN.
inline bool select_row(const float* row, int32_t len, int32_t k, int32_t* out, uint32_t* cand) {
    uint32_t need = static_cast<uint32_t>(k);  // the rank still to find inside the current bucket
    uint32_t hist[kHighBuckets];               // histogram, reused by the three passes

    // ---- Selection, pass 1: count the whole row by the top 11 bits of the key, checking for NaN ----
    for (uint32_t& h : hist) h = 0;
    for (int32_t i = 0; i < len; ++i) {
        const uint32_t bits = float_bits(row[i]);
        if (is_nan_bits(bits)) return false;
        ++hist[high_bits(bits_to_key(bits))];
    }
    const uint32_t b0 = pick_bucket(hist, kHighBuckets, need);

    // ---- Pass 2: scan the whole row again, copy the keys that fall in bucket b0 into cand, and count
    // them by the middle 11 bits. Once they are copied, pass 3 only looks at cand and never touches the row.
    for (uint32_t& h : hist) h = 0;
    int32_t n = 0;  // number of elements in cand
    for (int32_t i = 0; i < len; ++i) {
        const uint32_t key = score_to_key(row[i]);
        if (high_bits(key) == b0) {
            cand[n++] = key;
            ++hist[mid_bits(key)];
        }
    }
    const uint32_t b1 = pick_bucket(hist, kMidBuckets, need);

    // ---- Pass 3: count the keys in cand whose middle 11 bits equal b1 by their low 10 bits ----
    for (uint32_t b = 0; b < kLowBuckets; ++b) hist[b] = 0;
    for (int32_t j = 0; j < n; ++j) {
        const uint32_t key = cand[j];
        if (mid_bits(key) == b1) ++hist[low_bits(key)];
    }
    const uint32_t b2 = pick_bucket(hist, kLowBuckets, need);

    // The bucket indices picked by the three passes, put together, are the k-th key.
    // need is now how many of the positions whose key equals tau still get in: the tie quota, at least 1.
    const uint32_t tau = (b0 << 21) | (b1 << 10) | b2;
    uint32_t quota = need;

    // ---- Collect: scan left to right; keys above tau all get in, keys equal to tau get in in order
    // until the tie quota runs out. Scanning left to right makes the written positions ascending and
    // makes smaller positions win ties.
    int32_t written = 0;
    for (int32_t i = 0; i < len; ++i) {
        const uint32_t key = score_to_key(row[i]);
        if (key > tau) {
            out[written++] = i;
        } else if (key == tau && quota > 0) {
            out[written++] = i;
            --quota;
        }
    }
    assert(written == k);
    (void)written;
    return true;
}

// Processes any row and writes the result to out (k entries). Returns false when the row contains a NaN.
inline bool process_row(const float* row, int32_t len, int32_t k, int32_t* out, uint32_t* cand) {
    if (len <= k) {
        // Valid length at most k: everything is selected, positions 0..len-1 followed by -1 padding.
        // The scores are not read, so NaN is not checked either.
        for (int32_t i = 0; i < len; ++i) out[i] = i;
        for (int32_t i = len; i < k; ++i) out[i] = -1;
        return true;
    }
    return select_row(row, len, k, out, cand);
}

}  // namespace detail
}  // namespace dettopk
