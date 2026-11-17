// The two things every implementation shares in selection (a radix select): how the 32 bits of a key
// are split into three digits, and how a bucket is picked from a histogram. The CPU implementations and
// the GPU implementation all use this one definition, so the split and the pick rule are exactly the same.
#pragma once

#include <cstdint>

#include "key.h"

namespace dettopk {
namespace detail {

// The 32 bits of a key are split into three digits: the top 11 bits, the middle 11 and the low 10.
constexpr uint32_t kHighBuckets = 1u << 11;
constexpr uint32_t kMidBuckets = 1u << 11;
constexpr uint32_t kLowBuckets = 1u << 10;
DETTOPK_HOST_DEVICE inline uint32_t high_bits(uint32_t key) { return key >> 21; }
DETTOPK_HOST_DEVICE inline uint32_t mid_bits(uint32_t key) { return (key >> 10) & (kMidBuckets - 1); }
DETTOPK_HOST_DEVICE inline uint32_t low_bits(uint32_t key) { return key & (kLowBuckets - 1); }

// Pick: accumulate counts from the largest bucket downwards, find the bucket that holds the need-th
// largest key and return its index. hist has `buckets` buckets. On return, the number of elements in
// the larger buckets has been subtracted from need, which is now the rank still to find inside this bucket.
DETTOPK_HOST_DEVICE inline uint32_t pick_bucket(const uint32_t* hist, uint32_t buckets, uint32_t& need) {
    for (uint32_t b = buckets - 1; b > 0; --b) {
        if (hist[b] >= need) return b;
        need -= hist[b];
    }
    return 0;
}

}  // namespace detail
}  // namespace dettopk
