// Mapping from a score (fp32) to a key (32-bit unsigned integer). The CPU implementations and the GPU
// implementation share this one definition, so that they all compare by exactly the same rule.
//
// Why map at all: comparing the bit patterns of floats as unsigned integers gives the wrong order
// (negative numbers have larger bit patterns than positive ones, and among negatives a larger magnitude
// has a larger bit pattern). After the mapping, keys order the same way as scores, and the bucket
// counting that follows needs integer operations only.
//
// Conventions:
//   - +0 and -0 compare equal: -0 is first turned into +0, so both get the same key.
//   - +inf and -inf are ordinary values: the key of +inf is above those of all finite values, and the
//     key of -inf is below them.
//   - NaN takes no part in comparisons; the caller detects it beforehand with is_nan_bits.
#pragma once

#include <cstdint>
#include <cstring>

// These small functions are used on both the host and the GPU. The CUDA compiler has to be told that
// they are callable from both sides; other compilers need nothing.
#ifdef __CUDACC__
#define DETTOPK_HOST_DEVICE __host__ __device__
#else
#define DETTOPK_HOST_DEVICE
#endif

namespace dettopk {

// The 32-bit pattern of a float.
DETTOPK_HOST_DEVICE inline uint32_t float_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

// Whether a bit pattern is a NaN: exponent bits all ones and a nonzero mantissa.
DETTOPK_HOST_DEVICE inline bool is_nan_bits(uint32_t u) {
    return (u & 0x7FFFFFFFu) > 0x7F800000u;
}

// Bit pattern to key. Examples (score on the left, key on the right):
//   +inf -> 0xFF800000      1.0 -> 0xBF800000      +0, -0 -> 0x80000000
//   -1.0 -> 0x407FFFFF      -inf -> 0x007FFFFF
DETTOPK_HOST_DEVICE inline uint32_t bits_to_key(uint32_t u) {
    if (u == 0x80000000u) u = 0;          // fold -0 into +0
    if (u & 0x80000000u) return ~u;       // negative: flip every bit, so a larger magnitude gives a smaller key
    return u | 0x80000000u;               // non-negative: set the top bit, placing it above all negatives
}

DETTOPK_HOST_DEVICE inline uint32_t score_to_key(float x) {
    return bits_to_key(float_bits(x));
}

}  // namespace dettopk
