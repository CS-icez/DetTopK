// The interface between the benchmark driver and the implementations under test.
//
// A call has three steps, of which only the middle one is timed:
//   load   hands the input of this call to the implementation. An implementation on the GPU copies the
//          data to the GPU here.
//   run    performs one top-k. This is the step that is timed; an implementation on the GPU must wait
//          for the computation on the GPU to finish before returning.
//   fetch  makes sure the result is in in.out. An implementation on the GPU copies the output back here.
// This keeps the time of moving data out of the top-k. The same input is run several times in a row.
//
// An implementation that computes directly in host memory (with the function signature the tests use,
// see include/dettopk.h) only needs to be wrapped in HostImpl.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "dettopk_internal.h"

namespace bench {

// Before every run the output area is filled with this impossible value (the same on the host and on
// the GPU); a slot that was not written comes back as it.
constexpr int32_t kUnwritten = static_cast<int32_t>(0x80808080u);

struct Input {
    const float* scores;       // rows x width
    int64_t rows;
    int64_t width;
    const int32_t* valid_len;  // rows
    int32_t k;
    int32_t* out;              // rows x k, in host memory
};

class Impl {
public:
    virtual ~Impl() = default;
    virtual void load(const Input& in) = 0;
    virtual dettopk::Status run() = 0;
    virtual void fetch() = 0;
};

// Same as TopkFn in tests/test_topk.cpp.
using TopkFn = std::function<dettopk::Status(const float*, int64_t, int64_t, const int32_t*, int32_t,
                                          int32_t*, int64_t*)>;

class HostImpl : public Impl {
public:
    explicit HostImpl(TopkFn fn) : fn_(std::move(fn)) {}
    void load(const Input& in) override { in_ = in; }
    dettopk::Status run() override {
        return fn_(in_.scores, in_.rows, in_.width, in_.valid_len, in_.k, in_.out, nullptr);
    }
    void fetch() override {}

private:
    TopkFn fn_;
    Input in_{};
};

// The implementations on the GPU and the GPU model, defined in impl_gpu.cpp and present only in
// executables built with the GPU.
std::unique_ptr<Impl> make_gpu_impl(const dettopk::GpuConfig& config);
std::string gpu_info();

// NVIDIA CUB's batched top-k, an external baseline, defined in impl_cub.cu and present only in the
// executable built by make bench-cub.
std::unique_ptr<Impl> make_cub_impl();

}  // namespace bench
