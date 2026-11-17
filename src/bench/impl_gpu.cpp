// The GPU implementation in the benchmark: copy the input to the GPU (untimed), call dettopk::topk
// (timed), copy the output back (untimed). Compiled only into executables with the GPU (make gpu).
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include "impl.h"
#include "dettopk_internal.h"

namespace bench {

namespace {

class GpuImpl : public Impl {
public:
    explicit GpuImpl(const dettopk::GpuConfig& config) : config_(config) {}
    ~GpuImpl() override {
        cudaFree(scores_.ptr);
        cudaFree(len_.ptr);
        cudaFree(out_.ptr);
    }

    void load(const Input& in) override {
        // The parameters of the GPU implementation are global. One program may hold two instances with
        // different parameters (the complete one and the selection-only one); each sets its own when it runs.
        dettopk::set_gpu_config(config_);
        in_ = in;
        const size_t scores_bytes = static_cast<size_t>(in.rows * in.width) * sizeof(float);
        const size_t len_bytes = static_cast<size_t>(in.rows) * sizeof(int32_t);
        out_bytes_ = static_cast<size_t>(in.rows * in.k) * sizeof(int32_t);
        ok_ = grow(scores_, scores_bytes) && grow(len_, len_bytes) && grow(out_, out_bytes_) &&
              cudaMemcpy(scores_.ptr, in.scores, scores_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(len_.ptr, in.valid_len, len_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              mark_unwritten();
    }

    dettopk::Status run() override {
        if (!ok_) return dettopk::Status::kGpuError;
        return dettopk::topk(static_cast<const float*>(scores_.ptr), in_.rows, in_.width,
                             static_cast<const int32_t*>(len_.ptr), in_.k, static_cast<int32_t*>(out_.ptr), nullptr);
    }

    void fetch() override {
        if (!ok_) return;
        ok_ = cudaMemcpy(in_.out, out_.ptr, out_bytes_, cudaMemcpyDeviceToHost) == cudaSuccess && mark_unwritten();
    }

private:
    struct Buffer {
        void* ptr = nullptr;
        size_t bytes = 0;
    };

    // The buffers on the GPU are reallocated only when too small and reused from call to call.
    static bool grow(Buffer& b, size_t bytes) {
        if (bytes <= b.bytes) return true;
        cudaFree(b.ptr);
        b.ptr = nullptr;
        b.bytes = 0;
        if (cudaMalloc(&b.ptr, bytes) != cudaSuccess) return false;
        b.bytes = bytes;
        return true;
    }

    // Fill the output area on the GPU with an impossible value (every byte 0x80, i.e. kUnwritten). If the
    // next run fails to write its output, this is what comes back and the hash does not match.
    bool mark_unwritten() { return cudaMemset(out_.ptr, 0x80, out_bytes_) == cudaSuccess; }

    dettopk::GpuConfig config_;
    Input in_{};
    Buffer scores_, len_, out_;
    size_t out_bytes_ = 0;
    bool ok_ = false;
};

}  // namespace

std::unique_ptr<Impl> make_gpu_impl(const dettopk::GpuConfig& config) { return std::make_unique<GpuImpl>(config); }

std::string gpu_info() {
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        return "no usable GPU";
    }
    char buf[160];
    std::snprintf(buf, sizeof buf, ", sm_%d%d, %d SMs, global memory %.1f GiB, max shared memory per block %zu KB", prop.major,
                  prop.minor, prop.multiProcessorCount,
                  static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                  prop.sharedMemPerBlockOptin / 1024);
    return std::string(prop.name) + buf;
}

}  // namespace bench
