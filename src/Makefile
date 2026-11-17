# Usage (from src/, or from the repository root with make -C src):
#   make test    build and run the differential tests (prints only failed checks and a summary; for every
#                check: build/test_topk -v)
#   make asan    the same tests with the address and undefined-behaviour sanitizers
#   make tsan    the host-emulation rounds of the GPU implementation with the thread sanitizer (data races)
#   make bench   only build the benchmark driver build/bench_topk; how to run it is at the top of
#                bench/bench_topk.cpp
#   make bench-smoke  shrink the scale 16 times and run the three groups once, only to check that the
#                driver runs; the log goes to build/
#   make gpu     build the tests and the benchmark driver with the GPU implementation
#                (build/test_topk_gpu, build/bench_topk_gpu) without running them
#   make test-gpu  run the tests with the GPU implementation on a machine with a GPU
#   make example  build and run the minimal example examples/minimal.cpp with the CPU implementations
#   make example-gpu  the same example with the GPU implementation as well, on a machine with a GPU
#   make gpu-prof  build the profiling benchmark driver build/bench_topk_gpu_prof: the kernel records the
#                cycles of every step of a row and they are written to the log
#   make clean
#
# Directories: include/ shared headers, cpu/ the CPU implementations, gpu/ the GPU implementation,
# tests/ tests, examples/ the minimal example, bench/ the benchmark driver, logs/ raw benchmark logs, build/ build products (not in the
# repository).
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wconversion
CXXFLAGS += -fopenmp
CPPFLAGS += -Iinclude
BUILD    := build

CPU_SRCS  := cpu/topk_serial.cpp cpu/topk_omp.cpp
EMU_SRCS  := gpu/topk_gpu_emu.cpp
TEST_SRCS := tests/test_topk.cpp $(EMU_SRCS)
HDRS      := $(wildcard include/*.h) $(wildcard cpu/*.h) $(wildcard gpu/*.h)
BENCH_SRCS := bench/bench_topk.cpp
BENCH_HDRS := $(wildcard bench/*.h)

# The commit at the time the benchmark driver is built, recorded in the logs; a + is appended when there
# are uncommitted changes.
GIT_REV = $(shell r=$$(git rev-parse --short HEAD 2>/dev/null) && { git diff --quiet HEAD || r=$$r+; echo $$r; } || echo unknown)

.PHONY: test asan tsan bench bench-smoke example example-gpu gpu gpu-prof bench-cub test-gpu clean

test: $(BUILD)/test_topk
	./$(BUILD)/test_topk

asan: $(BUILD)/test_topk_asan
	./$(BUILD)/test_topk_asan

$(BUILD)/test_topk: $(TEST_SRCS) $(CPU_SRCS) $(HDRS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TEST_SRCS) $(CPU_SRCS) -o $@

$(BUILD)/test_topk_asan: $(TEST_SRCS) $(CPU_SRCS) $(HDRS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
		$(TEST_SRCS) $(CPU_SRCS) -o $@

# Thread sanitizer: runs only the host emulation of the GPU implementation. Two threads that read and
# write the same location without a barrier in between are reported here.
# setarch -R turns address randomization off for this one process: the thread sanitizer of g++ 13 fails
# to start under the randomization strength recent kernels use by default.
tsan: $(BUILD)/test_topk_tsan
	setarch $$(uname -m) -R ./$(BUILD)/test_topk_tsan emu

$(BUILD)/test_topk_tsan: $(TEST_SRCS) $(CPU_SRCS) $(HDRS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O1 -g -fsanitize=thread $(TEST_SRCS) $(CPU_SRCS) -o $@

bench: $(BUILD)/bench_topk

bench-smoke: $(BUILD)/bench_topk
	./$(BUILD)/bench_topk --scale 16 --reps 3 --out $(BUILD)/bench-smoke.csv

$(BUILD)/bench_topk: $(BENCH_SRCS) $(BENCH_HDRS) $(CPU_SRCS) $(HDRS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DBENCH_GIT_REV='"$(GIT_REV)"' -DBENCH_CXXFLAGS='"$(CXXFLAGS)"' \
		$(BENCH_SRCS) $(CPU_SRCS) -o $@

example: $(BUILD)/example
	./$(BUILD)/example

$(BUILD)/example: examples/minimal.cpp $(CPU_SRCS) $(HDRS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) examples/minimal.cpp $(CPU_SRCS) -o $@

# ---- GPU implementation: needs the CUDA toolkit. It builds on a machine without a GPU but cannot run ----
# CUDA_ARCH is the architecture of the target GPU: sm_90 for the H100. Change it for functional testing
# on another GPU, e.g. make gpu CUDA_ARCH=sm_86.
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_90
# -maxrregcount=32: at most 32 registers per thread. The registers of an SM are only enough for 2048
# threads with 32 each; if the kernel uses one more, fewer thread blocks can be resident: without this
# option nvcc 12.8 uses 34, a 512-thread block is resident 3 times per SM instead of 4, and on an H100
# everything is over 20% slower.
NVCCFLAGS ?= -O2 -std=c++17 -arch=$(CUDA_ARCH) -maxrregcount=32
GPU_LIBS  := -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart

gpu: $(BUILD)/test_topk_gpu $(BUILD)/bench_topk_gpu

test-gpu: $(BUILD)/test_topk_gpu
	./$(BUILD)/test_topk_gpu

$(BUILD)/topk_gpu.o: gpu/topk_gpu.cu $(HDRS) | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/test_topk_gpu: $(TEST_SRCS) $(CPU_SRCS) $(BUILD)/topk_gpu.o $(HDRS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DDETTOPK_WITH_GPU $(TEST_SRCS) $(CPU_SRCS) $(BUILD)/topk_gpu.o $(GPU_LIBS) -o $@

$(BUILD)/bench_topk_gpu: $(BENCH_SRCS) bench/impl_gpu.cpp $(BENCH_HDRS) $(CPU_SRCS) $(BUILD)/topk_gpu.o $(HDRS) | $(BUILD)
	$(CXX) $(CPPFLAGS) -I$(CUDA_HOME)/include $(CXXFLAGS) -DDETTOPK_WITH_GPU \
		-DBENCH_GIT_REV='"$(GIT_REV)"' -DBENCH_CXXFLAGS='"$(CXXFLAGS)"' \
		$(BENCH_SRCS) bench/impl_gpu.cpp $(CPU_SRCS) $(BUILD)/topk_gpu.o $(GPU_LIBS) -o $@

example-gpu: $(BUILD)/example_gpu
	./$(BUILD)/example_gpu

$(BUILD)/example_gpu: examples/minimal.cpp $(CPU_SRCS) $(BUILD)/topk_gpu.o $(HDRS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DDETTOPK_WITH_GPU examples/minimal.cpp $(CPU_SRCS) $(BUILD)/topk_gpu.o $(GPU_LIBS) -o $@

# Profiling build: the same code with -DDETTOPK_PROFILE. Only for looking at proportions; its times are not
# used as measurements.
gpu-prof: $(BUILD)/bench_topk_gpu_prof

$(BUILD)/topk_gpu_prof.o: gpu/topk_gpu.cu $(HDRS) | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -DDETTOPK_PROFILE -c $< -o $@

$(BUILD)/bench_topk_gpu_prof: $(BENCH_SRCS) bench/impl_gpu.cpp $(BENCH_HDRS) $(CPU_SRCS) $(BUILD)/topk_gpu_prof.o $(HDRS) | $(BUILD)
	$(CXX) $(CPPFLAGS) -I$(CUDA_HOME)/include $(CXXFLAGS) -DDETTOPK_WITH_GPU -DDETTOPK_PROFILE \
		-DBENCH_GIT_REV='"$(GIT_REV)"' -DBENCH_CXXFLAGS='"$(CXXFLAGS)"' \
		$(BENCH_SRCS) bench/impl_gpu.cpp $(CPU_SRCS) $(BUILD)/topk_gpu_prof.o $(GPU_LIBS) -o $@

# External baseline: NVIDIA CUB's batched top-k (bench/impl_cub.cu). It needs a separate download of the
# CCCL main branch, pointed to by CCCL_DIR, and only its deterministic mode on sm_90 and above is of
# interest. For example: make bench-cub CCCL_DIR=/path/to/cccl GIT_REV=...
# To only check the wrapper on a GPU below sm_90, add CUB_FLAGS=-DDETTOPK_CUB_NONDET (built as
# non-deterministic; only narrow rows can run).
CUB_FLAGS ?=
CCCL_DIR ?=
CCCL_INC = -I$(CCCL_DIR)/libcudacxx/include -I$(CCCL_DIR)/cub -I$(CCCL_DIR)/thrust
bench-cub: $(BUILD)/bench_topk_cub

$(BUILD)/impl_cub.o: bench/impl_cub.cu bench/impl.h $(HDRS) | $(BUILD)
	@test -n "$(CCCL_DIR)" || { echo "CCCL_DIR is required"; exit 1; }
	$(NVCC) $(CCCL_INC) $(CPPFLAGS) $(CUB_FLAGS) -O2 -std=c++17 -arch=$(CUDA_ARCH) --expt-relaxed-constexpr -c $< -o $@

$(BUILD)/bench_topk_cub: $(BENCH_SRCS) bench/impl_gpu.cpp $(BENCH_HDRS) $(CPU_SRCS) $(BUILD)/topk_gpu.o $(BUILD)/impl_cub.o $(HDRS) | $(BUILD)
	$(CXX) $(CPPFLAGS) -I$(CUDA_HOME)/include $(CXXFLAGS) -DDETTOPK_WITH_GPU -DDETTOPK_WITH_CUB \
		-DBENCH_GIT_REV='"$(GIT_REV)"' -DBENCH_CXXFLAGS='"$(CXXFLAGS)"' \
		$(BENCH_SRCS) bench/impl_gpu.cpp $(CPU_SRCS) $(BUILD)/topk_gpu.o $(BUILD)/impl_cub.o $(GPU_LIBS) -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
