#!/usr/bin/env bash
# Runs a full benchmark round from scratch on a machine with a GPU: build, differential tests, the three
# groups, and every baseline. The shapes of the groups are in bench/workload.h.
#
# Usage (from src/):
#   TAG=h100 GIT_REV=<commit> CCCL_DIR=<CCCL directory> DEEPSELECT_DIR=<directory holding deep_select/> FLASHINFER=1 bash bench/run_all.sh
# To keep it running when the connection drops, start it in the background:
#   setsid nohup env TAG=... GIT_REV=... CCCL_DIR=... DEEPSELECT_DIR=... FLASHINFER=1 bash bench/run_all.sh > run_all.log 2>&1 < /dev/null &
# The last line printed is ALL DONE; any failing step stops the run and the last line is then FAILED.
#
# Environment variables:
#   TAG             required; appended to the log file names, e.g. h100
#   GIT_REV         commit recorded in the logs
#   CUDA_ARCH       architecture of the target GPU, default sm_90 (H100)
#   REPS            repetitions per call, default 5; set 1 for a rehearsal
#   CPUS            cores the benchmark is confined to (taskset syntax), default 0-31. The OpenMP
#                   implementation uses as many threads as there are cores here
#   CCCL_DIR        the CCCL directory (containing libcudacxx/include, cub/cub and thrust/thrust; for the
#                   version see the top of bench/impl_cub.cu). When given, the driver with CUB is built
#                   and cub is measured; otherwise it is skipped
#   CUB=0           still build the driver with CUB but do not measure cub. For GPUs below sm_90: CUB's
#                   deterministic mode only runs on sm_90 and above
#   DEEPSELECT_DIR  the directory holding the deep_select/ package (built by bench/build_deepselect.sh).
#                   When given, DeepSelect is measured; otherwise it is skipped
#   FLASHINFER=1    measure FlashInfer's deterministic top-k as well. It has to be installed for PYTHON
#                   first (pip install flashinfer-python); it compiles its kernels on first use, which
#                   happens in the build step here
#   FLASHINFER_TIE_BREAK  the tie_break passed to FlashInfer, default 1 (ties go to the smaller
#                   position). That mode needs 128 KB of shared memory per SM; on a GPU with less, set 0
#                   to rehearse
#   PYTHON          interpreter for bench_torch.py, default python3 (it needs torch)
#   TEST_ARGS       arguments for the differential tests, default empty (run everything). Set it to gpu to
#                   run only the real-GPU part: the CPU implementations and the host emulation do not
#                   depend on the GPU and can be tested on any machine, and the host emulation starts
#                   over a thousand threads, which is very slow on a busy host
#
# Six logs are written (logs/bench-<date>-<time>-<TAG>-<name>.csv):
#   gpu-r1, gpu-r2   DetTopK (dettopk), CUB (cub) and the read-once baseline (read1) measured in turn in
#                    one process, all three groups. r1 is the reference; r2 measures again with the
#                    order of the three reversed, to see how much two rounds differ
#   cpu-g2           group 2: the serial and OpenMP implementations, with dettopk alongside (the driver
#                    compares the outputs of the three as it goes)
#   cpu-g13          groups 1 and 3: the OpenMP implementation and dettopk. The serial implementation is too
#                    slow and runs only in group 2
#   ext-r1, ext-r2   torch.topk (sorted=True), DeepSelect (output in ascending position order) and
#                    FlashInfer (deterministic, given the valid lengths); bench_torch.py measures them
#                    twice over the same list of calls, the second time in reverse order
set -u
TAG=${TAG:?TAG is required, e.g. h100}
CUDA_ARCH=${CUDA_ARCH:-sm_90}
GIT_REV=${GIT_REV:-unknown}
REPS=${REPS:-5}
CPUS=${CPUS:-0-31}
CCCL_DIR=${CCCL_DIR:-}
CUB=${CUB:-1}
CUB_FLAGS=${CUB_FLAGS:-}
DEEPSELECT_DIR=${DEEPSELECT_DIR:-}
FLASHINFER=${FLASHINFER:-0}
FLASHINFER_TIE_BREAK=${FLASHINFER_TIE_BREAK:-1}
PYTHON=${PYTHON:-python3}
TEST_ARGS=${TEST_ARGS:-}
export PATH=/usr/local/cuda/bin:$PATH
# Tools installed next to the interpreter are found even when PYTHON is in a virtual environment that
# has not been activated: FlashInfer runs ninja to compile its kernels.
case "$PYTHON" in */*) PATH=$(dirname "$PYTHON"):$PATH ;; esac

step() { echo; echo "===== $* ($(date +%H:%M:%S)) ====="; }
fail() { echo "FAILED: $*"; exit 1; }

step "machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "no GPU visible"
nvcc --version | tail -1
grep -m1 "model name" /proc/cpuinfo
echo "cores: $(nproc --all) in total, benchmark confined to $CPUS"
uptime
"$PYTHON" -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__)" || fail "no torch in $PYTHON"

step "build"
make gpu CUDA_ARCH="$CUDA_ARCH" GIT_REV="$GIT_REV" 2>&1 | grep -E "warning|error" && fail "the build has warnings or errors"
[ -x build/bench_topk_gpu ] && [ -x build/test_topk_gpu ] || fail "the executables were not built"
DRIVER=build/bench_topk_gpu
gpu_impls=dettopk,read1
gpu_impls_rev=read1,dettopk
if [ -n "$CCCL_DIR" ]; then
    [ -d "$CCCL_DIR/cub/cub" ] || fail "no cub/cub in CCCL_DIR"
    make bench-cub CUDA_ARCH="$CUDA_ARCH" GIT_REV="$GIT_REV" CCCL_DIR="$CCCL_DIR" CUB_FLAGS="$CUB_FLAGS" 2>&1 |
        grep -E "warning|error" && fail "the build of the driver with CUB has warnings or errors"
    [ -x build/bench_topk_cub ] || fail "the driver with CUB was not built"
    DRIVER=build/bench_topk_cub
    if [ "$CUB" = 1 ]; then
        gpu_impls=dettopk,cub,read1
        gpu_impls_rev=read1,cub,dettopk
    else
        echo "skipped: the driver with CUB is built but cub is not measured (CUB=0)"
    fi
else
    echo "skipped: no CCCL_DIR, CUB is not measured"
fi
ext_impls=torch_topk_sorted
ext_impls_rev=torch_topk_sorted
if [ -n "$DEEPSELECT_DIR" ]; then
    export PYTHONPATH="$DEEPSELECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON" -c "import deep_select" 2>&1 | grep -v -i numpy
    "$PYTHON" -c "import deep_select" 2>/dev/null || fail "cannot import deep_select from DEEPSELECT_DIR"
    ext_impls=torch_topk_sorted,deepselect_pos
    ext_impls_rev=deepselect_pos,torch_topk_sorted
else
    echo "skipped: no DEEPSELECT_DIR, DeepSelect is not measured"
fi
if [ "$FLASHINFER" = 1 ]; then
    # One small call, so that FlashInfer compiles its kernels here and not inside a measurement, and a
    # mode this GPU cannot run stops the round before anything is measured.
    "$PYTHON" - "$FLASHINFER_TIE_BREAK" <<'PY' || fail "FlashInfer cannot run (is it installed? tie_break=$FLASHINFER_TIE_BREAK)"
import sys, time, torch, flashinfer
t0 = time.time()
x = torch.randn(4, 8192, device="cuda")
out = flashinfer.top_k_ragged_transform(x, torch.zeros(4, dtype=torch.int32, device="cuda"),
                                        torch.full((4,), 8192, dtype=torch.int32, device="cuda"), 2048,
                                        deterministic=True, tie_break=int(sys.argv[1]))
torch.cuda.synchronize()
want = torch.topk(x, 2048, dim=1).indices.sort(dim=1).values
assert torch.equal(out.long().sort(dim=1).values, want), "FlashInfer selected a wrong set"
print(f"flashinfer {flashinfer.__version__}: first call took {time.time() - t0:.0f} s")
PY
    ext_impls=$ext_impls,flashinfer
    ext_impls_rev=flashinfer,$ext_impls_rev
else
    echo "skipped: FLASHINFER is not 1, FlashInfer is not measured"
fi

# The driver's summary table is long; keep only the totals and the error lines here. Every measurement
# of every call is in the logs.
bench() {
    taskset -c "$CPUS" "./$DRIVER" "$@" 2>&1 | grep -E "total|error|note:|cub:|log:|must not"
    [ "${PIPESTATUS[0]}" = 0 ] || fail "$DRIVER $*"
}

# Correctness: the kernel built for this GPU produces output bit-identical to the serial implementation.
step "differential tests"
# shellcheck disable=SC2086
taskset -c "$CPUS" ./build/test_topk_gpu $TEST_ARGS | tee build/test_topk_gpu.log
[ "${PIPESTATUS[0]}" = 0 ] || fail "the differential tests failed"
# An executable built without the GPU implementation also reports success: it just has no GPU checks.
grep -q "^GPU, default parameters: all" build/test_topk_gpu.log || fail "the tests did not include the GPU implementation"

# The main measurements of the three groups: the total and per-chunk times of prefill (group 1), two
# shapes with three distributions (group 2) and the latency of decode (group 3), plus how long it takes
# to merely read every row once when launched the same way (read1), to see how far the top-k is
# from that floor.
step "three groups: dettopk, cub, read-once baseline (round 1)"
bench --group 1,2,3 --impl "$gpu_impls" --reps "$REPS" --tag "$TAG-gpu-r1"

# Speedup over the CPU implementations. The serial implementation runs only in group 2.
step "group 2: serial, OpenMP, dettopk"
bench --group 2 --impl serial,omp,dettopk --reps "$REPS" --tag "$TAG-cpu-g2"

step "groups 1 and 3: OpenMP, dettopk"
bench --group 1,3 --impl omp,dettopk --reps "$REPS" --tag "$TAG-cpu-g13"

# torch.topk, DeepSelect and FlashInfer on the same GPU. Their data is generated by torch on the GPU
# with the same shapes and distributions as above; it is not bit-for-bit the same data.
step "torch.topk, DeepSelect, FlashInfer (round 1)"
"./$DRIVER" --plan > build/plan.csv || fail "cannot list the calls"
taskset -c "$CPUS" "$PYTHON" bench/bench_torch.py --plan build/plan.csv --reps "$REPS" --impls "$ext_impls" \
    --flashinfer-tie-break "$FLASHINFER_TIE_BREAK" \
    --out "logs/bench-$(date +%Y%m%d-%H%M%S)-$TAG-ext-r1.csv" || fail "bench_torch.py (round 1)"

# Measure again in reverse order: the difference between the two rounds is the run-to-run variation on
# this machine.
step "three groups: dettopk, cub, read-once baseline (round 2, reversed order)"
bench --group 1,2,3 --impl "$gpu_impls_rev" --reps "$REPS" --tag "$TAG-gpu-r2"

step "torch.topk, DeepSelect, FlashInfer (round 2, reversed order)"
taskset -c "$CPUS" "$PYTHON" bench/bench_torch.py --plan build/plan.csv --reps "$REPS" --impls "$ext_impls_rev" \
    --flashinfer-tie-break "$FLASHINFER_TIE_BREAK" \
    --out "logs/bench-$(date +%Y%m%d-%H%M%S)-$TAG-ext-r2.csv" || fail "bench_torch.py (round 2)"

step "done"
uptime
grep -H "errors" logs/bench-*-"$TAG"-*.csv
n=$(ls logs/bench-*-"$TAG"-*.csv | wc -l)
echo "$n logs"
[ "$n" = 6 ] || fail "there should be 6 logs"
echo "ALL DONE"
