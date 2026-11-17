#!/usr/bin/env bash
# Builds the CUDA extension of DeepSelect (deepseek-ai/DeepSelect, commit bfa4507) for the H100 (sm_90a),
# as an external baseline for bench_torch.py.
# Upstream only builds for Blackwell; sm_90a is commented out in its setup.py (the comment says to speed
# up compilation). GPUs below sm_90 have no TMA and cannot run it.
# No package is installed (ninja is not required): nvcc compiles the .cu files in parallel, g++ compiles
# api.cpp, and they are linked by hand. The compile flags are those of its setup.py with only the target
# architecture changed.
# Usage (the torch and Python versions must match those on the machine where it will run):
#   bash build_deepselect.sh [DIR]      DeepSelect is cloned into DIR/DeepSelect (DIR defaults to the current
#                                       directory) and the product is its deep_select/ directory
# Afterwards copy the whole deep_select/ directory to the H100; with PYTHONPATH pointing at its parent
# directory, import deep_select works. That parent directory is the DEEPSELECT_DIR of run_all.sh.
# CUDA_HOME is the CUDA toolkit (default /usr/local/cuda); JOBS is the number of parallel nvcc jobs.
set -eu
CUDA=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA/bin:$PATH
JOBS=${JOBS:-24}
DIR=${1:-$PWD}
mkdir -p "$DIR" && cd "$DIR"
if [ ! -d DeepSelect ]; then git clone -q https://github.com/deepseek-ai/DeepSelect.git; fi
cd DeepSelect
git checkout -q bfa4507
if [ ! -f csrc/3rdparty/cutlass/include/cute/tensor.hpp ]; then
    git submodule update --init --depth 1 csrc/3rdparty/cutlass || git submodule update --init csrc/3rdparty/cutlass
fi
echo "cutlass: $(git -C csrc/3rdparty/cutlass rev-parse --short HEAD)"

PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])" 2>/dev/null)
TORCHINC=$(python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I'+p for p in include_paths()))" 2>/dev/null)
TORCHLIB=$(python3 -c "from torch.utils.cpp_extension import library_paths; print(library_paths()[0])" 2>/dev/null)
ABI=$(python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))" 2>/dev/null)
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
INC="-Icsrc -Icsrc/3rdparty/cutlass/include -Icsrc/3rdparty/kerutils/include -I$CUDA/targets/x86_64-linux/include/cccl"
NVFLAGS="-O3 -std=c++20 -DDEEP_SELECT_IS_BUILD_ON_CUDA -Wno-deprecated-declarations
 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__
 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --ftz=false
 --ptxas-options=-v,--register-usage-level=10,--warn-on-spills -lineinfo
 -gencode arch=compute_90a,code=sm_90a -Xcompiler -fPIC"

# The list of source files is extracted from setup.py
python3 - > build90.list <<'EOF'
import re
s = open("setup.py").read()
block = s[s.index("CUDA_SOURCES = ["):s.index("]", s.index("CUDA_SOURCES = ["))]
for f in re.findall(r'"(csrc/[^"]+\.cu)"', block):
    print(f)
EOF
echo ".cu files to compile: $(wc -l < build90.list)"
mkdir -p build90
export NVFLAGS INC
start=$(date +%s)
xargs -P "$JOBS" -I{} sh -c 'o=build90/$(basename {} .cu).o; [ -f "$o" ] || nvcc $NVFLAGS $INC -c {} -o "$o" > "$o.log" 2>&1 || { echo "failed: {}"; tail -20 "$o.log"; exit 255; }' < build90.list
echo "nvcc took $(( $(date +%s) - start )) seconds"
grep -h -E "bytes spill|stack frame" build90/*.log | sort | uniq -c | sort -rn | head -5

g++ -O3 -std=c++20 -DNDEBUG -Wno-deprecated-declarations -DKERUTILS_IS_BUILD_ON_CUDA -DDEEP_SELECT_IS_BUILD_ON_CUDA \
    -DTORCH_EXTENSION_NAME=deep_select_cuda -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=$ABI \
    -fPIC $INC -I$CUDA/include $TORCHINC -I$PYINC -c csrc/api.cpp -o build90/api.o
g++ -shared build90/*.o -o deep_select/deep_select_cuda$EXT \
    -L$TORCHLIB -Wl,-rpath,$TORCHLIB -lc10 -ltorch -ltorch_cpu -ltorch_python -lc10_cuda -ltorch_cuda \
    -L$CUDA/lib64 -lcudart -L$CUDA/targets/x86_64-linux/lib/stubs -lcuda
ls -la deep_select/
python3 -c "import deep_select; print('import ok', deep_select.get_stride_requirement())" 2>&1 | grep -v -i numpy
echo BUILD DONE
