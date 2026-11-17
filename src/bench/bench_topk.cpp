// Benchmark driver: generates data call by call for the three groups of shapes of the target workload,
// times every implementation and writes every run to a log.
// What the three groups are and how data is generated is in workload.h; how to add an implementation is
// in impl.h and in make_impls below.
//
// Usage (from src/): build/bench_topk [options]
//   --group 1,2,3      groups to run (default 1, 2, 3; group 4 must be named, see workload.h)
//   --impl serial,omp  implementations to measure (default all)
//   --reps N           repetitions per call (default 5); summaries take the median
//   --threads N        threads of the OpenMP implementation (default all cores)
//   --scale S          shrink the scale S times, S a power of 2 from 1 to 64 (default 1, full scale).
//                      Only for checking the driver
//   --seed X           seed of the data (default 20261111)
//   --tag NAME         appended to the log file name, e.g. the machine name
//   --out FILE         log path (default logs/bench-DATE-TIME[-tag].csv)
//   --plan             only list the calls that would be made, without running. Other benchmark scripts
//                      (the one for torch.topk, for example) follow this list
// The executable with the GPU (build/bench_topk_gpu from make gpu) has two more implementations and six
// more options:
//   dettopk            the GPU implementation
//   dettopk_select     the GPU implementation with selection only and no collect, for timing the two
//                      phases separately: the time of dettopk minus its time is the time of collect.
//                      Measured only when named in --impl (not part of the default "all"). Its output
//                      is incomplete and checked separately, see below
//   read1, read2       read-only baselines, not a top-k: launched like dettopk, they only read every
//                      row once or twice the way selection does (see GpuConfig::read_passes in dettopk_internal.h), to
//                      see how fast this access pattern can read. Measured only when named; the output
//                      is only slot 0 of each row (the sum of the scores' bit patterns), compared with
//                      what the host computes
//   --gpu-threads N    threads per thread block, a multiple of 32 (default automatic, see gpu/topk_gpu.cu)
//   --gpu-cand N       number of keys the candidate buffer holds, 0 for none (default automatic, see
//                      gpu/topk_gpu.cu)
//   --gpu-bits N       words for the vote bits of collect, 0 to record none and read the row again at
//                      write-out (default automatic, see gpu/topk_gpu.cu)
//   --gpu-pairs N      number of candidates the merged path holds, 0 to disable it (default automatic,
//                      see gpu/topk_gpu.cu)
//   --gpu-sample N     rows with valid length at least N try the sampled path first, 0 to disable it
//                      (default kSampleMinLenAuto in dettopk_internal.h)
//   --gpu-split N      with few rows, the maximum number of blocks a row is split across, 0 for no
//                      splitting (default automatic, see split_parts in gpu/topk_gpu.cu)
// build/bench_topk_cub from make bench-cub adds one more implementation (an external baseline that
// needs a separate download of CCCL, see bench/impl_cub.cu):
//   cub                NVIDIA CUB's batched top-k, required to be deterministic with ties going to the
//                      smaller position; it selects without ordering, and the positions are sorted on
//                      the host before comparing
// With the GPU, the head of the log also records how each thread count is launched: the candidate
// buffer size, the maximum words of vote bits, the maximum words of the work area, the resident blocks
// per SM and the register count (# gpu_launch). The merged path's candidate capacity varies with the
// row width and follows from the work area maximum (see plan_work in gpu/topk_gpu.cu).
//
// The log is CSV: a few leading lines starting with # record the machine, the thread counts, the
// compile options and the command line; after that every run takes one line, with no aggregation.
// out_hash is the hash of the run's output: all implementations and all runs of one call must have
// exactly the same one, which the driver checks, reporting an error otherwise. A table summarized by
// median is also printed to the screen.
//
// Before every run the output area is filled with an impossible value, so an implementation that
// fails to write its output gets a mismatching hash.
//
// The output of dettopk_select differs from the others' (only tau and the tie quota per row) and is not
// compared with them. What it should produce is derived from the complete output of the same call and
// then compared; when the run has no complete implementation, only its runs are compared with each other.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>
#include <sys/resource.h>
#include <sys/utsname.h>

#include "impl.h"
#include "key.h"
#include "dettopk_internal.h"
#include "workload.h"

#ifndef BENCH_GIT_REV
#define BENCH_GIT_REV "unknown"
#endif
#ifndef BENCH_CXXFLAGS
#define BENCH_CXXFLAGS "unknown"
#endif

namespace {

using bench::Call;

// ---------- Implementations under test ----------

struct Entry {
    std::string name;
    int threads;       // thread count written to the log
    bool select_only;  // selection only: incomplete output, checked separately
    int read_passes;   // above 0: read-only baseline with this many passes; incomplete output, compared with the host's
    std::unique_ptr<bench::Impl> impl;
};

// New implementations are added here. An implementation on the GPU is another subclass of bench::Impl
// (see impl.h).
#ifdef DETTOPK_WITH_GPU
std::vector<Entry> make_impls(int omp_threads, const dettopk::GpuConfig& gpu_config) {
#else
std::vector<Entry> make_impls(int omp_threads) {
#endif
    std::vector<Entry> v;
    v.push_back({"serial", 1, false, 0, std::make_unique<bench::HostImpl>(dettopk::topk_serial)});
    v.push_back({"omp", omp_threads, false, 0, std::make_unique<bench::HostImpl>(dettopk::topk_omp)});
#ifdef DETTOPK_WITH_GPU
    v.push_back({"dettopk", 0, false, 0, bench::make_gpu_impl(gpu_config)});
    dettopk::GpuConfig select_only = gpu_config;
    select_only.select_only = true;
    v.push_back({"dettopk_select", 0, true, 0, bench::make_gpu_impl(select_only)});
    for (const int passes : {1, 2}) {
        dettopk::GpuConfig read = gpu_config;
        read.read_passes = passes;
        v.push_back({"read" + std::to_string(passes), 0, false, passes, bench::make_gpu_impl(read)});
    }
#endif
#ifdef DETTOPK_WITH_CUB
    v.push_back({"cub", 0, false, 0, bench::make_cub_impl()});
#endif
    return v;
}

// ---------- Command line ----------

struct Options {
    std::vector<int> groups = {1, 2, 3};
    std::vector<std::string> impls;  // empty means all
    int reps = 5;
    int threads = 0;  // 0 means all cores
    int scale = 1;
    uint64_t seed = 20261111;
    std::string tag;
    std::string out;
    bool plan_only = false;
    int gpu_threads = 0;  // 0 means automatic
    int gpu_cand = -1;    // negative means automatic
    int gpu_bits = -1;    // negative means automatic
    int gpu_pairs = -1;   // negative means automatic
    int gpu_sample = -1;  // negative means automatic
    int gpu_split = -1;   // negative means automatic
};

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "bench_topk: %s (usage at the top of bench/bench_topk.cpp)\n", msg.c_str());
    std::exit(2);
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');) parts.push_back(item);
    return parts;
}

Options parse(int argc, char** argv) {
    Options o;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--plan") {
                o.plan_only = true;
                continue;
            }
            if (i + 1 >= argc) die("option " + a + " is unknown or lacks a value");
            const std::string v = argv[++i];
            if (a == "--group") {
                o.groups.clear();
                for (const std::string& s : split(v)) o.groups.push_back(std::stoi(s));
            } else if (a == "--impl") {
                o.impls = split(v);
            } else if (a == "--reps") {
                o.reps = std::stoi(v);
            } else if (a == "--threads") {
                o.threads = std::stoi(v);
            } else if (a == "--scale") {
                o.scale = std::stoi(v);
            } else if (a == "--seed") {
                o.seed = std::stoull(v);
            } else if (a == "--tag") {
                o.tag = v;
            } else if (a == "--out") {
                o.out = v;
            } else if (a == "--gpu-threads") {
                o.gpu_threads = std::stoi(v);
            } else if (a == "--gpu-cand") {
                o.gpu_cand = std::stoi(v);
            } else if (a == "--gpu-bits") {
                o.gpu_bits = std::stoi(v);
            } else if (a == "--gpu-pairs") {
                o.gpu_pairs = std::stoi(v);
            } else if (a == "--gpu-sample") {
                o.gpu_sample = std::stoi(v);
            } else if (a == "--gpu-split") {
                o.gpu_split = std::stoi(v);
            } else {
                die("unknown option " + a);
            }
        }
    } catch (const std::exception&) {
        die("an option value is not a number");
    }
    for (int g : o.groups) {
        if (g < 1 || g > 4) die("groups are 1 to 4");
    }
    if (o.groups.empty()) die("no group to run");
    if (o.reps < 1) die("--reps must be at least 1");
    if (o.threads < 0) die("--threads must not be negative");
    if (o.scale < 1 || o.scale > 64 || (o.scale & (o.scale - 1)) != 0) die("--scale must be a power of 2 from 1 to 64");
#ifndef DETTOPK_WITH_GPU
    if (o.gpu_threads != 0 || o.gpu_cand >= 0 || o.gpu_bits >= 0 || o.gpu_pairs >= 0 || o.gpu_sample >= 0 ||
        o.gpu_split >= 0) die("this executable was built without the GPU implementation (use bench_topk_gpu from make gpu)");
#endif
    return o;
}

// ---------- Machine information ----------

// The first line of the file that starts with key; returns what follows the colon.
std::string proc_field(const char* path, const std::string& key) {
    std::ifstream f(path);
    for (std::string line; std::getline(f, line);) {
        if (line.compare(0, key.size(), key) != 0) continue;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const size_t start = line.find_first_not_of(" \t", colon + 1);
        return start == std::string::npos ? "" : line.substr(start);
    }
    return "unknown";
}

std::string first_line(const char* path) {
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);
    return line.empty() ? "unknown" : line;
}

std::string env_or(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "(unset)";
}

std::string time_string(const char* format) {
    const std::time_t now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof buf, format, std::localtime(&now));
    return buf;
}

void write_header(std::FILE* log, const Options& o, const bench::Scenario& sc, int argc, char** argv,
                  int gen_threads) {
    utsname u{};
    uname(&u);
    std::fprintf(log, "# benchmark log of the deterministic batched top-k; format at the top of bench/bench_topk.cpp\n");
    std::fprintf(log, "# time: %s\n", time_string("%Y-%m-%d %H:%M:%S").c_str());
    std::fprintf(log, "# cpu: %s\n", proc_field("/proc/cpuinfo", "model name").c_str());
    std::fprintf(log, "# logical_cpus: %d\n", omp_get_num_procs());
    std::fprintf(log, "# mem_total: %s\n", proc_field("/proc/meminfo", "MemTotal").c_str());
    std::fprintf(log, "# os: %s %s %s\n", u.sysname, u.release, u.machine);
    std::fprintf(log, "# compiler: %s\n", __VERSION__);
    std::fprintf(log, "# cxxflags: %s\n", BENCH_CXXFLAGS);
    std::fprintf(log, "# git_rev: %s\n", BENCH_GIT_REV);
    std::fprintf(log, "# omp_threads: %d\n", omp_get_max_threads());
    std::fprintf(log, "# gen_threads: %d\n", gen_threads);
    std::fprintf(log, "# OMP_PROC_BIND: %s\n", env_or("OMP_PROC_BIND").c_str());
    std::fprintf(log, "# OMP_PLACES: %s\n", env_or("OMP_PLACES").c_str());
    std::fprintf(log, "# args:");
    for (int i = 1; i < argc; ++i) std::fprintf(log, " %s", argv[i]);
    std::fprintf(log, "\n");
    std::fprintf(log, "# scale: %d (chunk=%lld k=%d max_elems=%lld blocks=%d)%s\n", o.scale,
                 static_cast<long long>(sc.chunk), sc.k, static_cast<long long>(sc.max_elems), sc.blocks,
                 o.scale == 1 ? "" : ", reduced scale, times are not measurements");
    std::fprintf(log, "# seed: %llu\n", static_cast<unsigned long long>(o.seed));
    std::fprintf(log, "# reps: %d\n", o.reps);
    std::fprintf(log, "# loadavg_start: %s\n", first_line("/proc/loadavg").c_str());
}

// ---------- Running ----------

// Hash of the output (FNV-1a over 32-bit units).
struct Fnv {
    uint64_t h = 0xCBF29CE484222325ull;
    void add(int32_t v) {
        h ^= static_cast<uint32_t>(v);
        h *= 0x100000001B3ull;
    }
};

uint64_t hash_output(const int32_t* out, int64_t n) {
    Fnv f;
    for (int64_t i = 0; i < n; ++i) f.add(out[i]);
    return f.h;
}

// The hash of the output a selection-only run should produce, derived from the complete output `full`
// of the same call. Rows with valid length at most k have their normal output; for the other rows the
// first two slots are tau and the tie quota and the remaining slots are not written: tau is the smallest
// key among the selected positions, and the quota is the number of selected positions whose key equals tau.
uint64_t hash_select_only(const bench::Input& in, const int32_t* full) {
    Fnv f;
    for (int64_t r = 0; r < in.rows; ++r) {
        const int32_t* row = full + r * in.k;
        if (in.valid_len[r] <= in.k) {
            for (int32_t j = 0; j < in.k; ++j) f.add(row[j]);
            continue;
        }
        uint32_t tau = UINT32_MAX, quota = 0;
        for (int32_t j = 0; j < in.k; ++j) {
            const uint32_t key = dettopk::score_to_key(in.scores[r * in.width + row[j]]);
            if (key < tau) {
                tau = key;
                quota = 0;
            }
            if (key == tau) ++quota;
        }
        f.add(static_cast<int32_t>(tau));
        if (in.k > 1) f.add(static_cast<int32_t>(quota));
        for (int32_t j = 2; j < in.k; ++j) f.add(bench::kUnwritten);
    }
    return f.h;
}

// The hash of the output a read-only baseline (read1, read2) should produce: rows with valid
// length at most k have their normal output; for the other rows slot 0 is the sum of the bit patterns of
// the scores in the valid range times the number of passes (wrapping at 32 bits), and the remaining
// slots are not written.
uint64_t hash_read(const bench::Input& in, int passes) {
    std::vector<uint32_t> sums(static_cast<size_t>(in.rows), 0);
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < in.rows; ++r) {
        uint32_t s = 0;
        const float* row = in.scores + r * in.width;
        for (int32_t i = 0; i < in.valid_len[r]; ++i) s += dettopk::float_bits(row[i]);
        sums[static_cast<size_t>(r)] = s * static_cast<uint32_t>(passes);
    }
    Fnv f;
    for (int64_t r = 0; r < in.rows; ++r) {
        if (in.valid_len[r] <= in.k) {
            for (int32_t j = 0; j < in.k; ++j) f.add(j < in.valid_len[r] ? j : -1);
            continue;
        }
        f.add(static_cast<int32_t>(sums[static_cast<size_t>(r)]));
        for (int32_t j = 1; j < in.k; ++j) f.add(bench::kUnwritten);
    }
    return f.h;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void print_plan(const std::vector<Call>& plan) {
    std::printf("group,dist,block,call,rows,width,k,len_first,len_step\n");
    for (const Call& c : plan) {
        std::printf("%d,%s,%d,%d,%lld,%lld,%d,%d,%d\n", c.group, bench::dist_name(c.dist), c.block, c.call,
                    static_cast<long long>(c.rows), static_cast<long long>(c.width), c.k, c.len_first,
                    c.len_step);
    }
}

// med[i][j] is the median (in seconds) of implementation j on call i. The time of a configuration is
// the sum of the medians of its calls. "Elements" are rows times width, without subtracting the part
// beyond the valid length.
void print_summary(const std::vector<Call>& plan, const std::vector<Entry>& impls,
                   const std::vector<std::vector<double>>& med, int scale) {
    std::printf("\ng  b  dist      rows x width   calls  impl      thr median sum      elements/s\n");
    for (size_t b = 0; b < plan.size();) {
        size_t e = b;
        double elems = 0;
        while (e < plan.size() && plan[e].config == plan[b].config) {
            elems += static_cast<double>(plan[e].rows * plan[e].width);
            ++e;
        }
        for (size_t j = 0; j < impls.size(); ++j) {
            double t = 0;
            for (size_t i = b; i < e; ++i) t += med[i][j];
            std::printf("%-2d %-2d %-7s %6lld x %-7lld %5zu  %-8s %4d %10.3f ms %12.3e\n", plan[b].group,
                        plan[b].block, bench::dist_name(plan[b].dist), static_cast<long long>(plan[b].rows),
                        static_cast<long long>(plan[b].width), e - b, impls[j].name.c_str(),
                        impls[j].threads, t * 1e3, elems / t);
        }
        b = e;
    }
    // Group 1 is totalled per distribution (each distribution is one complete cold prefill).
    for (const bench::Dist d : {bench::Dist::kGumbel, bench::Dist::kLogistic, bench::Dist::kNormal}) {
        for (size_t j = 0; j < impls.size(); ++j) {
            double t = 0;
            bool any = false;
            for (size_t i = 0; i < plan.size(); ++i) {
                if (plan[i].group != 1 || plan[i].dist != d) continue;
                t += med[i][j];
                any = true;
            }
            if (any) {
                std::printf("group 1 total (%s), %s: %.3f s per layer, %d layers take %.1f s\n", bench::dist_name(d),
                            impls[j].name.c_str(), t, bench::kLayers, t * bench::kLayers);
            }
        }
    }
    if (scale != 1) {
        std::printf("scale reduced %d times: the times above only show that the driver runs and are not "
                    "measurements.\n", scale);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    const bench::Scenario sc = bench::make_scenario(o.scale);
    const std::vector<Call> plan = bench::build_plan(sc, o.groups);
    if (o.plan_only) {
        print_plan(plan);
        return 0;
    }

    const int gen_threads = omp_get_num_procs();  // data generation always uses all cores, whatever the thread count under test
    if (o.threads > 0) omp_set_num_threads(o.threads);

#ifdef DETTOPK_WITH_GPU
    dettopk::GpuConfig gpu_config;
    if (o.gpu_threads > 0) gpu_config.threads = o.gpu_threads;
    if (o.gpu_cand >= 0) gpu_config.cand_capacity = o.gpu_cand;
    if (o.gpu_bits >= 0) gpu_config.bits_capacity = o.gpu_bits;
    if (o.gpu_pairs >= 0) gpu_config.pair_capacity = o.gpu_pairs;
    if (o.gpu_sample >= 0) gpu_config.sample_min_len = o.gpu_sample;
    if (o.gpu_split >= 0) gpu_config.split_max = o.gpu_split;
    std::vector<Entry> impls = make_impls(omp_get_max_threads(), gpu_config);
#else
    std::vector<Entry> impls = make_impls(omp_get_max_threads());
#endif
    if (o.impls.empty()) {
        // By default all implementations are measured; selection-only and the read-only baselines are not
        // implementations and must be named.
        impls.erase(std::remove_if(impls.begin(), impls.end(),
                                   [](const Entry& e) { return e.select_only || e.read_passes > 0; }),
                    impls.end());
    } else {
        std::vector<Entry> chosen;
        for (const std::string& name : o.impls) {
            auto it = std::find_if(impls.begin(), impls.end(), [&](const Entry& e) { return e.name == name; });
            if (it == impls.end() || !it->impl) die("no implementation named " + name + ", or it is listed twice");
            chosen.push_back(std::move(*it));
        }
        impls = std::move(chosen);
    }

    std::string path = o.out;
    if (path.empty()) {
        path = "logs/bench-" + time_string("%Y%m%d-%H%M%S") + (o.tag.empty() ? "" : "-" + o.tag) + ".csv";
    }
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::FILE* log = std::fopen(path.c_str(), "w");
    if (!log) die("cannot open log file " + path);
    write_header(log, o, sc, argc, argv, gen_threads);
#ifdef DETTOPK_WITH_GPU
    std::fprintf(log, "# gpu: %s\n", bench::gpu_info().c_str());
#ifdef DETTOPK_PROFILE
    std::fprintf(log, "# profile: profiling build, times are not measurements; after each call one line "
                      "# prof: group,dist,block,call,rows,width,impl,rows that went through selection,"
                      "then the cycles these rows spent in each step (sample,cutoff,pass 1 count,pick,"
                      "pass 2 count,pick,pass 3 count,pick,vote fix-up,vote,prefix sum,write-out; "
                      "the sampled path reuses the slots from pass 1 to the vote fix-up, see ProfStep in gpu/block_row.h), "
                      "and finally the number of these rows per sampled-path outcome (not tried,completed,"
                      "fell back: too many samples,cutoff too high,does not fit)\n");
#endif
    std::fprintf(log, "# gpu_config: threads=%s cand=%s bits=%s pairs=%s sample=%s split=%s\n",
                 gpu_config.threads > 0 ? std::to_string(gpu_config.threads).c_str() : "auto",
                 gpu_config.cand_capacity >= 0 ? std::to_string(gpu_config.cand_capacity).c_str() : "auto",
                 gpu_config.bits_capacity >= 0 ? std::to_string(gpu_config.bits_capacity).c_str() : "auto",
                 gpu_config.pair_capacity >= 0 ? std::to_string(gpu_config.pair_capacity).c_str() : "auto",
                 gpu_config.sample_min_len >= 0 ? std::to_string(gpu_config.sample_min_len).c_str()
                                                : ("auto(" + std::to_string(dettopk::kSampleMinLenAuto) + ")").c_str(),
                 gpu_config.split_max >= 0 ? std::to_string(gpu_config.split_max).c_str() : "auto");
    {
        // How each thread count is launched. Resident blocks times threads is the number of resident
        // threads per SM; check that it is the same before comparing thread counts.
        dettopk::set_gpu_config(gpu_config);
        std::string launch;
        for (const int t : {128, 256, 512, 1024}) {
            dettopk::GpuLaunchInfo li{};
            char buf[192];
            if (dettopk::gpu_launch_info(t, &li) == dettopk::Status::kOk) {
                std::snprintf(buf, sizeof buf,
                              "%sthreads=%d cand=%d bits=%d work=%d blocks_per_sm=%d (plain %d) regs=%d local=%d",
                              launch.empty() ? "" : "; ", t, li.cand_capacity, li.bits_capacity, li.work_max,
                              li.blocks_per_sm,
                              li.blocks_per_sm_plain, li.regs, li.local_bytes);
            } else {
                std::snprintf(buf, sizeof buf, "%sthreads=%d unavailable", launch.empty() ? "" : "; ", t);
            }
            launch += buf;
        }
        std::fprintf(log, "# gpu_launch: %s\n", launch.c_str());
        std::fprintf(stderr, "gpu_launch: %s\n", launch.c_str());
    }
#endif
    std::fprintf(log, "group,dist,block,call,rows,width,k,len_first,len_step,impl,threads,rep,seconds,out_hash\n");

    // The buffers are allocated once for the largest call and reused by every call.
    int64_t max_elems = 0, max_rows = 0, max_out = 0;
    for (const Call& c : plan) {
        max_elems = std::max(max_elems, c.rows * c.width);
        max_rows = std::max(max_rows, c.rows);
        max_out = std::max(max_out, c.rows * c.k);
    }
    const std::unique_ptr<float[]> scores(new float[static_cast<size_t>(max_elems)]);
    std::vector<int32_t> lens(static_cast<size_t>(max_rows));
    std::vector<int32_t> out(static_cast<size_t>(max_out));
    const int32_t kUnwritten = bench::kUnwritten;
    const bool any_select_only =
        std::any_of(impls.begin(), impls.end(), [](const Entry& e) { return e.select_only; });
    bool select_only_unchecked = false;  // whether on some call the selection-only run had no complete output to check against

    std::vector<std::vector<double>> med(plan.size(), std::vector<double>(impls.size(), 0.0));
    int errors = 0;

    for (size_t i = 0; i < plan.size(); ++i) {
        const Call& c = plan[i];
        if (i == 0 || plan[i - 1].config != c.config) {
            std::fprintf(stderr, "group %d, block %d, %s, %lld x %lld\n", c.group, c.block, bench::dist_name(c.dist),
                         static_cast<long long>(c.rows), static_cast<long long>(c.width));
        }
        bench::fill_call(c, o.seed, scores.get(), lens.data(), gen_threads);
        const bench::Input in{scores.get(), c.rows, c.width, lens.data(), c.k, out.data()};
        const int64_t out_n = c.rows * c.k;

        bool have_hash = false;
        uint64_t want_hash = 0;
        // The hashes of the selection-only runs are recorded first and checked once all implementations of
        // this call have run (a complete implementation may come after it).
        struct SelectRun {
            size_t impl;
            int rep;
            uint64_t hash;
        };
        std::vector<SelectRun> select_runs;
        bool have_select_hash = false;
        uint64_t want_select_hash = 0;
        uint64_t want_read_hash[3] = {0, 0, 0};  // what the read-only baselines should produce, computed on demand
        bool have_read_hash[3] = {false, false, false};
        for (size_t j = 0; j < impls.size(); ++j) {
            bench::Impl& impl = *impls[j].impl;
            impl.load(in);
            // The very first call of the program runs once unmeasured: one-time costs such as creating the
            // thread pool do not count.
            if (i == 0) impl.run();
#ifdef DETTOPK_PROFILE
            dettopk::gpu_profile_take();  // discard what the warm-up and the previous implementation left
#endif

            std::vector<double> secs;
            for (int rep = 0; rep < o.reps; ++rep) {
                std::fill(out.begin(), out.begin() + out_n, kUnwritten);
                const auto t0 = std::chrono::steady_clock::now();
                const dettopk::Status st = impl.run();
                const auto t1 = std::chrono::steady_clock::now();
                impl.fetch();
                const double s = std::chrono::duration<double>(t1 - t0).count();
                const uint64_t h = hash_output(out.data(), out_n);
                secs.push_back(s);

                if (st != dettopk::Status::kOk) {
                    ++errors;
                    std::fprintf(stderr, "  error: %s returned a status other than kOk (run %d)\n", impls[j].name.c_str(), rep);
                }
                if (impls[j].read_passes > 0) {
                    const int passes = impls[j].read_passes;
                    if (!have_read_hash[passes]) {
                        want_read_hash[passes] = hash_read(in, passes);
                        have_read_hash[passes] = true;
                    }
                    if (h != want_read_hash[passes]) {
                        ++errors;
                        std::fprintf(stderr, "  error: %s run %d: output differs from what the host computes\n",
                                     impls[j].name.c_str(), rep);
                    }
                } else if (impls[j].select_only) {
                    select_runs.push_back({j, rep, h});
                } else if (!have_hash) {
                    want_hash = h;
                    have_hash = true;
                    // out now holds a complete output; while it is there, derive what a selection-only
                    // run should produce.
                    if (any_select_only) {
                        want_select_hash = hash_select_only(in, out.data());
                        have_select_hash = true;
                    }
                } else if (h != want_hash) {
                    ++errors;
                    std::fprintf(stderr, "  error: %s run %d: output differs from the first run of this call\n",
                                 impls[j].name.c_str(), rep);
                }
                std::fprintf(log, "%d,%s,%d,%d,%lld,%lld,%d,%d,%d,%s,%d,%d,%.9f,%016llx\n", c.group,
                             bench::dist_name(c.dist), c.block, c.call, static_cast<long long>(c.rows),
                             static_cast<long long>(c.width), c.k, c.len_first, c.len_step,
                             impls[j].name.c_str(), impls[j].threads, rep, s,
                             static_cast<unsigned long long>(h));
            }
            med[i][j] = median(secs);
#ifdef DETTOPK_PROFILE
            if (impls[j].name.rfind("dettopk", 0) == 0 || impls[j].read_passes > 0) {
                const dettopk::GpuProfile pr = dettopk::gpu_profile_take();
                std::fprintf(log, "# prof: %d,%s,%d,%d,%lld,%lld,%s,%llu", c.group, bench::dist_name(c.dist), c.block,
                             c.call, static_cast<long long>(c.rows), static_cast<long long>(c.width),
                             impls[j].name.c_str(), static_cast<unsigned long long>(pr.rows));
                for (const uint64_t cyc : pr.step_cycles) std::fprintf(log, ",%llu", static_cast<unsigned long long>(cyc));
                for (const uint64_t n : pr.sample_rows) std::fprintf(log, ",%llu", static_cast<unsigned long long>(n));
                std::fprintf(log, "\n");
            }
#endif
        }
        if (!select_runs.empty() && !have_select_hash) {
            want_select_hash = select_runs[0].hash;
            select_only_unchecked = true;
        }
        for (const SelectRun& run : select_runs) {
            if (run.hash == want_select_hash) continue;
            ++errors;
            std::fprintf(stderr, "  error: %s run %d: tau or the tie quota %s\n", impls[run.impl].name.c_str(), run.rep,
                         have_select_hash ? "differs from what the complete output implies"
                                          : "differs from the first run of this call");
        }
        std::fflush(log);
    }

    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    std::fprintf(log, "# loadavg_end: %s\n", first_line("/proc/loadavg").c_str());
    std::fprintf(log, "# peak_rss_mib: %.0f\n", static_cast<double>(ru.ru_maxrss) / 1024.0);
    if (any_select_only) {
        std::fprintf(log, "# select_only_check: %s\n",
                     select_only_unchecked
                         ? "runs only compared with each other (no complete implementation to check against)"
                         : "compared with the tau and tie quota implied by the complete output");
    }
    std::fprintf(log, "# errors: %d\n", errors);
    std::fclose(log);

    print_summary(plan, impls, med, o.scale);
    if (select_only_unchecked) {
        std::printf("note: this run has no complete implementation; the selection-only runs were only compared "
                    "with each other.\n");
    }
    std::printf("log: %s\n", path.c_str());
    if (errors) {
        std::printf("%d errors (see the error lines above); this log must not be used\n", errors);
        return 1;
    }
    return 0;
}
