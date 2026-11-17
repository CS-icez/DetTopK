#!/usr/bin/env python3
"""Summarizes benchmark logs into tables: each call takes the median of its runs, and the time of a
configuration is the sum of the medians of its calls.

Usage (from src/):
  python3 bench/summarize.py logs/bench-*-h100-*.csv

Logs written by bench_topk and by bench_torch.py are both accepted. A configuration is a block in group 1
and a shape with a distribution in groups 2 and 3.
It also checks that every call in a log has a single output hash (torch logs have no hashes and are not
checked; dettopk_select runs selection only and read1/read2 only read the rows, so their outputs
differ from the others' and each is compared across its own runs).
When a log has both dettopk and dettopk_select, a split table is printed too: selection is the time of
dettopk_select and collect is the difference.
For logs of the profiling build (make gpu-prof), the share of each step in the cycles of a row is printed.
load() and summarize() are also used by other scripts (plotting, comparisons across logs).
"""
import csv
import statistics
import sys
from collections import defaultdict

LAYERS = 61


# Logs written before the implementations were renamed use these names; they are read as the new ones.
OLD_NAMES = {"gpu": "dettopk", "gpu_select": "dettopk_select", "gpu_read1": "read1", "gpu_read2": "read2"}

def load(path):
    """Returns (the key-value pairs of the log head, the rows of the runs).

    The # prof: lines of a profiling build are kept separately in head["prof"] (a list).
    """
    head, lines = {"prof": []}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# prof:"):
                fields = line.split(":", 1)[1].strip().split(",")
                fields[6] = OLD_NAMES.get(fields[6], fields[6])
                head["prof"].append(fields)
            elif line.startswith("#"):
                key, _, value = line[1:].partition(":")
                head[key.strip()] = value.strip()
            else:
                lines.append(line)
    rows = list(csv.DictReader(lines))
    for r in rows:
        r["impl"] = OLD_NAMES.get(r["impl"], r["impl"])
    return head, rows


def summarize(rows):
    """Returns {(group, block, dist, width, rows of the whole block, impl, threads):
                (sum of medians [s], number of calls, number of elements)}.

    The configurations of group 3 have the same width and differ in the number of rows; the last call of
    a block in group 1 may have fewer rows, so a configuration is identified by the rows of the whole
    block (the sum over its calls).
    """
    per_call = defaultdict(list)
    for r in rows:
        call = (r["group"], r["block"], r["dist"], r["width"], r["call"], r["rows"], r["impl"], r["threads"])
        per_call[call].append(float(r["seconds"]))
    sums = defaultdict(lambda: [0.0, 0, 0])
    block_rows = defaultdict(int)
    for (group, block, dist, width, _call, n_rows, impl, threads), secs in per_call.items():
        # The configurations of groups 3 and 4 all belong to "block 0" and are told apart by the number
        # of rows; the others are merged per block
        part = n_rows if group in ("3", "4") else ""
        key = (group, block, dist, width, part, impl, threads)
        sums[key][0] += statistics.median(secs)
        sums[key][1] += 1
        sums[key][2] += int(n_rows) * int(width)
        block_rows[key] += int(n_rows)
    return {(g, b, d, int(w), block_rows[(g, b, d, w, p, i, t)], i, int(t)): tuple(v)
            for (g, b, d, w, p, i, t), v in sums.items()}


def check_hashes(rows):
    """Number of calls that show more than one output hash."""
    seen = defaultdict(set)
    for r in rows:
        if r["out_hash"] != "-":
            call = (r["group"], r["block"], r["dist"], r["width"], r["call"], r["rows"])
            # The implementations with incomplete output each form their own class: selection only, read
            # once, read twice
            kind = r["impl"] if r["impl"] in ("dettopk_select", "read1", "read2") else "full"
            seen[call + (kind,)].add(r["out_hash"])
    return sum(1 for hashes in seen.values() if len(hashes) > 1)


def main():
    for path in sys.argv[1:]:
        head, rows = load(path)
        table = summarize(rows)
        print(f"== {path}")
        for key in ("cpu", "gpu", "gpu_config", "git_rev", "args", "scale"):
            if key in head:
                print(f"   {key}: {head[key]}")
        print(f"   calls with inconsistent output hashes: {check_hashes(rows)}")
        print("   g  b  dist      rows x width   calls  impl             thr         ms   elements/s")
        order = sorted(table, key=lambda k: (k[0], int(k[1]), k[3], k[4], k[2], k[5]))
        for key in order:
            group, block, dist, width, n_rows, impl, threads = key
            secs, calls, elems = table[key]
            print(f"   {group:<2} {block:<2} {dist:<7} {n_rows:>6} x {width:<7} {calls:>5}  {impl:<15} {threads:>4} "
                  f"{secs * 1e3:>10.3f} {elems / secs:>12.3e}")
        totals = defaultdict(float)  # group 1 is totalled per distribution: each distribution is one complete cold prefill
        for (group, _b, dist, _w, _r, impl, threads), (secs, _c, _e) in table.items():
            if group == "1":
                totals[(dist, impl, threads)] += secs
        for (dist, impl, threads), secs in sorted(totals.items()):
            print(f"   group 1 total ({dist}), {impl} ({threads} threads): {secs:.3f} s per layer, {LAYERS} layers take {secs * LAYERS:.1f} s")
        print_split(table)
        print_profile(head["prof"])
        print()


STEPS = ["sample", "cutoff", "count1", "pick1", "count2", "pick2", "count3", "pick3", "fixup", "vote", "prefix", "write"]
# Outcomes of the sampled path (SampleOutcome in gpu/block_row.h): the last 5 numbers of a profile line.
# Logs that predate the sampled path do not have them.
OUTCOMES = ["none", "done", "many", "high", "full"]
# Logs that predate the sampled path lack the sample and cutoff steps as well as the last 5 numbers.
STEPS_10 = ["count1", "pick1", "count2", "pick2", "count3", "pick3", "fixup", "vote", "prefix", "write"]
# Logs that predate the merged path lack the fix-up step.
STEPS_9 = ["count1", "pick1", "count2", "pick2", "count3", "pick3", "vote", "prefix", "write"]


def print_profile(prof):
    """Profiling build: totals the cycles of each step per configuration and prints the average cycles
    per row and the share of each step.

    The oldest profiling logs have only two numbers per row, pick and whole row (10 fields); for them
    only the share of the pick is printed. Step-by-step logs without the fix-up step (9 steps) are printed
    with STEPS_9 and those that predate the sampled path (10 steps) with STEPS_10.
    When sampled-path outcomes are present, the number of rows per outcome is printed as well.
    """
    if not prof:
        return
    if len(prof[0]) == 10:
        print_profile_old(prof)
        return
    n_fields = len(prof[0])
    has_outcome = n_fields == 8 + len(STEPS) + len(OUTCOMES)
    steps_names = {8 + len(STEPS_9): STEPS_9, 8 + len(STEPS_10): STEPS_10}.get(n_fields, STEPS)
    n_steps = len(steps_names)
    sums = defaultdict(lambda: [0] * (n_steps + 1))
    outcomes = defaultdict(lambda: [0] * len(OUTCOMES))
    for fields in prof:
        group, dist, block, _call, n_rows, width, impl, n = fields[:8]
        key = (group, block, dist, width, n_rows if group in ("3", "4") else "", impl)
        acc = sums[key]
        acc[0] += int(n)
        for j, cyc in enumerate(fields[8:8 + n_steps]):
            acc[j + 1] += int(cyc)
        if has_outcome:
            for j, cnt in enumerate(fields[8 + n_steps:]):
                outcomes[key][j] += int(cnt)
    print("   profile (share of each step in the cycles of a row):\n         g  b  dist      rows x width   impl          cyc/row "
          + " ".join(f"{name:>6}" for name in steps_names))
    for key in sorted(sums, key=lambda k: (k[0], int(k[1]), k[3], k[4], k[2], k[5])):
        group, block, dist, width, n_rows, impl = key
        n, steps = sums[key][0], sums[key][1:]
        row = sum(steps)
        if n == 0 or row == 0:
            continue
        print(f"         {group:<2} {block:<2} {dist:<7} {n_rows:>6} x {width:<7} {impl:<12} {row / n:>8.0f} "
              + " ".join(f"{c / row:>6.1%}" for c in steps))
    if not has_outcome:
        return
    print("   sampled path (rows per outcome):\n         g  b  dist      rows x width   impl         "
          + " ".join(f"{name:>8}" for name in OUTCOMES))
    for key in sorted(outcomes, key=lambda k: (k[0], int(k[1]), k[3], k[4], k[2], k[5])):
        group, block, dist, width, n_rows, impl = key
        print(f"         {group:<2} {block:<2} {dist:<7} {n_rows:>6} x {width:<7} {impl:<12} "
              + " ".join(f"{c:>8}" for c in outcomes[key]))


def print_profile_old(prof):
    sums = defaultdict(lambda: [0, 0, 0])
    for group, dist, block, _call, n_rows, width, impl, pick, row, n in prof:
        key = (group, block, dist, width, n_rows if group in ("3", "4") else "", impl)
        sums[key][0] += int(pick)
        sums[key][1] += int(row)
        sums[key][2] += int(n)
    print("   profile:\n         g  b  dist      rows x width   impl         pick cyc/row    row cyc/row     share")
    for key in sorted(sums, key=lambda k: (k[0], int(k[1]), k[3], k[4], k[2], k[5])):
        group, block, dist, width, n_rows, impl = key
        pick, row, n = sums[key]
        if n == 0 or row == 0:
            continue
        print(f"         {group:<2} {block:<2} {dist:<7} {n_rows:>6} x {width:<7} {impl:<12} {pick / n:>12.0f} "
              f"{row / n:>14.0f} {pick / row:>9.1%}")


def print_split(table):
    """How much selection and collect each take: dettopk minus dettopk_select on the same configuration."""
    full = {k[:5]: v[0] for k, v in table.items() if k[5] == "dettopk"}
    select = {k[:5]: v[0] for k, v in table.items() if k[5] == "dettopk_select"}
    both = sorted(set(full) & set(select), key=lambda k: (k[0], int(k[1]), k[3], k[4], k[2]))
    if not both:
        return
    print("   split (ms):   g  b  dist      rows x width      total   select  collect    share")
    for key in both:
        group, block, dist, width, n_rows = key
        t, s = full[key] * 1e3, select[key] * 1e3
        print(f"                 {group:<2} {block:<2} {dist:<7} {n_rows:>6} x {width:<7} {t:>8.3f} {s:>8.3f} "
              f"{t - s:>8.3f} {(t - s) / t:>8.1%}")
    for dist in sorted({k[2] for k in both if k[0] == "1"}):
        t = sum(full[k] for k in both if k[0] == "1" and k[2] == dist)
        s = sum(select[k] for k in both if k[0] == "1" and k[2] == dist)
        print(f"   group 1 total ({dist}): {t:.3f} s per layer, of which selection {s:.3f} s and collect {t - s:.3f} s ({(t - s) / t:.1%})")


if __name__ == "__main__":
    main()
