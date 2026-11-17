#!/usr/bin/env python3
"""Compares the output hashes of the same call across logs: among the implementations of a new round,
and against earlier rounds.

Usage (from src/):
  python3 bench/check_hashes.py 'logs/bench-*-h100-*.csv' -- 'logs/bench-*.csv'
The new logs come before "--" and all the logs to compare against come after it (both may be globs; the
new logs are included automatically).

A call is identified by (distribution, block, call within the block, rows, width, k, valid length of the
first row, step): with the same seed these determine the data (see workload.h), whichever group the call
belongs to. The implementations with incomplete output (dettopk_select, read1, read2) each form
their own class; the others (serial, omp, dettopk, cub) must be bit-identical and form one class.
Only logs at full scale, with the default seed and without driver errors are compared; logs without
hashes, such as those of torch, are skipped.
Ablation prototypes (with "ablate" on the command line) end early on purpose and have incomplete output;
differing from the new logs is expected of them, and only their number is reported. For any other log, a
single differing call gets the log listed, the last line is HASH MISMATCH and the exit status is nonzero.
"""
import csv
import glob
import sys
from collections import defaultdict

PARTIAL = ("dettopk_select", "read1", "read2")
# Logs written before the implementations were renamed use these names; they are read as the new ones.
OLD_NAMES = {"gpu": "dettopk", "gpu_select": "dettopk_select", "gpu_read1": "read1", "gpu_read2": "read2"}
SEED = "20261111"


def load(path):
    head, lines = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                head[key.strip()] = value.strip()
            else:
                lines.append(line)
    rows = list(csv.DictReader(lines))
    for r in rows:
        r["impl"] = OLD_NAMES.get(r["impl"], r["impl"])
    return head, rows


def call_of(r):
    kind = r["impl"] if r["impl"] in PARTIAL else "full"
    return (r["dist"], r["block"], r["call"], r["rows"], r["width"], r["k"], r["len_first"], r["len_step"], kind)


def main():
    sep = sys.argv.index("--")
    new = {p for g in sys.argv[1:sep] for p in glob.glob(g)}
    paths = sorted({p for g in sys.argv[sep + 1:] for p in glob.glob(g)} | new)
    seen = defaultdict(lambda: defaultdict(set))  # call -> hash -> {(log, impl)}
    heads, skipped, runs = {}, defaultdict(int), 0
    for p in paths:
        head, rows = load(p)
        why = None
        if not head.get("scale", "1 ").startswith("1 "):
            why = "reduced scale"
        elif head.get("seed", SEED) != SEED:
            why = "different seed"
        elif head.get("errors", "0") != "0":
            why = "driver reported errors"
        elif not any(r.get("out_hash", "-") != "-" for r in rows):
            why = "no hashes"
        if why:
            skipped[why] += 1
            if p in new and why != "no hashes":
                print(f"note: new log {p} was not compared ({why})")
            continue
        heads[p] = head
        for r in rows:
            if r["out_hash"] != "-":
                seen[call_of(r)][r["out_hash"]].add((p, r["impl"]))
                runs += 1
    print(f"compared {len(heads)} logs, {runs} runs; skipped: {dict(skipped) or 'none'}")

    # Every call in the new logs should have a single hash; with that as the reference, see for each old
    # log how many calls agree and how many differ.
    ref, new_bad = {}, []
    for c, by_hash in seen.items():
        mine = {h for h, who in by_hash.items() if any(p in new for p, _ in who)}
        if len(mine) > 1:
            new_bad.append(c)
        elif mine:
            ref[c] = next(iter(mine))
    print(f"{len(new & set(heads))} new logs: {len(ref) + len(new_bad)} distinct calls, of which {len(new_bad)} are inconsistent among the new logs themselves")
    for c in new_bad[:10]:
        print("  inconsistent:", c, {h: sorted(who)[:4] for h, who in seen[c].items()})
    impls = defaultdict(set)
    for c in ref:
        impls[c[-1]].update(i for p, i in seen[c][ref[c]] if p in new)
    for kind in sorted(impls):
        print(f"  {kind}: {sum(1 for c in ref if c[-1] == kind)} calls, implementations in the new logs {sorted(impls[kind])}")

    agree, differ = defaultdict(set), defaultdict(set)
    for c, want in ref.items():
        for h, who in seen[c].items():
            for p, _ in who:
                if p not in new:
                    (agree if h == want else differ)[p].add(c)
    old = set(agree) | set(differ)
    clean = [p for p in old if p not in differ]
    covered = set().union(*(agree[p] for p in clean)) if clean else set()
    ablate = [p for p in differ if "ablate" in heads[p].get("args", "")]
    real = sorted(set(differ) - set(ablate))
    print(f"{len(old)} old logs contain these calls: {len(clean)} agree everywhere, together covering {len(covered)} of the {len(ref)} calls of the new logs")
    print(f"{len(differ)} differ somewhere: {len(ablate)} ablation prototypes (expected to differ) and {len(real)} others")
    for p in real:
        print(f"  {p}: {len(agree[p])} calls agree, {len(differ[p])} differ; args: {heads[p].get('args', '')}; git_rev: {heads[p].get('git_rev', '')}")
    if new_bad or real:
        print("HASH MISMATCH")
        return 1
    print("HASH OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
