"""Summarize CSV from bench_distances.mojo using per-cell medians."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(path: Path) -> None:
    cells = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(line for line in stream if not line.startswith("#")):
            key = tuple(int(row[name]) for name in ("mode", "dimension", "width", "unroll", "guarded"))
            cells[key].append(float(row["ns_per_distance"]))
    medians = {key: statistics.median(values) for key, values in cells.items()}
    print(f"\n{path}: {len(cells)} cells, {min(map(len, cells.values()))} samples/cell")
    for mode, name in enumerate(("L2", "IP", "L2 batch4", "IP batch4")):
        for small in (True, False):
            dimensions = sorted({key[1] for key in cells if key[0] == mode and (key[1] <= 32) == small})
            baseline_unroll = 4 if mode < 2 else 1
            policies = sorted({(key[2], key[3]) for key in cells if key[0] == mode and key[4] == 1})
            scores = []
            for width, unroll in policies:
                ratios = [medians[mode, d, 8, baseline_unroll, 1] / medians[mode, d, width, unroll, 1] for d in dimensions]
                scores.append((statistics.geometric_mean(ratios), width, unroll))
            print(name, "PQ<=32" if small else "embeddings>=128", "top policies (speedup,width,unroll):", sorted(scores, reverse=True)[:5])
    for mode in (0, 1):
        changes = []
        for key, value in medians.items():
            if key[0] == mode and key[4] == 0:
                guarded = medians[mode, key[1], key[2], 4, 1]
                changes.append(guarded / value)
        print("unguarded", "L2" if mode == 0 else "IP", "geomean speedup", round(statistics.geometric_mean(changes), 3), "worst", round(min(changes), 3))
    print("embedding dimension winners (mode,d,width,unroll,ns):")
    for mode in range(4):
        for d in (128, 384, 768, 1536):
            options = [(value, key[2], key[3]) for key, value in medians.items() if key[0] == mode and key[1] == d and key[4] == 1]
            value, width, unroll = min(options)
            print(mode, d, width, unroll, round(value, 3))
    print("width8 vs old rerank width64 (unroll4), speedups by mode and dimension:")
    for mode in (0, 1):
        print(mode, [(d, round(medians[mode, d, 64, 4, 1] / medians[mode, d, 8, 4, 1], 3)) for d in (128, 384, 768, 1536)])
    if any(key[0] == 6 for key in cells):
        print("tuned wrappers vs width8 baseline, embeddings>=128:")
        for mode in range(4):
            unroll = 4 if mode < 2 else 1
            ratios = [medians[mode, d, 8, unroll, 1] / medians[mode + 6, d, 0, 0, 1] for d in (128, 384, 768, 1536)]
            print(mode, "geomean", round(statistics.geometric_mean(ratios), 3), "per dimension", [round(r, 3) for r in ratios])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.csv:
        summarize(path)
