"""Per-category table for a `fast-rs specbench` results file.  usage: python3 scripts/summary.py results/qwen3-4b.jsonl"""
import json, sys
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1])]
by = defaultdict(list)
for r in rows: by[r["category"]].append(r)
print(f"{'category':16} {'questions':>9} {'plain t/s':>10} {'fast t/s':>9} {'speedup':>8} {'identical':>10}")
def line(name, rs):
    b = sum(r["baseline_speed"] for r in rs) / len(rs); m = sum(r["method_speed"] for r in rs) / len(rs)
    print(f"{name:16} {len(rs):9} {b:10.1f} {m:9.1f} {m/b:7.2f}x {sum(r['identical'] for r in rs):>5}/{len(rs)}")
for c in sorted(by): line(c, by[c])
line("overall", rows)
