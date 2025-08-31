#!/usr/bin/env python3
import json
import os
import sys

def load(dirpath):
    data = {}
    for root, _, files in os.walk(dirpath):
        for f in files:
            if f.endswith('.json'):
                p = os.path.join(root, f)
                try:
                    with open(p) as fh:
                        obj = json.load(fh)
                        data.update(obj)
                except Exception:
                    pass
    return data

def main():
    if len(sys.argv) < 4:
        print("usage: compare_iai.py <baseline_dir> <latest_dir> --slack <fraction>")
        sys.exit(0)
    baseline_dir, latest_dir = sys.argv[1], sys.argv[2]
    slack = 0.10
    if '--slack' in sys.argv:
        i = sys.argv.index('--slack')
        slack = float(sys.argv[i+1])

    if not os.path.isdir(baseline_dir) or not os.path.isdir(latest_dir):
        print("warn: baseline or latest not present; skipping perf gate")
        sys.exit(0)

    base = load(baseline_dir)
    new = load(latest_dir)
    failed = []
    for k, v in base.items():
        if k not in new:
            print(f"warn: missing latest metric for {k}")
            continue
        b = float(v)
        n = float(new[k])
        if n > b * (1.0 + slack):
            failed.append((k, b, n))
    if failed:
        print("Perf regressions (beyond slack):")
        for (k, b, n) in failed:
            print(f"  {k}: baseline={b}, latest={n}")
        sys.exit(1)
    else:
        print("Perf within baseline slack.")

if __name__ == '__main__':
    main()

