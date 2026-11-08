"""
analyze_listening.py — 블라인드 청취 결과 집계

    python analyze_listening.py --key listening_test/key.json --csv listening_kim.csv listening_lee.csv …

방법별 평균 점수(95% 부트스트랩 신뢰구간)와, 첫 번째 방법이 아닌 각 방법 vs 기준 방법의 짝 비교(Wilcoxon, 같은 듣는 사람·같은 곡끼리).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--ref", default="v5_best", help="짝 비교의 기준 방법")
    args = ap.parse_args()
    key = json.load(open(args.key, encoding="utf-8"))
    clips, systems = key["clips"], key["systems"]
    score = defaultdict(dict)          # (question) -> {(listener, source, system): score}
    for f in args.csv:
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            c = clips[r["clip"]]
            score[r["question"]][(r["listener"], c["source"], c["system"])] = float(r["score"])
    listeners = sorted({k[0] for d in score.values() for k in d})
    print(f"듣는 사람 {len(listeners)}명: {', '.join(listeners)}\n")
    rng = np.random.default_rng(0)
    for q, d in score.items():
        print(f"[{q}]")
        for s in systems:
            v = np.array([x for (l, src, sy), x in d.items() if sy == s])
            if len(v) == 0:
                continue
            bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            print(f"  {s:18s} 평균 {v.mean():.2f}  95% [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]  (n={len(v)})")
        for s in systems:
            if s == args.ref:
                continue
            pairs = [(d[(l, src, args.ref)], d[(l, src, s)]) for (l, src, sy) in d
                     if sy == s and (l, src, args.ref) in d]
            if len(pairs) >= 6:
                a, b = np.array(pairs).T
                if np.any(a != b):
                    p = wilcoxon(a, b).pvalue
                    print(f"  {args.ref} vs {s}: 평균 차 {np.mean(a - b):+.2f}, {args.ref} 우세 {int(np.sum(a > b))} / 열세 {int(np.sum(a < b))} / 같음 {int(np.sum(a == b))}, p={p:.4f}")
        print()


if __name__ == "__main__":
    main()
