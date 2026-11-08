"""
eval_real.py — 평가 전용 실제 옛 녹음(real_tests/) 전체를 여러 모델로 변환하고, 정답 없이 잡음 지표를 비교

사용 예:
    python eval_real.py --ckpt runs/v3/best.pt runs/v5/best.pt
    python eval_real.py --ckpt runs/v5/best.pt --systems moliner_denoise moliner_dsp   (기존 복원 모델과 비교)

- real_tests/ 아래 오디오 파일을 모두 찾음 (결과 폴더 real_tests/out 은 제외)
- 모델마다 real_tests/out/<파일이름>__<run이름>.wav 로 저장 (이미 있으면 건너뜀, --force 로 다시)
- 파일별 표 + 모든 파일 평균 표 출력, real_tests/out/summary.csv 저장 (논문 표용)
"""

from __future__ import annotations

import argparse
import csv
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from infer import SR, convert, load_audio
from measure_real import GUIDE, load, measure
from model import SpecUNet

EXT = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".webm", ".opus"}


def run_name(ckpt: Path) -> str:
    return f"{ckpt.parent.name}_{ckpt.stem}"          # runs/v5/best.pt → v5_best


def main():
    ap = argparse.ArgumentParser(description="real_tests 전체 변환 + 잡음 지표 비교")
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--systems", nargs="*", default=[],
                    help="이미 만들어진 다른 방법의 결과도 함께 측정: <dir>/out/<파일이름>__<이름>.wav (예: moliner_denoise moliner_dsp)")
    ap.add_argument("--dir", default="real_tests")
    ap.add_argument("--out", default=None, help="기본: <dir>/out")
    ap.add_argument("--force", action="store_true", help="이미 변환한 파일도 다시 변환")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(args.dir)
    out = Path(args.out) if args.out else root / "out"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in EXT and out.resolve() not in p.resolve().parents)
    if not files:
        raise SystemExit(f"[오류] {root} 에 오디오 파일이 없습니다.")
    print(f"평가 파일 {len(files)}개, 모델 {len(args.ckpt)}개, 다른 방법 {len(args.systems)}개, 장치 {args.device}")
    missing = [f"{p.stem}__{s}.wav" for s in args.systems for p in files if not (out / f"{p.stem}__{s}.wav").exists()]
    if missing:
        raise SystemExit(f"[오류] 다른 방법의 결과 파일이 없습니다: {missing[:5]} (먼저 만드세요, 예: restore_baseline.py)")

    device = torch.device(args.device)
    outputs = {p: [] for p in files}
    for c in args.ckpt:
        c = Path(c)
        ck = torch.load(c, map_location=device)
        model = SpecUNet(base=ck["args"].get("base", 32), mono=ck["args"].get("mono", False)).to(device).eval()
        model.load_state_dict(ck["model"])
        name = run_name(c)
        print(f"\n[{name}] step {ck.get('step', '?')}")
        for p in files:
            o = out / f"{p.stem}__{name}.wav"
            outputs[p].append((name, o))
            if o.exists() and not args.force:
                print(f"  있음, 건너뜀: {o.name}")
                continue
            x, sr = load_audio(str(p))
            if sr != SR:
                g = gcd(sr, SR)
                x = resample_poly(x, SR // g, sr // g).astype(np.float32)
            peak_in = float(np.abs(x).max() + 1e-9)
            y = convert(model, x, device)
            if np.abs(y).max() > 0.99:                # infer.py 와 같은 처리
                y = y / (np.abs(y).max() + 1e-9) * min(peak_in, 0.95)
            sf.write(o, y, SR, subtype="PCM_24")
            print(f"  변환: {o.name} ({len(y) / SR:.0f}초)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    keys = list(GUIDE)
    for s in args.systems:
        for p in files:
            outputs[p].append((s, out / f"{p.stem}__{s}.wav"))
    cols = ["input"] + [run_name(Path(c)) for c in args.ckpt] + list(args.systems)
    rows, per = [], {k: {c: [] for c in cols} for k in keys}
    for p in files:
        res = [measure(load(str(p)))[0]] + [measure(load(str(o)))[0] for _, o in outputs[p]]
        print(f"\n== {p.relative_to(root).as_posix()}")
        print(f"{'':34}" + "".join(f"{c:>12}" for c in cols))
        for k in keys:
            print(f"{k:34}" + "".join(f"{r[k]:>12.1f}" for r in res))
            for c, r in zip(cols, res):
                per[k][c].append(r[k])
                rows.append({"file": p.relative_to(root).as_posix(), "system": c, "metric": k, "value": f"{r[k]:.3f}"})

    print(f"\n== 전체 {len(files)}개 평균 (중앙값)")
    print(f"{'':34}" + "".join(f"{c:>18}" for c in cols) + "   보는 법")
    for k in keys:
        cells = "".join(f"{np.mean(per[k][c]):>10.1f} ({np.median(per[k][c]):>5.1f})" for c in cols)
        print(f"{k:34}{cells}   {GUIDE[k]}")

    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "system", "metric", "value"])
        w.writeheader(); w.writerows(rows)
    print(f"\n저장: {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
