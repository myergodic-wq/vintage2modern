"""
diagnose_suppression.py — 잡음 제거가 음악까지 깎는지 진단 (2026-11-04, PROGRESS r31)

청취 3차에서 Moliner+DSP 가 v7+r2 보다 유의하게 좋았음 (같은 DSP → 차이는 잡음 제거 핵심).
가설: v7 이 과하게 지워 고음·잔향 꼬리까지 깎음. 입력 대비로 세 가지를 잼 (정답이 없으므로 방법끼리 비교):
  음악 이득   : 음악이 큰 프레임(입력 에너지 상위 30%)에서 대역별 출력/입력 dB. 낮을수록 그 대역을 많이 깎음
  잡음 바닥   : 조용한 프레임(하위 10%)에서 대역별 출력/입력 dB. 낮을수록 잡음을 많이 지움
  꼬리 감쇠   : 큰 프레임 뒤 70~230 ms 의 에너지 비(250 Hz~4 kHz), 출력 − 입력 dB. 음수가 클수록 꼬리를 빨리 끊음
한계: 큰 프레임에도 잡음이 섞여 있어 '음악 이득'이 낮은 것이 음악 손실인지 잡음 제거인지 단독으로는 구분 못 함 → 잡음 바닥·청취와 함께 해석

사용 예:
    python diagnose_suppression.py --dir real_tests --systems v7_best moliner_denoise
"""

from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from audio_io import SR, load_audio

BANDS = [(63, 250), (250, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 12000), (12000, 16000)]
EXT = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".webm", ".opus"}
NFFT, HOP = 4096, 1024


def spec(x):
    f, _, Z = signal.stft(x, SR, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None, padded=False)
    return f, np.abs(Z) ** 2


def measure(x_in, x_out):
    """같은 길이의 모노 입력·출력 → (음악 이득[band], 잡음 바닥[band], 꼬리 감쇠 dB)"""
    n = min(len(x_in), len(x_out))
    f, Pi = spec(x_in[:n]); _, Po = spec(x_out[:n])
    e = Pi.sum(0)
    loud, quiet = e >= np.percentile(e, 70), e <= np.percentile(e, 10)
    bm = [(f >= lo) & (f < hi) for lo, hi in BANDS]
    db = lambda a, b: 10 * np.log10((a + 1e-20) / (b + 1e-20))
    gain = [db(Po[m][:, loud].sum(), Pi[m][:, loud].sum()) for m in bm]
    floor = [db(Po[m][:, quiet].sum(), Pi[m][:, quiet].sum()) for m in bm]
    mid = (f >= 250) & (f < 4000)
    ei, eo = Pi[mid].sum(0), Po[mid].sum(0)
    ks = range(3, 11)                                      # 70~230 ms 뒤
    tails = []
    for t in np.where(loud)[0]:
        if t + 10 >= len(ei):
            continue
        di = db(np.mean([ei[t + k] for k in ks]), ei[t])
        if di < -6:                                        # 입력에서 실제로 소리가 줄어드는 곳만 (음이 끝나는 곳)
            tails.append(db(np.mean([eo[t + k] for k in ks]), eo[t]) - di)
    return np.array(gain), np.array(floor), float(np.median(tails)) if tails else float("nan"), len(tails)


def read_mono(p):
    try:
        x, sr = sf.read(p, dtype="float64", always_2d=True); x = x.mean(1)
    except Exception:
        x, sr = load_audio(str(p)); x = x.astype(np.float64)
    if sr != SR:
        g = gcd(sr, SR); x = signal.resample_poly(x, SR // g, sr // g)
    return x


def main():
    ap = argparse.ArgumentParser(description="잡음 제거가 음악까지 깎는지 진단")
    ap.add_argument("--dir", default="real_tests")
    ap.add_argument("--systems", nargs="+", required=True)
    args = ap.parse_args()
    root, od = Path(args.dir), Path(args.dir) / "out"
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXT and od.resolve() not in p.resolve().parents)
    res = {s: [] for s in args.systems}
    for p in files:
        x = read_mono(p)
        for s in args.systems:
            res[s].append(measure(x, read_mono(od / f"{p.stem}__{s}.wav")))
    names = [f"{lo / 1000:g}-{hi / 1000:g}k" for lo, hi in BANDS]
    for title, i in (("음악 이득 (큰 프레임, 출력−입력 dB, 곡별 중앙값)", 0), ("잡음 바닥 (조용한 프레임, 출력−입력 dB, 곡별 중앙값)", 1)):
        print(f"\n{title}\n{'':18}" + "".join(f"{n:>10}" for n in names))
        for s in args.systems:
            print(f"{s:18}" + "".join(f"{v:>10.1f}" for v in np.median([r[i] for r in res[s]], 0)))
    print(f"\n꼬리 감쇠 (음 끝난 뒤 70~230 ms, 출력−입력 dB, 곡별 값의 중앙값. 음수 클수록 꼬리를 빨리 끊음)")
    for s in args.systems:
        t = [r[2] for r in res[s]]
        print(f"{s:18}{np.nanmedian(t):>10.1f}   (곡별: " + " ".join(f"{v:.1f}" for v in t) + f", 잰 구간 곡별 최소 {min(r[3] for r in res[s])}개)")
    if len(args.systems) == 2:
        a, b = args.systems
        print(f"\n{a} − {b} 곡별 차이 (음악 이득, 중앙값과 {a} 가 더 깎은 곡 수 / {len(files)})")
        d = np.array([ra[0] - rb[0] for ra, rb in zip(res[a], res[b])])
        print(f"{'':18}" + "".join(f"{n:>10}" for n in names))
        print(f"{'중앙 차':18}" + "".join(f"{v:>10.1f}" for v in np.median(d, 0)))
        print(f"{'더 깎은 곡':18}" + "".join(f"{int(v):>10d}" for v in (d < 0).sum(0)))


if __name__ == "__main__":
    main()
