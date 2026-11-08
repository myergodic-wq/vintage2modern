"""
spatial.py — 물리 기반 공간 렌더링 단계 (학습 없음, 2026-11-04, PROGRESS 4-6 절)

모노 복원 결과(가운데)를 받아 스테레오로 만듭니다. 학습된 홀·스테레오(v5)가 청취에서 실패해서 분리한 단계입니다.
    1. 고음 쉘빙 EQ (회귀 모델이 잡음과 함께 깎은 고음 보정, 기본 6 kHz↑ +6 dB)
    2. 마른 소리는 가운데에 그대로 (v5 의 '가운데가 없다' 문제를 구조적으로 막음)
    3. 좌우가 서로 다른 확산 잔향 (서로 독립인 잡음으로 만든 IR → 진짜로 비상관된 좌우)
    4. (hall) 저음 모노: 좌우 차이를 mono_below Hz 아래에서 제거, 대역별 잔향 시간(저음 길게, 고음 짧게), 폭 조절

프리셋
    r2   : 청취 2차에서 'v5 가운데 + 단순 처리'(4.00점)로 쓴 dsp_baseline.simple_dsp 와 같은 결과 (기준)
    hall : r2 + 저음 모노(150 Hz) + 대역별 잔향 시간 (개선 후보 — 청취에서 r2 보다 나을 때만 채택)

사용 예:
    python spatial.py 입력.wav 출력.wav --preset hall
    python spatial.py --dir real_tests --systems v7_best v5_best --preset r2
        → real_tests/out/<파일>__<방법>.wav 의 가운데를 렌더링해 <파일>__<방법>_r2.wav 로 저장 (청취 3차 묶음용)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
PRESETS = {
    "r2":   dict(rt60=1.8, dry=0.85, wet=0.35, width=1.0, shelf_db=6.0, shelf_hz=6000, mono_below=0, bands=False),
    "hall": dict(rt60=1.8, dry=0.85, wet=0.35, width=1.0, shelf_db=6.0, shelf_hz=6000, mono_below=150, bands=True),
}
BAND_RT = (500, 4000, 1.15, 0.55)     # 500 Hz 아래 잔향 시간 ×1.15, 4 kHz 위 ×0.55 (공기 흡수로 고음이 빨리 사라짐)


def high_shelf(x, gain_db, f0, sr=SR):
    """dsp_baseline.simple_dsp 와 같은 쉘빙 필터 (같은 계수, 같은 lfilter)."""
    if gain_db == 0:
        return x.copy()
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    al, cw = np.sin(w0) / 2 * np.sqrt(2), np.cos(w0)
    b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * cw),
         A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * al)]
    a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * cw),
         (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * al]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x)


def make_irs(rt60, seed, bands, sr=SR):
    """좌우 IR 2개 (서로 독립인 잡음). bands=False 면 simple_dsp 와 같은 IR."""
    rng = np.random.default_rng(seed)
    n = int(sr * max(2.0, 1.1 * rt60 * (BAND_RT[2] if bands else 1)))
    t = np.arange(n) / sr
    sos = signal.butter(2, 5000 / (sr / 2), output="sos")
    if bands:   # 주파수 영역에서 합이 정확히 1 인 대역 가중치 (교차점 앞뒤 반 옥타브씩 코사인으로 넘어감)
        f = np.fft.rfftfreq(n, 1 / sr)
        ramp = lambda fc: np.clip((np.log2(np.maximum(f, 1) / fc) + 0.5), 0, 1)      # fc 반 옥타브 아래 0 → 위 1
        w_lo = np.cos(np.pi / 2 * ramp(BAND_RT[0])) ** 2
        w_hi = np.sin(np.pi / 2 * ramp(BAND_RT[1])) ** 2
        w_mid = 1 - w_lo - w_hi
    irs = []
    for _ in range(2):
        base = signal.sosfilt(sos, rng.standard_normal(n))
        if bands:
            B = np.fft.rfft(base)
            lo, mid, hi = (np.fft.irfft(B * w, n) for w in (w_lo, w_mid, w_hi))
            ir = (lo * np.exp(-6.9 * t / (rt60 * BAND_RT[2])) + mid * np.exp(-6.9 * t / rt60)
                  + hi * np.exp(-6.9 * t / (rt60 * BAND_RT[3])))
        else:
            ir = base * np.exp(-6.9 * t / rt60)
        ir[: int(0.02 * sr)] = 0                         # 20 ms 이전 반사음 없음
        irs.append(ir / (np.sqrt(np.sum(ir ** 2)) + 1e-12))
    return irs


def render(m, preset="r2", seed=0, sr=SR, **over):
    """모노 (T,) → 스테레오 (T, 2). over 로 프리셋 값 바꾸기 (예: width=0.7)."""
    p = {**PRESETS[preset], **over}
    m = np.asarray(m, np.float64)
    y = high_shelf(m, p["shelf_db"], p["shelf_hz"], sr)
    if p["wet"] > 0:
        cl, cr = (signal.fftconvolve(y, ir)[: len(y)] for ir in make_irs(p["rt60"], seed, p["bands"], sr))
    else:
        cl = cr = np.zeros_like(y)
    M = p["dry"] * y + p["wet"] * (cl + cr) / 2
    S = p["width"] * p["wet"] * (cl - cr) / 2
    if p["mono_below"] > 0:                              # 저음 모노 (위상 안 바뀌게 앞뒤로 거르기)
        S = signal.sosfiltfilt(signal.butter(4, p["mono_below"] / (sr / 2), "high", output="sos"), S)
    s = np.stack([M + S, M - S], 1)
    return s / (np.abs(s).max() + 1e-9) * (np.abs(m).max() + 1e-9)     # simple_dsp 와 같은 크기 맞춤


def main():
    ap = argparse.ArgumentParser(description="모노 복원 결과 → 스테레오 (물리 기반 공간 렌더링)")
    ap.add_argument("input", nargs="?")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--preset", default="r2", choices=list(PRESETS))
    ap.add_argument("--dir", help="일괄 처리: <dir>/out/<파일>__<방법>.wav")
    ap.add_argument("--systems", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    for k in ("rt60", "dry", "wet", "width", "shelf_db", "shelf_hz", "mono_below"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=None)
    args = ap.parse_args()
    over = {k: getattr(args, k) for k in ("rt60", "dry", "wet", "width", "shelf_db", "shelf_hz", "mono_below")
            if getattr(args, k) is not None}

    jobs = []
    if args.dir:
        od = Path(args.dir) / "out"
        for s in args.systems:
            src = sorted(od.glob(f"*__{s}.wav"))
            if not src:
                raise SystemExit(f"[오류] {od} 에 *__{s}.wav 가 없습니다 (먼저 eval_real.py 로 변환)")
            jobs += [(p, p.with_name(f"{p.stem}_{args.preset}.wav")) for p in src]
    elif args.input and args.output:
        jobs = [(Path(args.input), Path(args.output))]
    else:
        ap.error("입력·출력 파일 또는 --dir 과 --systems 가 필요합니다")

    for src, dst in jobs:
        x, sr = sf.read(src, dtype="float64", always_2d=True)
        if sr != SR:
            raise SystemExit(f"[오류] {src}: {sr} Hz (44.1 kHz 만 지원, infer.py 출력은 44.1 kHz)")
        sf.write(dst, render(x.mean(1), args.preset, args.seed, **over).astype(np.float32), SR, subtype="PCM_24")
        print(f"  {dst.name}")


if __name__ == "__main__":
    main()
