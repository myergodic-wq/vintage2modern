"""
dsp_baseline.py — 단순 처리 기준선 (torch 없이 쓸 수 있게 분리): 고음 쉘빙 EQ + 좌우가 다른 홀 잔향.
baselines.py 와 restore_baseline.py 가 함께 사용.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

SR = 44100


def simple_dsp(x: np.ndarray, rng_seed: int = 0) -> np.ndarray:
    """모노 (T,) → 스테레오 (T, 2). 고음 쉘빙 EQ + 좌우가 다른 인공 홀 잔향."""
    # 6kHz 위를 +6dB 올리는 쉘빙 (옛 녹음의 답답함 보정)
    A, f0 = 10 ** (6 / 40), 6000
    w0 = 2 * np.pi * f0 / SR
    al = np.sin(w0) / 2 * np.sqrt(2)
    cw = np.cos(w0)
    b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * cw),
         A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * al)]
    a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * cw),
         (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * al]
    y = signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x)
    # RT60 1.8초 홀 잔향, 좌우 서로 다른 잡음 → 넓은 공간감
    rng = np.random.default_rng(rng_seed)
    n = int(SR * 2.0)
    env = np.exp(-6.9 * np.arange(n) / SR / 1.8)
    sos = signal.butter(2, 5000 / (SR / 2), output="sos")
    out = []
    for _ in range(2):
        ir = signal.sosfilt(sos, rng.standard_normal(n)) * env
        ir[: int(0.02 * SR)] = 0                      # 20ms 이전 반사음 없음
        ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
        out.append(0.85 * y + 0.35 * signal.fftconvolve(y, ir)[: len(y)])
    s = np.stack(out, 1)
    return s / (np.abs(s).max() + 1e-9) * (np.abs(x).max() + 1e-9)
