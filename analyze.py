"""
analyze.py — 검증 샘플의 입력/출력/정답을 숫자와 그림으로 비교

귀로 차이가 애매할 때, 모델이 '어디를' 고쳤는지(혹은 못 고쳤는지) 확인합니다.

사용 예:
    python analyze.py runs/quick/samples/step0005000

출력:
    - 대역별 에너지 표: 입력·출력이 정답에 얼마나 가까운지 (dB)
    - 스테레오 폭: 좌우 차이 에너지 (정답 대비)
    - 정답과의 거리: 입력보다 출력이 가까워졌는지
    - analysis.png: 스펙트로그램 3장 + 평균 스펙트럼 비교
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

BANDS = [(0, 500), (500, 2000), (2000, 5000), (5000, 10000), (10000, 16000), (16000, 22050)]


def mono(x):
    return x.mean(1) if x.ndim == 2 else x


def band_db(x, sr):
    f, p = signal.welch(mono(x), sr, nperseg=4096)
    return [10 * np.log10(p[(f >= lo) & (f < hi)].sum() + 1e-20) for lo, hi in BANDS], f, p


def width_db(x):
    """좌우 차이(Side) 에너지 / 가운데(Mid) 에너지. 모노면 매우 낮음."""
    if x.ndim == 1:
        return -99.0
    m, s = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    return 10 * np.log10((s ** 2).mean() / ((m ** 2).mean() + 1e-20) + 1e-20)


def lsd(a, b, sr):
    """로그 스펙트럼 거리 (낮을수록 정답에 가까움)."""
    _, _, A = signal.stft(mono(a), sr, nperseg=2048)
    _, _, B = signal.stft(mono(b), sr, nperseg=2048)
    n = min(A.shape[1], B.shape[1])
    la, lb = np.log10(np.abs(A[:, :n]) ** 2 + 1e-10), np.log10(np.abs(B[:, :n]) ** 2 + 1e-10)
    return float(np.mean(np.sqrt(np.mean((la - lb) ** 2, axis=0))))


def main():
    ap = argparse.ArgumentParser(description="검증 샘플 분석")
    ap.add_argument("sample_dir", help="예: runs/quick/samples/step0005000")
    args = ap.parse_args()
    d = Path(args.sample_dir)

    ids = sorted({p.name.split("_")[0] for p in d.glob("*_input.wav")})
    if not ids:
        raise SystemExit(f"[오류] {d} 에 *_input.wav 가 없습니다.")

    names = [f"{lo // 1000 if lo >= 1000 else lo}{'k' if lo >= 1000 else ''}-{hi / 1000:g}k" for lo, hi in BANDS]
    print("대역별 정답과의 차이 (dB, 0에 가까울수록 좋음. 음수 = 정답보다 약함)\n")
    header = f"{'':10}" + "".join(f"{n:>11}" for n in names)
    summary = []

    for i in ids:
        x, sr = sf.read(d / f"{i}_input.wav")
        y, _ = sf.read(d / f"{i}_output.wav")
        t, _ = sf.read(d / f"{i}_target.wav")
        bx, _, _ = band_db(x, sr); by, _, _ = band_db(y, sr); bt, _, _ = band_db(t, sr)
        print(f"[샘플 {i}]")
        print(header)
        print(f"{'입력':10}" + "".join(f"{a - c:>+11.1f}" for a, c in zip(bx, bt)))
        print(f"{'출력':10}" + "".join(f"{a - c:>+11.1f}" for a, c in zip(by, bt)))
        lx, ly = lsd(x, t, sr), lsd(y, t, sr)
        wy, wt = width_db(y), width_db(t)
        print(f"  정답과의 거리(LSD): 입력 {lx:.3f} → 출력 {ly:.3f}  "
              f"({'개선' if ly < lx else '악화'} {abs(lx - ly) / lx * 100:.1f}%)")
        print(f"  스테레오 폭: 출력 {wy:.1f} dB, 정답 {wt:.1f} dB  (모노에 가까우면 매우 낮은 값)\n")
        summary.append((lx, ly, wy, wt))

    lx, ly, wy, wt = np.mean(summary, axis=0)
    print("=" * 60)
    print(f"평균 정답과의 거리: 입력 {lx:.3f} → 출력 {ly:.3f}  ({(lx - ly) / lx * 100:+.1f}% 개선)")
    print(f"평균 스테레오 폭: 출력 {wy:.1f} dB / 정답 {wt:.1f} dB")

    # 그림: 첫 샘플
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib 이 없어 그림은 건너뜀: pip install matplotlib)")
        return
    i = ids[0]
    x, sr = sf.read(d / f"{i}_input.wav"); y, _ = sf.read(d / f"{i}_output.wav"); t, _ = sf.read(d / f"{i}_target.wav")
    fig = plt.figure(figsize=(15, 8))
    for k, (title, s) in enumerate([("Input", x), ("Output", y), ("Target", t)]):
        ax = fig.add_subplot(2, 3, k + 1)
        f, tt, S = signal.spectrogram(mono(s), sr, nperseg=2048, noverlap=1536)
        ax.pcolormesh(tt, f / 1000, 10 * np.log10(S + 1e-14), vmin=-130, vmax=-30, shading="auto", cmap="magma")
        ax.set_title(title); ax.set_xlabel("time (s)")
        if k == 0:
            ax.set_ylabel("kHz")
    ax = fig.add_subplot(2, 1, 2)
    for title, s, c in [("Input", x, "#888"), ("Output", y, "#d4572a"), ("Target", t, "#2a6fd4")]:
        _, f, p = band_db(s, sr)
        ax.semilogx(f[1:], 10 * np.log10(p[1:] + 1e-20), label=title, color=c, lw=1.2)
    ax.set_xlim(50, sr / 2); ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Average spectrum")
    plt.tight_layout()
    out = d / "analysis.png"
    plt.savefig(out, dpi=100)
    print(f"그림 저장: {out}")


if __name__ == "__main__":
    main()
