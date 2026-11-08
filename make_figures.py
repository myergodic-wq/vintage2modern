"""
make_figures.py — 논문 그림 3개를 만듦 (PNG + PDF, LaTeX 용)

    fig1_simulator : 열화 시뮬레이터 구조도 (데이터 없이 그림)
    fig2_synthetic : 합성 검증 예시 — 입력(옛날풍 모노) / 모델 출력 / 정답의 스펙트로그램
    fig3_real      : 실제 옛 녹음 — 원본 / v5 출력의 스펙트로그램 (조용한 구간 잡음 바닥 비교 포함)

사용 예 (선생님 PC):
    python make_figures.py --ckpt runs/v5/best.pt --data data/pairs5 ^
        --real real_tests/rach_1920_prelude.mp3 --real-out real_tests/out/rach_1920_prelude__v5_best.wav
결과: docs/fig1_simulator.(png|pdf), docs/fig2_synthetic.(png|pdf), docs/fig3_real.(png|pdf)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import FancyBboxPatch
from scipy import signal

SR = 44100
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {out / name}.png / .pdf")


def fig_simulator(out: Path):
    stages = [
        ("Modern\nstereo\n(target)", "#e7f1fb"),
        ("Mono +\nsmall room", "#f0efec"), ("Period mic\nEQ", "#f0efec"), ("Band limit +\ncompression", "#f0efec"),
        ("Tape:\nsaturation,\nresponse*,\nwow*, hiss*,\nhum", "#f9eeda"),
        ("Transfer\nfilter,\nclicks", "#f0efec"),
        ("78 rpm\nsurface:\ngroove noise*,\nswish, pops,\nscratches", "#f9eeda"),
        ("Uploader\nnoise\nreduction", "#efedff"), ("Lossy codec\nOpus/AAC/\nMP3, 1–2×,\nrealigned", "#efedff"),
        ("Historical-\nstyle mono\n(input)", "#e7f1fb"),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, len(stages) * 1.1); ax.set_ylim(0, 2.2); ax.axis("off")
    for i, (txt, col) in enumerate(stages):
        x = i * 1.1 + 0.05
        ax.add_patch(FancyBboxPatch((x, 0.55), 0.95, 1.1, boxstyle="round,pad=0.03", fc=col, ec="#555555", lw=0.8))
        ax.text(x + 0.475, 1.1, txt, ha="center", va="center", fontsize=6.8, linespacing=1.15)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 1.1, 1.1), xytext=(x + 0.98, 1.1),
                        arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"))
    groups = [(1, 3, "recording"), (4, 6, "carrier (tape or disc)"), (7, 8, "platform")]
    for a, b, name in groups:
        x0, x1 = a * 1.1 + 0.05, b * 1.1 + 1.0
        ax.plot([x0, x1], [0.4, 0.4], color="#777777", lw=0.8)
        ax.text((x0 + x1) / 2, 0.22, name, ha="center", fontsize=7.5, color="#444444")
    ax.text(0.05, 1.95, "* measured: tape response, wow/flutter and hiss from the Akai 4000D dataset; "
            "groove noise harvested from public-domain 78 rpm transfers", fontsize=7, color="#7a4d00")
    save(fig, out, "fig1_simulator")


def spec_db(x, nfft=2048):
    f, t, S = signal.spectrogram(x, SR, nperseg=nfft, noverlap=nfft * 3 // 4)
    return f, t, 10 * np.log10(S + 1e-12)


def panel(ax, x, title, vmin, vmax):
    f, t, S = spec_db(x)
    im = ax.pcolormesh(t, f / 1000, S, vmin=vmin, vmax=vmax, shading="auto", cmap="magma", rasterized=True)
    ax.set_title(title, fontsize=9); ax.set_xlabel("time (s)"); ax.set_ylim(0, 20)
    return im


def load_mono(p):
    from infer import load_audio
    x, sr = load_audio(str(p))
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR); x = signal.resample_poly(x, SR // g, sr // g)
    return x.astype(np.float64)


def fig_synthetic(out: Path, ckpt: str, data: str, device: str, index: int):
    import torch
    from infer import convert
    from model import SpecUNet
    rows = [json.loads(l) for l in open(Path(data) / "meta.jsonl", encoding="utf-8")]
    val = [r for r in rows if r["split"] == "val" and r["degrade"]["preset"] in ("shellac_1925", "acoustic_pre1925")]
    if not val:
        val = [r for r in rows if r["split"] == "val"]
    r = val[index % len(val)]
    x, _ = sf.read(Path(data) / r["input"], dtype="float32")
    y, _ = sf.read(Path(data) / r["target"], dtype="float32", always_2d=True)
    tm = y.mean(1)
    x = x * (np.sqrt(np.mean(tm ** 2)) / (np.sqrt(np.mean(x ** 2)) + 1e-12))    # 학습 때와 같은 레벨 맞춤
    dev = torch.device(device)
    ck = torch.load(ckpt, map_location=dev)
    m = SpecUNet(base=ck["args"].get("base", 32), mono=ck["args"].get("mono", False)).to(dev).eval(); m.load_state_dict(ck["model"])
    o = convert(m, x.astype(np.float32), dev).mean(1)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    vmax = spec_db(tm)[2].max(); vmin = vmax - 90
    for ax, s, t in zip(axes, (x, o, tm), (f"Input ({r['degrade']['preset']}, mono)", "Output (v5, mid)", "Target (mid)")):
        im = panel(ax, s, t, vmin, vmax)
    axes[0].set_ylabel("frequency (kHz)")
    fig.colorbar(im, ax=axes, label="dB", shrink=0.9)
    save(fig, out, "fig2_synthetic")
    print(f"  예시: {r['input']} ({r['source_file']})")


def fig_real(out: Path, real: str, real_out: str, start: float, dur: float):
    x = load_mono(real)
    y, sr = sf.read(real_out, dtype="float32", always_2d=True)
    assert sr == SR
    o = y.mean(1).astype(np.float64)
    a, b = int(start * SR), int((start + dur) * SR)
    x, o = x[a:b], o[a:b]
    x = x * (np.sqrt(np.mean(o ** 2)) / (np.sqrt(np.mean(x ** 2)) + 1e-12))
    fig = plt.figure(figsize=(11, 3.4))
    g = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.9], wspace=0.55)
    ax1, ax2, cax, ax3 = fig.add_subplot(g[0]), fig.add_subplot(g[1]), fig.add_subplot(g[2]), fig.add_subplot(g[3])
    vmax = max(spec_db(x)[2].max(), spec_db(o)[2].max()); vmin = vmax - 90
    panel(ax1, x, f"Original ({Path(real).name[:28]})", vmin, vmax)
    im = panel(ax2, o, "Converted (v5, mid)", vmin, vmax)
    ax2.set_yticklabels([]); ax1.set_ylabel("frequency (kHz)")
    # 조용한 구간(하위 10% 프레임)의 평균 스펙트럼 = 잡음 바닥
    for s, lab, col in ((x, "original", "#d4572a"), (o, "converted", "#2a6fd4")):
        f, t, S = signal.spectrogram(s, SR, nperseg=4096, noverlap=2048)
        e = S.sum(0); q = S[:, e <= np.percentile(e, 10)].mean(1)
        ax3.semilogx(f[1:], 10 * np.log10(q[1:] + 1e-20), label=lab, color=col, lw=1)
    ax3.set_xlim(30, SR / 2); ax3.grid(alpha=0.3); ax3.set_title("Noise floor (quietest 10% of frames)", fontsize=9)
    ax3.set_xlabel("Hz"); ax3.legend(fontsize=8)
    ax3.yaxis.tick_right(); ax3.yaxis.set_label_position("right"); ax3.set_ylabel("dB")
    cb = fig.colorbar(im, cax=cax, label="dB")
    save(fig, out, "fig3_real")


def main():
    ap = argparse.ArgumentParser(description="논문 그림 만들기")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--index", type=int, default=0, help="합성 예시로 쓸 검증 쌍 번호")
    ap.add_argument("--real", default=None)
    ap.add_argument("--real-out", default=None)
    ap.add_argument("--start", type=float, default=0.0, help="실제 녹음 그림 시작 시각(초)")
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    fig_simulator(out)
    if args.ckpt and args.data:
        import torch
        fig_synthetic(out, args.ckpt, args.data, args.device or ("cuda" if torch.cuda.is_available() else "cpu"), args.index)
    else:
        print("그림 2 건너뜀 (--ckpt, --data 필요)")
    if args.real and args.real_out:
        fig_real(out, args.real, args.real_out, args.start, args.dur)
    else:
        print("그림 3 건너뜀 (--real, --real-out 필요)")


if __name__ == "__main__":
    main()
