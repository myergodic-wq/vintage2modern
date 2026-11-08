"""
infer.py — 학습된 모델로 옛 녹음을 변환

사용 예:
    python infer.py runs/baseline/best.pt 옛녹음.flac 변환결과.wav
    python infer.py runs/baseline/best.pt 옛녹음.webm 변환결과.wav   (mp3·webm·m4a·opus 는 ffmpeg 필요)

긴 음원은 겹치는 구간으로 나눠 처리한 뒤 부드럽게 이어 붙입니다.
"""

from __future__ import annotations

import argparse
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from audio_io import load_audio  # noqa: F401  (예전처럼 infer.load_audio 로도 쓸 수 있게)
from model import SpecUNet

SR = 44100


@torch.no_grad()
def convert(model, x: np.ndarray, device, chunk_sec=10.0, overlap_sec=1.0) -> np.ndarray:
    """x: 모노 (T,) 44.1kHz → 스테레오 (T, 2)"""
    T = len(x)
    chunk, ov = int(chunk_sec * SR), int(overlap_sec * SR)
    hop = chunk - ov
    out = np.zeros((T, 2), np.float32)
    wsum = np.zeros(T, np.float32)
    fade = np.linspace(0, 1, ov, dtype=np.float32)

    for s in range(0, max(T - ov, 1), hop):
        seg = x[s:s + chunk]
        n = len(seg)
        y = model(torch.from_numpy(seg[None, None].astype(np.float32)).to(device))[0].float().cpu().numpy().T
        w = np.ones(n, np.float32)
        if s > 0:
            w[:min(ov, n)] = fade[:min(ov, n)]
        if s + chunk < T:
            w[-ov:] = np.minimum(w[-ov:], fade[::-1])
        out[s:s + n] += y * w[:, None]
        wsum[s:s + n] += w
    return out / np.maximum(wsum, 1e-6)[:, None]


def main():
    ap = argparse.ArgumentParser(description="옛 모노 녹음 → 현대 스테레오 변환")
    ap.add_argument("checkpoint")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--chunk", type=float, default=10.0)
    ap.add_argument("--spatial", choices=["r2", "hall", "none"], default=None,
                    help="모노 복원 모델(v7~)의 공간 렌더링 프리셋 (기본 r2, none = 모노 그대로). 스테레오 모델(v5 이하)에는 쓰지 않음")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=device)
    model = SpecUNet(base=ck["args"].get("base", 32), mono=ck["args"].get("mono", False)).to(device).eval()
    model.load_state_dict(ck["model"])

    x, sr = load_audio(args.input)                  # 스테레오 파일이어도 모노로
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    peak_in = float(np.abs(x).max() + 1e-9)

    y = convert(model, x, device, args.chunk)
    preset = None
    if model.mono:                                  # 복원(모노) → 물리 기반 공간 렌더링 (PROGRESS 4-6 절)
        preset = args.spatial or "r2"
        if preset != "none":
            from spatial import render
            y = render(y.mean(1), preset).astype(np.float32)
    elif args.spatial:
        print("[안내] 스테레오 모델이라 --spatial 은 쓰지 않습니다")
    y = y / (np.abs(y).max() + 1e-9) * min(peak_in, 0.95) if np.abs(y).max() > 0.99 else y
    sf.write(args.output, y, SR, subtype="PCM_24")
    print(f"저장: {args.output}  ({len(y) / SR:.1f}초, 44.1kHz" + (f", 공간 프리셋 {preset}" if preset else "") + ")")


if __name__ == "__main__":
    main()
