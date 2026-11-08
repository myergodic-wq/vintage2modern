"""
restore_baseline.py — 기존 복원 모델 기준선: Moliner & Välimäki 2단계 U-Net 잡음 제거 (ICASSP 2022, MIT 라이선스)

논문 비교용. 같은 평가 녹음(real_tests/)을 기존 복원 모델로 처리해 real_tests/out/ 에 저장:
    <이름>__moliner_denoise.wav  : 잡음 제거만 (모노를 좌우에 복사)
    <이름>__moliner_dsp.wav      : 잡음 제거 + 단순 처리(고음 쉘빙 EQ + 좌우가 다른 홀 잔향, baselines.py 와 같은 것)
그 뒤 eval_real.py --systems moliner_denoise moliner_dsp 로 v5 와 같은 지표로 비교.

준비 (한 번만):
    git clone https://github.com/eloimoliner/denoising-historical-recordings ext/dhr
    (체크포인트) https://github.com/eloimoliner/denoising-historical-recordings/releases/download/v0.0/checkpoint.zip
                 → ext/dhr_ckpt/ 에 압축 풀기 (checkpoint, checkpoint.index, checkpoint.data-00000-of-00001)
    TensorFlow 는 별도 가상환경 권장 (PyTorch 환경과 충돌 방지):
      python -m venv tfenv  →  tfenv/Scripts/activate  →  pip install tensorflow tf_keras omegaconf soundfile scipy numpy
사용 예:
    python restore_baseline.py --repo ext/dhr --ckpt ext/dhr_ckpt/checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")       # 원 저장소는 TF 2.3(Keras 2) 체크포인트
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import soundfile as sf
from scipy import signal

from audio_io import SR, load_audio          # torch 불필요 (TensorFlow 전용 가상환경에서 실행 가능)
from dsp_baseline import simple_dsp

EXT = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".webm", ".opus"}


def build(repo: str, ckpt: str):
    import tensorflow as tf
    from omegaconf import OmegaConf
    sys.path.insert(0, str(Path(repo).resolve()))
    import unet
    conf = OmegaConf.load(Path(repo) / "conf" / "conf.yaml")
    model = unet.build_model_denoise(unet_args=conf.unet)
    model.load_weights(ckpt).assert_existing_objects_matched()     # 가중치가 하나라도 안 맞으면 오류
    win, hop = int(conf.stft.win_size), int(conf.stft.hop_size)

    def denoise_seg(x: np.ndarray) -> np.ndarray:
        X = tf.signal.stft(x.astype(np.float32), frame_length=win, frame_step=hop,
                           window_fn=tf.signal.hamming_window, pad_end=True)
        inp = tf.stack([tf.math.real(X), tf.math.imag(X)], -1)[None]
        pred = model(inp, training=False)[0][0]                      # 2단계 출력
        inv = tf.signal.inverse_stft_window_fn(hop, forward_window_fn=tf.signal.hamming_window)
        y = tf.signal.inverse_stft(tf.complex(pred[..., 0], pred[..., 1]), win, hop, window_fn=inv)
        return np.asarray(y)[: len(x)]
    return denoise_seg


def denoise(denoise_seg, x: np.ndarray, seg_sec=5.0, ov=2048) -> np.ndarray:
    """원 저장소와 같은 5초 구간 + 2048 표본 겹침(한 창 페이드)으로 이어 붙임."""
    seg = int(seg_sec * SR)
    hop = seg - ov
    out = np.zeros(len(x)); wsum = np.zeros(len(x))
    fade = np.hanning(2 * ov)
    for s in range(0, max(len(x) - ov, 1), hop):
        chunk = x[s:s + seg]
        n = len(chunk)
        pad = np.concatenate([chunk, np.zeros(seg - n)]) if n < seg else chunk
        y = denoise_seg(pad)[:n]
        w = np.ones(n)
        if s > 0:
            w[:min(ov, n)] = fade[:ov][:min(ov, n)]
        if s + seg < len(x):
            w[-ov:] = np.minimum(w[-ov:], fade[ov:])
        out[s:s + n] += y * w; wsum[s:s + n] += w
    return out / np.maximum(wsum, 1e-6)


def main():
    ap = argparse.ArgumentParser(description="기존 복원 모델(Moliner 2022) 기준선 만들기")
    ap.add_argument("--repo", required=True, help="denoising-historical-recordings 저장소 경로")
    ap.add_argument("--ckpt", required=True, help="체크포인트 접두어 (…/checkpoint)")
    ap.add_argument("--dir", default="real_tests")
    ap.add_argument("--out", default=None, help="기본: <dir>/out")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.dir); out = Path(args.out) if args.out else root / "out"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in EXT and out.resolve() not in p.resolve().parents)
    print(f"평가 파일 {len(files)}개")
    seg_fn = build(args.repo, args.ckpt)
    for p in files:
        o1, o2 = out / f"{p.stem}__moliner_denoise.wav", out / f"{p.stem}__moliner_dsp.wav"
        if o1.exists() and o2.exists() and not args.force:
            print(f"  있음, 건너뜀: {p.name}"); continue
        x, sr = load_audio(str(p))
        if sr != SR:
            from math import gcd
            g = gcd(sr, SR); x = signal.resample_poly(x, SR // g, sr // g)
        x = x.astype(np.float64)
        y = denoise(seg_fn, x)
        pk = np.abs(y).max() + 1e-9
        y = y / pk * min(float(np.abs(x).max()), 0.95) if pk > 0.99 else y
        sf.write(o1, np.stack([y, y], 1).astype(np.float32), SR, subtype="PCM_24")
        sf.write(o2, simple_dsp(y).astype(np.float32), SR, subtype="PCM_24")
        print(f"  {p.name}: {len(x) / SR:.0f}초 → {o1.name}, {o2.name}", flush=True)


if __name__ == "__main__":
    main()
