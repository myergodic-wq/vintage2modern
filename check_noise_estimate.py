"""
check_noise_estimate.py — 자기 학습(RemixIT형) 가능성 검증: '입력 − 교사 출력' 이 진짜 잡음을 얼마나 잘 추정하나

정답을 아는 테스트:
  깨끗한 정답 → (잡음 없는) 옛 녹음 체인 → c   +   알려진 잡음 n (테이프 히스·험, 크래클, 78rpm 표면)  →  x = c + n
  교사 모델 T 로 x 를 변환 → 가운데 신호 m
  잡음 추정 두 가지:
    A. n̂ = x − g·m                 (전체 이득 하나만 맞춤)
    B. n̂ = x − H(m)                (주파수별 전달 함수 H 를 파일 전체에서 최소제곱으로 맞춤: 교사가 바꾼 음색·대역을
                                    입력 쪽으로 되돌림. 잡음은 m 과 상관이 없으니 H 에 거의 안 들어감)
  잡음 추정 정확도 = 10·log10(|n|² / |n̂ − n|²)  [dB]. 클수록 좋음. 0 dB = 아무 추정도 안 한 것과 같음.

판단 기준(사전에 정함): 중앙값 10 dB 이상 → RemixIT형 진행 / 3~10 dB → 부분적 / 3 dB 미만 → Noisy Student형으로 전환

사용 예 (선생님 PC, 실제 검증 정답 사용):
    python check_noise_estimate.py --ckpt runs/v5/best.pt --data data/pairs5 --n 40
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

import degrade as D
from infer import SR, convert
from model import SpecUNet

NOISY = ("shellac_1925", "acoustic_pre1925", "tape_1950s")


def chain_with_known_noise(y: np.ndarray, sr: int, preset: str, seed: int):
    """degrade() 와 같은 순서이되, 잡음을 따로 기록. 손실 압축·업로더 잡음 제거(비선형)는 뺌."""
    p = D.PRESETS[preset]
    rng = np.random.default_rng(seed)
    c, _ = D.to_mono(y.astype(np.float64), rng)
    c, _ = D.small_room(c, sr, rng, p)
    c, _ = D.mic_color(c, sr, rng, p)
    c, lm = D.bandlimit(c, sr, rng, p)
    c, _ = D.compress(c, sr, rng, p)
    c, _ = D.tape_saturate(c, rng, p)
    c, _ = D.wow_flutter(c, sr, rng, p, None)
    noisy, _ = D.tape_noise(c, sr, rng, p, None)
    n = noisy - c
    r2 = copy.deepcopy(rng)                                   # 같은 필터를 정답 쪽과 잡음 쪽에 똑같이
    c, _ = D.transfer_filter(c, sr, rng, lm["lowpass_hz"])
    n, _ = D.transfer_filter(n, sr, r2, lm["lowpass_hz"])
    out, _ = D.disc_crackle(c + n, sr, rng, p)
    out, _ = D.disc_surface(out, sr, rng, p)
    n = out - c
    k = 0.9 / (np.max(np.abs(out)) + 1e-9)
    return (c * k).astype(np.float32), (n * k).astype(np.float32)


def fit_H(x: np.ndarray, m: np.ndarray, nfft=4096):
    """주파수별 복소 이득 H = Σ X·M* / Σ |M|²  (프레임 전체), 1/3 옥타브 정도로 매끄럽게."""
    _, _, X = signal.stft(x, SR, nperseg=nfft)
    _, _, M = signal.stft(m, SR, nperseg=nfft)
    H = (X * M.conj()).sum(1) / ((np.abs(M) ** 2).sum(1) + 1e-12)
    H = signal.savgol_filter(H.real, 31, 2) + 1j * signal.savgol_filter(H.imag, 31, 2)
    _, y = signal.istft(H[:, None] * M, SR, nperseg=nfft)
    return y[: len(x)]


def snr(n, e):
    return 10 * np.log10(np.sum(n ** 2) / (np.sum((e - n) ** 2) + 1e-20))


def band(z, lo, hi):
    sos = signal.butter(4, [max(lo, 20) / (SR / 2), min(hi, SR / 2 * 0.99) / (SR / 2)], "band", output="sos")
    return signal.sosfiltfilt(sos, z)


def main():
    ap = argparse.ArgumentParser(description="자기 학습용 잡음 추정 정확도 검증 (정답을 아는 테스트)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=None, help="쌍 폴더 (검증용 정답 사용). 없으면 --wav 사용")
    ap.add_argument("--wav", nargs="*", default=[], help="깨끗한 스테레오 파일 직접 지정")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.data:
        rows = [json.loads(l) for l in open(Path(args.data) / "meta.jsonl", encoding="utf-8")]
        tg = sorted({r["target"] for r in rows if r["split"] == "val"})
        files = [Path(args.data) / t for t in tg]
    else:
        files = [Path(w) for w in args.wav]
    rs = np.random.default_rng(0)
    files = [files[i] for i in sorted(rs.choice(len(files), min(args.n, len(files)), replace=False))]
    if not files:
        raise SystemExit("[오류] 검증할 파일이 없습니다.")

    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=dev)
    model = SpecUNet(base=ck["args"].get("base", 32), mono=ck["args"].get("mono", False)).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"교사 {args.ckpt} (step {ck.get('step')}), 파일 {len(files)}개, 장치 {dev}\n")

    res = []
    for i, f in enumerate(files):
        y, sr = sf.read(f, dtype="float32", always_2d=True)
        assert sr == SR
        preset = NOISY[i % len(NOISY)]
        c, n = chain_with_known_noise(y, SR, preset, seed=1000 + i)
        x = c + n
        m = convert(model, x, dev).mean(1).astype(np.float64)
        g = float(np.dot(x, m) / (np.dot(m, m) + 1e-12))
        eA = x - g * m
        eB = x - fit_H(x, m)
        r = {"preset": preset, "noise_db": 10 * np.log10(np.sum(n ** 2) / np.sum(c ** 2)),
             "A": snr(n, eA), "B": snr(n, eB)}
        for lo, hi, name in ((20, 300, "B<300"), (300, 4000, "B300-4k"), (4000, 22000, "B>4k")):
            r[name] = snr(band(n, lo, hi), band(eB, lo, hi))
        res.append(r)
        print(f"  [{i + 1}/{len(files)}] {preset:17s} 잡음/음악 {r['noise_db']:6.1f} dB   A {r['A']:5.1f}   B {r['B']:5.1f}   "
              f"(B 대역별 {r['B<300']:5.1f} / {r['B300-4k']:5.1f} / {r['B>4k']:5.1f}) dB", flush=True)

    print("\n잡음 추정 정확도 (dB, 클수록 좋음, 0 = 추정 못 함) — 중앙값 [하위 25%]")
    for k in ("A", "B", "B<300", "B300-4k", "B>4k"):
        v = np.array([r[k] for r in res])
        print(f"  {k:8s} {np.median(v):6.1f}  [{np.percentile(v, 25):6.1f}]")
    for pr in NOISY:
        v = np.array([r["B"] for r in res if r["preset"] == pr])
        if len(v):
            print(f"  B, {pr:17s} {np.median(v):6.1f}")
    b = float(np.median([r["B"] for r in res]))
    verdict = "RemixIT형 진행" if b >= 10 else ("부분적 — 대역·프리셋별로 판단" if b >= 3 else "Noisy Student형으로 전환")
    print(f"\n판단 (사전 기준: ≥10 진행 / 3~10 부분적 / <3 전환): B 중앙값 {b:.1f} dB → {verdict}")


if __name__ == "__main__":
    main()
