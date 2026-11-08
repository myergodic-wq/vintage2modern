"""
baselines.py — 같은 검증 데이터에서 '기준점'과 모델을 나란히 비교

손실 숫자(예: 1.68)만으로는 좋은지 알 수 없으므로, 기준점과 함께 봅니다.
    1. 아무것도 안 함      입력 모노를 좌우에 그대로 복사 (출발점)
    2. 단순 처리           고음 살리는 EQ + 인공 홀 잔향으로 좌우 넓히기
                           (엔지니어가 손으로 하는 리마스터링을 흉내 낸 것)
    3. 모델                지정한 체크포인트들

사용 예:
    python baselines.py --data data/pairs3 --ckpt runs/v3/best.pt
    python baselines.py --data data/pairs4 --ckpt runs/v3/best.pt runs/v4/best.pt

출력 항목 (모두 낮을수록 좋음, 단 '폭'은 정답에 가까울수록 좋음)
    전체   = 음색 + 0.5 × 공간감   (train.py 의 검증 손실과 같은 계산)
    음색   = 좌우 각 채널의 소리 차이 (로그의 lr)
    공간감 = 좌우 차이(스테레오)의 차이 (로그의 side)
    거리   = 로그 스펙트럼 거리 (analyze.py 와 같음)
    폭     = 스테레오 폭 dB (정답 값과 비교)
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy import signal

from analyze import lsd, width_db
from dsp_baseline import simple_dsp
from dataset import PairDataset
from losses import StereoLoss
from model import SpecUNet

SR = 44100


def main():
    ap = argparse.ArgumentParser(description="기준점 vs 모델 비교")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", nargs="*", default=[], help="비교할 체크포인트(여러 개 가능)")
    ap.add_argument("--n", type=int, default=300, help="사용할 검증 쌍 수 (전부: 0)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    va = PairDataset(args.data, "val", crop_sec=6)
    idx = range(len(va)) if args.n <= 0 else np.linspace(0, len(va) - 1, min(args.n, len(va))).astype(int)
    loss_fn = StereoLoss().to(device)

    models = {}
    for c in args.ckpt:
        ck = torch.load(c, map_location=device)
        m = SpecUNet(base=ck["args"].get("base", 32), mono=ck["args"].get("mono", False)).to(device).eval()
        m.load_state_dict(ck["model"])
        from pathlib import Path
        models[f"모델 {Path(c).parent.name}/{Path(c).name} (step {ck.get('step', '?')})"] = m

    names = ["아무것도 안 함", "단순 처리 (EQ+잔향)"] + list(models)
    acc = {k: [] for k in names}
    with torch.no_grad():
        for k, i in enumerate(idx):
            x, y = va[int(i)]
            xn, t = x[0].numpy().astype(np.float64), y.numpy().T
            outs = {"아무것도 안 함": np.stack([xn, xn], 1), "단순 처리 (EQ+잔향)": simple_dsp(xn, k)}
            for name, m in models.items():
                outs[name] = m(x[None].to(device))[0].float().cpu().numpy().T
            yt = y[None].to(device)
            for name, o in outs.items():
                ot = torch.from_numpy(np.ascontiguousarray(o.T, dtype=np.float32))[None].to(device)
                total, parts = loss_fn(ot, yt)
                acc[name].append([total.item(), parts["lr"], parts["side"], lsd(o, t, SR), width_db(o)])
            acc.setdefault("_target_width", []).append(width_db(t))
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(idx)}", flush=True)

    tw = float(np.mean(acc.pop("_target_width")))
    print(f"\n검증 쌍 {len(idx)}개 기준 (낮을수록 좋음 / 폭은 정답 {tw:.1f}dB 에 가까울수록 좋음)")
    print(f"{'':42}{'전체':>8}{'음색':>8}{'공간감':>8}{'거리':>8}{'폭(dB)':>9}")
    base = np.mean(acc["아무것도 안 함"], 0)
    for name in names:
        r = np.mean(acc[name], 0)
        print(f"{name:42}{r[0]:8.3f}{r[1]:8.3f}{r[2]:8.3f}{r[3]:8.3f}{r[4]:9.1f}"
              + ("" if name == "아무것도 안 함" else f"   (출발점 대비 전체 {(base[0] - r[0]) / base[0] * 100:+.0f}%)"))


if __name__ == "__main__":
    main()
