"""
compare_mono.py — 스테레오 모델의 '가운데' vs 모노 복원 모델 비교 (2026-11-03, PROGRESS r23)

배경: 청취 2차에서 'v5 가운데 + 단순 처리'가 기존 전문 모델과 대등 → 복원은 모노 모델이, 공간감은 렌더링 단계가 맡기로 함.
질문: 가운데만 직접 학습한 모노 모델(v7)이 v5 의 (L+R)/2 보다 나은가?
둘 다 '가운데'만 정답 가운데((L+R)/2)와 비교합니다 (공간감은 비교 대상 아님).

두 가지 쓰임:
  1) 짧은 비교 학습 — 같은 데이터·seed·배치 순서로 스테레오 방식과 모노 방식을 N 스텝씩 학습한 뒤 비교
       python compare_mono.py --data data/pairs5 --steps 1000
  2) 학습이 끝난 체크포인트 비교 (학습 없음)
       python compare_mono.py --data data/pairs5 --steps 0 --ckpt runs/v5/best.pt runs/v7/best.pt

보는 법 (모두 정답 가운데 기준):
  다중해상도 = 학습 손실과 같은 계산 (낮을수록 좋음)
  거리      = 로그 스펙트럼 거리 LSD (낮을수록 좋음)
  대역 오차 = |출력 - 정답| dB, 곡마다 계산한 뒤 평균 (낮을수록 좋음)
  고음 수준 = 출력 - 정답 dB 의 평균 (음수 = 정답보다 약함. 청취에서 지적된 '답답함')
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from analyze import BANDS, band_db, lsd
from dataset import PairDataset
from losses import MonoLoss, StereoLoss
from model import SpecUNet

SR = 44100
HI = [3, 4]                                   # BANDS 의 5-10k, 10-16k


def train_short(mono, data, steps, device, batch, base, crop):
    torch.manual_seed(0)                      # 두 설정 모두 같은 초기값(출력층 제외)·같은 배치 순서
    tr = PairDataset(data, "train", crop_sec=crop)
    m = SpecUNet(base=base, mono=mono).to(device)
    torch.manual_seed(1)                      # 출력층 크기(4/2)에 따라 쓰인 난수 개수가 달라도, 이후 자르기·음량 증강을 똑같이
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    lf = (MonoLoss() if mono else StereoLoss()).to(device)
    amp = torch.bfloat16 if device.type == "cuda" else None
    g = torch.Generator().manual_seed(0)
    name = "모노" if mono else "스테레오"
    for s in range(steps):
        idx = torch.randint(0, len(tr), (batch,), generator=g)
        xb, yb = zip(*[tr[int(i)] for i in idx])
        x, y = torch.stack(xb).to(device), torch.stack(yb).to(device)
        for pg in opt.param_groups:
            pg["lr"] = 2e-4 * min(1.0, (s + 1) / 100)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            yh = m(x)
        l, _ = lf(yh, y)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % max(1, steps // 5) == 0:
            print(f"  [{name}] step {s + 1}/{steps}  loss {l.item():.4f}", flush=True)
    return m.eval()


@torch.no_grad()
def evaluate(model, va, idx, device):
    """model=None 이면 입력 그대로 (출발점). 반환: [다중해상도, 거리, 2-5k, 5-10k, 10-16k 오차, 5-10k, 10-16k 수준]"""
    mr = MonoLoss()                           # 평가 계산은 모두 CPU (모델 출력을 .cpu() 로 옮기므로). GPU 에 두면 장치 불일치 오류
    amp = torch.bfloat16 if device.type == "cuda" else None
    rows = []
    for i in idx:
        x, y = va[int(i)]
        if model is None:
            yh = x.repeat(2, 1)[None]
        else:
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                yh = model(x[None].to(device)).float().cpu()
        l, _ = mr(yh, y[None])
        o, t = yh[0].mean(0).numpy(), y.mean(0).numpy()      # 가운데끼리
        bo, bt = np.array(band_db(o, SR)[0]), np.array(band_db(t, SR)[0])
        d = bo - bt
        rows.append([l.item(), lsd(o, t, SR), abs(d[2]), abs(d[3]), abs(d[4]), d[3], d[4]])
    return np.mean(rows, 0)


def main():
    ap = argparse.ArgumentParser(description="스테레오 모델의 가운데 vs 모노 복원 모델")
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=1000, help="짧은 비교 학습 스텝 (0 = 학습 없이 --ckpt 만 비교)")
    ap.add_argument("--ckpt", nargs="*", default=[], help="함께 비교할 체크포인트")
    ap.add_argument("--n", type=int, default=300, help="검증 쌍 수 (전부: 0)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--crop", type=float, default=6.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    va = PairDataset(args.data, "val", crop_sec=6)
    idx = range(len(va)) if args.n <= 0 else np.linspace(0, len(va) - 1, min(args.n, len(va))).astype(int)
    print(f"장치 {device}, 검증 쌍 {len(idx)}개\n")

    results = [("입력 그대로", evaluate(None, va, idx, device))]
    if args.steps > 0:
        for mono in (False, True):
            m = train_short(mono, args.data, args.steps, device, args.batch, args.base, args.crop)
            results.append((f"{'모노' if mono else '스테레오'} {args.steps}스텝", evaluate(m, va, idx, device)))
            del m
    for c in args.ckpt:
        ck = torch.load(c, map_location=device)
        m = SpecUNet(base=ck["args"].get("base", 32), mono=ck["args"].get("mono", False)).to(device).eval()
        m.load_state_dict(ck["model"])
        kind = "모노" if ck["args"].get("mono", False) else "스테레오→가운데"
        results.append((f"{Path(c).parent.name} step {ck.get('step', '?')} ({kind})", evaluate(m, va, idx, device)))
        del m

    hb = [f"{BANDS[k][0] // 1000}-{BANDS[k][1] // 1000}k" for k in (2, 3, 4)]
    print("\n가운데 기준, 정답 가운데와 비교 (낮을수록 좋음 / 수준은 0 에 가까울수록, 음수 = 정답보다 약함)")
    print(f"{'':34}{'다중해상도':>10}{'거리':>8}" + "".join(f"{h + ' 오차':>11}" for h in hb)
          + f"{'5-10k 수준':>12}{'10-16k 수준':>12}")
    for name, r in results:
        print(f"{name:34}{r[0]:10.4f}{r[1]:8.3f}{r[2]:11.1f}{r[3]:11.1f}{r[4]:11.1f}{r[5]:+12.1f}{r[6]:+12.1f}")


if __name__ == "__main__":
    main()
