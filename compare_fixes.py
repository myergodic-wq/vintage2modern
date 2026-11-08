"""
compare_fixes.py — 2026-10-30 수정 사항이 실제로 효과가 있는지 같은 조건으로 비교

세 가지 설정을 똑같은 데이터·시드·스텝 수로 짧게 학습하고, 검증 셋에서 비교합니다.
    A: 이전 방식   (선형 손실만, 입력 음량 그대로)
    B: 음량 맞춤만
    C: 수정 전부   (음량 맞춤 + 압축 스펙트럼 손실)  ← 현재 train.py 기본값

사용 예 (RTX 4090에서 수 분):
    python compare_fixes.py --data data/pairs --steps 1000

보는 법:
    - '거리'가 입력보다 출력에서 많이 줄어들수록 좋음
    - '폭'은 출력이 정답(오른쪽 값)에 가까울수록 좋음
    - '5-10k / 10-16k 오차'는 잘린 고음을 얼마나 되살렸는지. 작아질수록 좋음
C가 A보다 나쁘면 수정을 되돌려야 하니 결과를 꼭 확인하세요.
"""

import argparse

import numpy as np
import torch

from analyze import band_db, lsd, width_db
from dataset import PairDataset
from losses import StereoLoss
from model import SpecUNet

SR = 44100


def run(name, data, steps, device, comp_w, level_match, batch, base, crop):
    torch.manual_seed(0)
    tr = PairDataset(data, "train", crop_sec=crop, level_match=level_match)
    va = PairDataset(data, "val", crop_sec=6, level_match=level_match)
    m = SpecUNet(base=base).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, betas=(0.9, 0.99))
    lf = StereoLoss(comp_w=comp_w).to(device)
    amp = torch.bfloat16 if device.type == "cuda" else None
    g = torch.Generator().manual_seed(0)
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
            print(f"  [{name}] step {s + 1}/{steps}", flush=True)
    m.eval()
    res = []
    with torch.no_grad():
        for i in range(min(len(va), 64)):
            x, y = va[i]
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                yh = m(x[None].to(device))[0].float().cpu().numpy().T
            x, t = x[0].numpy(), y.numpy().T
            bx, by, bt = band_db(x, SR)[0], band_db(yh, SR)[0], band_db(t, SR)[0]
            res.append([lsd(x, t, SR), lsd(yh, t, SR), width_db(yh), width_db(t),
                        abs(bx[3] - bt[3]), abs(by[3] - bt[3]), abs(bx[4] - bt[4]), abs(by[4] - bt[4])])
    return np.mean(res, 0)


def main():
    ap = argparse.ArgumentParser(description="수정 사항 효과 비교 (A/B/C)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--crop", type=float, default=6.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"장치 {device}, 설정당 {args.steps}스텝\n")

    configs = [("A: 이전 방식", 0.0, False), ("B: 음량 맞춤만", 0.0, True), ("C: 수정 전부", 10.0, True)]
    results = []
    for name, comp_w, lm in configs:
        print(f"{name} 학습 중...")
        results.append((name, run(name, args.data, args.steps, device, comp_w, lm,
                                  args.batch, args.base, args.crop)))

    print("\n" + "=" * 100)
    for name, r in results:
        print(f"{name:14} 거리 {r[0]:.3f}→{r[1]:.3f} ({(r[0] - r[1]) / r[0] * 100:+5.1f}%) | "
              f"폭 {r[2]:6.1f} / 정답 {r[3]:5.1f}dB | 5-10k 오차 {r[4]:4.1f}→{r[5]:4.1f}dB | "
              f"10-16k 오차 {r[6]:4.1f}→{r[7]:4.1f}dB")


if __name__ == "__main__":
    main()
