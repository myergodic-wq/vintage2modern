"""
train.py — 기준선 모델 학습

사용 예 (RTX 4090):
    python train.py --data data/pairs --out runs/baseline

중단 후 같은 명령을 다시 실행하면 마지막 체크포인트에서 이어서 학습합니다.
검증 때마다 runs/.../samples/ 에 (입력, 출력, 정답) 음원을 저장하니 귀로 확인하세요.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader

from dataset import PairDataset
from losses import MonoLoss, StereoLoss
from model import SpecUNet


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def validate(model, loader, loss_fn, device, amp_dtype, sample_dir=None, n_samples=3, sr=44100):
    model.eval()
    tot, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            y_hat = model(x)
        loss, _ = loss_fn(y_hat, y)
        tot += loss.item() * x.shape[0]; n += x.shape[0]
        if sample_dir and i < n_samples:
            sample_dir.mkdir(parents=True, exist_ok=True)
            sf.write(sample_dir / f"{i}_input.wav", x[0, 0].cpu().numpy(), sr)
            sf.write(sample_dir / f"{i}_output.wav", y_hat[0].float().T.cpu().numpy().clip(-1, 1), sr)
            sf.write(sample_dir / f"{i}_target.wav", y[0].T.cpu().numpy(), sr)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="기준선 모델 학습 (모노 → 스테레오)")
    ap.add_argument("--data", required=True, help="make_pairs.py 출력 폴더")
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=float, default=6.0, help="학습 구간 길이(초)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--base", type=int, default=32, help="모델 폭 (작게: 16, 크게: 48)")
    ap.add_argument("--mono", action="store_true",
                    help="모노 복원 모델 (v7~): 정답의 가운데((L+R)/2)만 학습. 공간감은 별도 렌더링 단계")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--no-amp", action="store_true", help="bf16 끄기 (디버깅용)")
    ap.add_argument("--compile", action="store_true", help="torch.compile 사용 (더 빠를 수 있음)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None,
                    help="학습 난수 고정 (모델 초기값·데이터 순서·잘라낼 위치). 논문의 seed 반복 실험용. 기본: 고정 안 함")
    args = ap.parse_args()
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        print(f"학습 seed: {args.seed}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and not args.no_amp) else None
    torch.backends.cudnn.benchmark = True

    train_ds = PairDataset(args.data, "train", crop_sec=args.crop)
    val_ds = PairDataset(args.data, "val", crop_sec=args.crop)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                          drop_last=True, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers)
    print(f"학습 쌍 {len(train_ds)}개, 검증 쌍 {len(val_ds)}개, 장치 {device}, bf16={amp_dtype is not None}")

    model = SpecUNet(base=args.base, mono=args.mono).to(device)
    print(f"파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M" + (" (모노 복원 모델)" if args.mono else ""))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    loss_fn = (MonoLoss() if args.mono else StereoLoss()).to(device)

    step, best = 0, float("inf")
    ckpt = out / "last.pt"
    if ckpt.exists():
        s = torch.load(ckpt, map_location=device)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        step, best = s["step"], s.get("best", best)
        print(f"체크포인트에서 이어서 학습: step {step}")

    fwd = torch.compile(model) if args.compile else model
    log_f = open(out / "log.csv", "a", newline="")
    log = csv.writer(log_f)
    if step == 0:
        log.writerow(["step", "train_loss", "lr_loss", "side_loss", "val_loss", "lr", "sec_per_step"])

    model.train()
    t0, it = time.time(), iter(train_dl)
    while step < args.steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dl); x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        lr = lr_at(step, args.steps, args.warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            y_hat = fwd(x)
        loss, parts = loss_fn(y_hat, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % args.log_every == 0:
            sps = (time.time() - t0) / args.log_every; t0 = time.time()
            eta_h = sps * (args.steps - step) / 3600
            print(f"step {step:>7}  loss {loss.item():.4f}  (lr {parts['lr']:.4f} side {parts['side']:.4f})"
                  f"  {sps:.2f}s/step  남은시간 {eta_h:.1f}h")
            log.writerow([step, f"{loss.item():.5f}", f"{parts['lr']:.5f}", f"{parts['side']:.5f}", "", f"{lr:.2e}", f"{sps:.3f}"])
            log_f.flush()

        if step % args.val_every == 0 or step == args.steps:
            v = validate(model, val_dl, loss_fn, device, amp_dtype, out / "samples" / f"step{step:07d}")
            print(f"  검증 손실 {v:.4f}" + ("  (최고 기록, best.pt 저장)" if v < best else ""))
            log.writerow([step, "", "", "", f"{v:.5f}", "", ""]); log_f.flush()
            state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                     "best": min(best, v), "args": vars(args)}
            torch.save(state, ckpt)
            if v < best:
                best = v
                torch.save({"model": model.state_dict(), "step": step, "args": vars(args)}, out / "best.pt")

    log_f.close()
    print("학습 완료")


if __name__ == "__main__":
    main()
