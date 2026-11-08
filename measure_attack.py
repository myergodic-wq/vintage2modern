"""
measure_attack.py — 타건 어택(과도응답)이 뭉개지는지 측정 (정답이 있는 합성 검증 데이터)

정답(현대 녹음)에서 음이 시작되는 순간들을 찾고, 그 순간마다 정답과 모델 출력의 어택을 비교:
    상승 시간      : 포락선이 최고값의 10% → 90% 까지 오르는 시간 (ms, 짧을수록 날카로움)
    최대 상승률    : 0~20ms 안에서 포락선이 가장 빨리 오르는 속도 (최고값의 %/ms, 클수록 날카로움)
    어택/여운 비   : 0~20ms 에너지 ÷ 20~100ms 에너지 (dB, 클수록 '때리는' 느낌)
포락선은 전 대역과 2kHz 이상(타건의 '광채') 두 가지로 봄. 1ms 간격, 2ms 창.

정답을 아는 테스트로 먼저 검증: python measure_attack.py --selftest
사용 예 (선생님 PC):  python measure_attack.py --ckpt runs/v5/best.pt --data data/pairs5 --n 150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
HOP = SR // 1000           # 1ms
WIN = 2 * HOP              # 2ms


def envelope(x: np.ndarray, band: str) -> np.ndarray:
    if band == "hf":
        x = signal.sosfiltfilt(signal.butter(4, 2000 / (SR / 2), "high", output="sos"), x)
    n = (len(x) - WIN) // HOP + 1
    idx = np.arange(WIN)[None, :] + HOP * np.arange(max(n, 0))[:, None]
    return np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-12)          # 1ms 간격 RMS


def onsets(target: np.ndarray, min_gap_ms=80, rise_db=6.0) -> list[int]:
    """
    정답에서 뚜렷한 음 시작 찾기: 10ms 뒤 포락선이 직전 5~50ms 의 **최대값**보다 rise_db 이상 큰 지점.
    (최소값 기준이면 울림의 맥놀이(beating) 골짜기를 음 시작으로 잘못 잡음 — 자체 검증에서 확인)
    """
    e = 20 * np.log10(envelope(target, "full"))
    floor = e.max() - 50
    on, last = [], -10 ** 9
    for i in range(50, len(e) - 100):
        if e[i] < floor:
            continue
        if i < 50:
            continue
        if e[i + 10] - e[i - 50:i - 5].max() > rise_db and e[i] - e[i - 1] > 1.0 and i - last > min_gap_ms:
            on.append(i); last = i
    return on


def attack_stats(x: np.ndarray, on: list[int], band: str) -> np.ndarray:
    """각 시작점(정답 기준, ms 단위 index)에서 [상승시간, 최대 상승률, 어택/여운 비]."""
    e = envelope(x, band)
    out = []
    for i in on:
        a = e[i - 5:i + 40]
        if len(a) < 45:
            continue
        pk = int(np.argmax(a)); peak = a[pk]
        base = a[:5].min()
        lo, hi = base + 0.1 * (peak - base), base + 0.9 * (peak - base)
        t10 = int(np.argmax(a >= lo)); t90 = int(np.argmax(a >= hi))
        rise = max(t90 - t10, 0)
        slope = float(np.diff(a[: 5 + 20]).max() / (peak + 1e-12) * 100)     # 최고값의 % / ms (dB 는 무음 근처에서 불안정)
        att = np.sum(e[i:i + 20] ** 2); sus = np.sum(e[i + 20:i + 100] ** 2) / 4    # 같은 길이로 맞춤
        out.append([rise, slope, 10 * np.log10(att / (sus + 1e-20) + 1e-20)])
    return np.array(out)


def compare(target, outputs: dict, on) -> dict:
    res = {}
    for band in ("full", "hf"):
        t = attack_stats(target, on, band)
        for name, o in outputs.items():
            s = attack_stats(o, on, band)
            n = min(len(t), len(s))
            res[(name, band)] = s[:n] - t[:n]      # 모델 − 정답 (음 하나하나 짝지어)
    return res


def selftest():
    """정답을 아는 테스트: 어택 길이를 아는 가짜 피아노 음 + 일부러 뭉갠 버전."""
    rng = np.random.default_rng(0)
    n = SR * 8
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in range(12):
        st = int((0.3 + 0.6 * k) * SR); f0 = 110 * 2 ** (rng.integers(0, 36) / 12)
        tt = t[: n - st]
        tone = sum(np.sin(2 * np.pi * f0 * h * np.sqrt(1 + 3e-4 * h * h) * tt) / h for h in range(1, 25) if f0 * h < 18000)
        att = 1 - np.exp(-tt / 0.0008)                         # 약 2ms 만에 올라오는 날카로운 어택
        click = rng.standard_normal(len(tt)) * np.exp(-tt / 0.002) * 0.5   # 해머 충격음
        x[st:] += (tone * att + click) * np.exp(-tt * 2.0)
    x /= np.abs(x).max()
    on = onsets(x)
    print(f"찾은 음 시작: {len(on)}개 (넣은 것 12개)")

    def smear(y, ms):                                          # 시간 방향으로 뭉개기 (알려진 양)
        L = int(ms * SR / 1000)
        w = np.hanning(2 * L + 1); w /= w.sum()
        return np.convolve(y, w, mode="same") if L > 0 else y

    def soften_env(y, ms):                                     # 어택만 느리게 (포락선 곱)
        out = y.copy()
        for i in on:
            s = i * HOP
            L = int(ms * SR / 1000)
            seg = out[s:s + 4 * L]
            out[s:s + 4 * L] = seg * (1 - np.exp(-np.arange(len(seg)) / L))
        return out

    outs = {"동일": x, "뭉갬 0.5ms": smear(x, 0.5), "어택 5ms 느리게": soften_env(x, 5), "어택 15ms 느리게": soften_env(x, 15)}
    r = compare(x, outs, on)
    print(f"{'':16}{'대역':>5}{'상승시간 차(ms)':>16}{'상승률 차(%/ms)':>18}{'어택/여운 차(dB)':>18}")
    for name in outs:
        for band in ("full", "hf"):
            d = r[(name, band)]
            print(f"{name:16}{band:>5}{np.median(d[:, 0]):16.1f}{np.median(d[:, 1]):18.2f}{np.median(d[:, 2]):18.2f}")
    d0 = r[("동일", "full")]; d5 = r[("어택 5ms 느리게", "full")]; d15 = r[("어택 15ms 느리게", "full")]
    ok = (np.abs(d0).max() < 1e-9 and np.median(d5[:, 0]) > 2 and np.median(d15[:, 0]) > np.median(d5[:, 0])
          and np.median(d15[:, 1]) < np.median(d5[:, 1]) < 0 and np.median(d15[:, 2]) < np.median(d5[:, 2]) < 0
          and len(on) == 12)
    print("\n자체 검증:", "통과" if ok else "실패")
    return ok


def main():
    ap = argparse.ArgumentParser(description="타건 어택 보존 측정")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=150, help="검증 쌍 수")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    import torch
    from infer import convert
    from model import SpecUNet
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = {}
    for c in args.ckpt:
        ck = torch.load(c, map_location=dev)
        m = SpecUNet(base=ck["args"].get("base", 32), mono=ck["args"].get("mono", False)).to(dev).eval(); m.load_state_dict(ck["model"])
        models[f"{Path(c).parent.name}_{Path(c).stem}"] = m

    rows = [json.loads(l) for l in open(Path(args.data) / "meta.jsonl", encoding="utf-8")]
    val = [r for r in rows if r["split"] == "val"]
    rs = np.random.default_rng(0)
    val = [val[i] for i in sorted(rs.choice(len(val), min(args.n, len(val)), replace=False))]
    acc, n_on = {}, 0
    for k, r in enumerate(val):
        x, _ = sf.read(Path(args.data) / r["input"], dtype="float32")
        y, _ = sf.read(Path(args.data) / r["target"], dtype="float32", always_2d=True)
        tm = y.mean(1).astype(np.float64)
        x = x * (np.sqrt(np.mean(tm ** 2)) / (np.sqrt(np.mean(x ** 2)) + 1e-12))      # 학습 때와 같은 레벨 맞춤
        on = onsets(tm)
        if not on:
            continue
        n_on += len(on)
        outs = {"입력(열화)": x.astype(np.float64)}
        for name, m in models.items():
            outs[name] = convert(m, x, dev).mean(1).astype(np.float64)
        for key, d in compare(tm, outs, on).items():
            acc.setdefault(key, []).append(d)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(val)}", flush=True)

    if n_on == 0:
        raise SystemExit("[오류] 검증 정답에서 뚜렷한 음 시작을 찾지 못했습니다.")
    print(f"\n검증 쌍 {len(val)}개, 음 시작 {n_on}개 — 정답 대비 차이의 중앙값 [하위25%, 상위75%]")
    print("  상승시간 차 > 0 = 어택이 느려짐(뭉개짐) / 상승률 차 < 0 = 덜 날카로움 / 어택·여운 차 < 0 = 덜 '때림'")
    for band in ("full", "hf"):
        print(f"\n[{'전 대역' if band == 'full' else '2kHz 이상 (타건의 광채)'}]")
        print(f"{'':16}{'상승시간 차(ms)':>22}{'상승률 차(%/ms)':>24}{'어택/여운 차(dB)':>24}")
        for name in ["입력(열화)"] + list(models):
            d = np.concatenate(acc[(name, band)])
            cells = "".join(f"{np.median(d[:, j]):8.2f} [{np.percentile(d[:, j], 25):6.2f},{np.percentile(d[:, j], 75):6.2f}]"
                            for j in range(3))
            print(f"{name:16}{cells}")


if __name__ == "__main__":
    main()
