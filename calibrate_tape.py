"""
calibrate_tape.py — Aalto 테이프 녹음기 데이터셋에서 '진짜 테이프'의 성질을 측정

측정하는 것 (테이프 속도별: 7.5 IPS / 3.75 IPS)
    1. 테이프 히스 모음      Silence_* 폴더의 무음 녹음 → 실제 잡음 조각
    2. 주파수 특성          ReelToReel_* 짝의 소리 채널(왼쪽) → 테이프가 음색을 바꾸는 곡선
    3. 속도 흔들림 궤적     ReelToReel_* 짝의 펄스 채널(오른쪽) → 실제 wow/flutter

결과: assets/tape_profile.npz (degrade.py 가 있으면 자동으로 사용)

사용 예:
    python calibrate_tape.py C:\\aalto

데이터 출처 (CC BY 4.0, 출처 표시 필요):
    O. Mikkonen, A. Wright, E. Moliner, V. Välimäki,
    "Neural Modeling of Magnetic Tape Recorders", DAFx 2023.
    Dataset: https://doi.org/10.5281/zenodo.8026271

데이터셋의 .npy 궤적 파일은 열지 않습니다 (pickle 형식이라 열면 코드가 실행될 수 있음).
궤적은 펄스 채널에서 직접 계산합니다.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
SPEEDS = ("7.5", "3.75")


def find_folders(root: Path, prefix: str) -> dict[str, list[Path]]:
    """속도별로 해당 폴더 목록. (_SPLIT 은 같은 녹음을 자른 복사본이라 제외)"""
    out = {s: [] for s in SPEEDS}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix) or d.name.endswith("_SPLIT"):
            continue
        m = re.search(r"IPS\[([\d.]+)\]", d.name)
        if m and m.group(1) in out:
            out[m.group(1)].append(d)
    return out


def pairs_in(folder: Path) -> list[tuple[Path, Path]]:
    res = []
    for inp in sorted(folder.rglob("input_*.wav")):
        tgt = inp.with_name(inp.name.replace("input_", "target_", 1))
        if tgt.exists():
            res.append((inp, tgt))
    return res


def read(path: Path, seconds: float | None, start_sec: float = 0.0) -> np.ndarray:
    info = sf.info(path)
    if info.samplerate != SR:
        raise SystemExit(f"[오류] 샘플레이트가 {info.samplerate} 입니다 (예상 {SR}): {path}")
    start = int(start_sec * SR)
    frames = -1 if seconds is None else int(seconds * SR)
    x, _ = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    return x


# ---------------------------------------------------------------------------
# 1. 테이프 히스
# ---------------------------------------------------------------------------

def noise_bank(folders: list[Path], n_clips: int, clip_sec: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """무음 녹음의 target 에서 잡음 조각을 고르게 뽑음. 반환: (조각들 int16, 평균 잡음 PSD)"""
    files = [t for f in folders for _, t in pairs_in(f)]
    if not files:
        files = [p for f in folders for p in sorted(f.rglob("target_*.wav"))]
    if not files:
        return np.zeros((0, 0), np.int16), None
    clip = int(clip_sec * SR)
    clips, psds = [], []
    sos = signal.butter(2, 20 / (SR / 2), btype="high", output="sos")   # 직류·초저역 제거
    for i in range(n_clips):
        f = files[i % len(files)]
        dur = sf.info(f).duration
        st = float(rng.uniform(1.0, max(1.0, dur - clip_sec - 1.0)))
        x = read(f, clip_sec, st)[:, 0].astype(np.float64)
        x = signal.sosfiltfilt(sos, x)
        if len(x) < clip:
            continue
        psds.append(signal.welch(x, SR, nperseg=8192)[1])
        # 잡음 레벨이 -60dBFS 근처라 그대로 16비트로 담으면 해상도가 부족 → 정규화해서 저장
        clips.append(np.clip(x / (np.std(x) + 1e-12) * 0.1, -1, 1))
    bank = (np.stack(clips) * 32767).astype(np.int16)
    return bank, np.mean(psds, axis=0)


# ---------------------------------------------------------------------------
# 2. 주파수 특성
# ---------------------------------------------------------------------------

def freq_response(pairs, noise_psd, seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    |H(f)|^2 = (Pyy - Pnn) / Pxx  (파워 스펙트럼 비율이라 지연·흔들림과 무관)
    반환: (주파수, 이득 dB (200~2000Hz 평균을 0으로 맞춤), 신뢰할 수 있는 구간 여부)
    """
    Pxx, Pyy = 0.0, 0.0
    for inp, tgt in pairs:
        x = read(inp, seconds)[:, 0].astype(np.float64)
        y = read(tgt, seconds)[:, 0].astype(np.float64)
        n = min(len(x), len(y))
        f, px = signal.welch(x[:n], SR, nperseg=8192)
        _, py = signal.welch(y[:n], SR, nperseg=8192)
        Pxx, Pyy = Pxx + px, Pyy + py
    Pxx, Pyy = Pxx / len(pairs), Pyy / len(pairs)
    Pnn = noise_psd if noise_psd is not None else np.zeros_like(Pyy)
    H2 = np.maximum(Pyy - Pnn, Pyy * 1e-3) / (Pxx + 1e-30)
    # 출력이 잡음의 2배(3dB) 이상인 곳만 신뢰. 가짜 데이터 검증에서 2배가 가장 정확했음
    # (1.25배: 잡음 때문에 띄엄띄엄 유효점이 생김 / 4배: 측정 가능한 구간을 너무 일찍 포기)
    ok = (Pyy > 2 * Pnn) & (Pxx > 1e-20)
    # 저음부터 끊김 없이 이어진 구간까지만 사용: 1/6 옥타브 창 안의 유효 비율이 절반 아래로
    # 떨어지는 첫 주파수(1kHz 이상)를 측정 상한으로
    cutoff = f[-1]
    for i in np.where(f >= 1000)[0]:
        win = (f >= f[i] / 2 ** (1 / 12)) & (f <= f[i] * 2 ** (1 / 12))
        if ok[win].mean() < 0.5:
            cutoff = f[i]
            break
    valid = ok & (f > 20) & (f < cutoff)
    g = 10 * np.log10(H2 + 1e-30)
    ref = (f >= 200) & (f <= 2000) & valid
    g -= np.median(g[ref])
    # 1/6 옥타브로 부드럽게 (측정 잡음 제거)
    gs = np.copy(g)
    for i, fi in enumerate(f):
        if fi <= 0:
            continue
        band = (f >= fi / 2 ** (1 / 12)) & (f <= fi * 2 ** (1 / 12)) & valid
        if band.any():
            gs[i] = np.mean(g[band])
    # 측정 구간 밖: 아래는 가장 가까운 값, 위는 마지막 반 옥타브의 기울기로 이어감
    # (내려가는 경우만, -60dB 하한). 측정 상한 위는 원리적으로 잡음에 묻혀 알 수 없는 영역.
    idx = np.where(valid)[0]
    if len(idx):
        gs[: idx[0]] = gs[idx[0]]
        last = f[idx[-1]]
        sel = (f >= last / np.sqrt(2)) & (f <= last) & valid
        slope = np.polyfit(np.log2(f[sel]), gs[sel], 1)[0] if sel.sum() > 3 else 0.0
        hi = f > last
        gs[hi] = np.maximum(gs[idx[-1]] + min(slope, 0.0) * np.log2(f[hi] / last), -60.0)
    return f, gs, valid


# ---------------------------------------------------------------------------
# 3. 속도 흔들림 (펄스 채널)
# ---------------------------------------------------------------------------

def pulse_times(p: np.ndarray) -> np.ndarray:
    """펄스 위치(초)를 표본 이하 정밀도로. 펄스 모양과 무관하게 절댓값 포락선의 봉우리를 씀."""
    env = np.abs(signal.hilbert(p))
    env = signal.savgol_filter(env, 9, 2)
    h = 0.4 * np.percentile(env, 99.9)
    # 우선 대략적인 간격을 구한 뒤, 그 간격의 60%를 최소 거리로
    pk, _ = signal.find_peaks(env, height=h, distance=int(0.001 * SR))
    if len(pk) < 10:
        return np.array([])
    period = np.median(np.diff(pk))
    pk, _ = signal.find_peaks(env, height=h, distance=max(1, int(0.6 * period)))
    # 포물선 보간으로 봉우리 위치 미세 조정
    ok = (pk > 0) & (pk < len(env) - 1)
    pk = pk[ok]
    a, b, c = env[pk - 1], env[pk], env[pk + 1]
    denom = a - 2 * b + c
    off = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / denom, 0.0)
    return (pk + np.clip(off, -0.5, 0.5)) / SR


def wow_trajectory(inp: Path, tgt: Path, seconds: float, out_rate: int) -> tuple[np.ndarray, dict]:
    x = read(inp, seconds)
    y = read(tgt, seconds)
    if x.shape[1] < 2 or y.shape[1] < 2:
        return np.array([]), {"error": "펄스 채널(오른쪽)이 없음"}
    # 테이프 지연(녹음 헤드→재생 헤드)은 수백 ms 일 수 있어 소리 채널로 전체 지연을 먼저 찾음
    n = min(len(x), len(y), 30 * SR)
    c = signal.correlate(y[:n, 0], x[:n, 0], mode="full", method="fft")
    lag = (np.argmax(np.abs(c)) - (n - 1)) / SR

    ti, tt = pulse_times(x[:, 1]), pulse_times(y[:, 1])
    if len(ti) < 100 or len(tt) < 100:
        return np.array([]), {"error": f"펄스를 충분히 찾지 못함 (입력 {len(ti)}, 출력 {len(tt)})"}
    period = float(np.median(np.diff(ti)))
    # 각 입력 펄스에 대응하는 출력 펄스를 '현재 지연'을 따라가며 찾음.
    # 녹음·재생 속도가 미세하게 달라 지연이 계속 벌어지므로 처음 지연에 고정하면 안 됨.
    cur = lag
    t_list, d_list, miss = [], [], 0
    for tk in ti:
        pred = tk + cur
        j = int(np.searchsorted(tt, pred))
        best = None
        for jj in (j - 1, j):
            if 0 <= jj < len(tt) and (best is None or abs(tt[jj] - pred) < abs(tt[best] - pred)):
                best = jj
        if best is not None and abs(tt[best] - pred) < 0.3 * period:
            dk = tt[best] - tk
            t_list.append(tk); d_list.append(dk)
            cur = 0.8 * cur + 0.2 * dk            # 지연 변화를 천천히 따라감
        else:
            miss += 1
    rate_ok = len(t_list) / len(ti)
    if rate_ok < 0.9:
        return np.array([]), {"error": f"펄스 짝 맞추기 실패 비율 높음 ({1 - rate_ok:.0%})", "period_ms": period * 1000}
    t, d = np.array(t_list), np.array(d_list)
    # 튀는 값(잘못 잡힌 펄스) 제거
    med = signal.medfilt(d, 5)
    keep = np.abs(d - med) < 5 * np.median(np.abs(d - med)) + 1e-6
    t, d = t[keep], d[keep]
    grid = np.arange(t[0], t[-1], 1 / out_rate)
    traj = np.interp(grid, t, d)
    # 일정한 지연과 느린 속도 차이(10초보다 긴 변화)는 제거 → 흔들림만 남김
    from scipy.ndimage import uniform_filter1d
    traj = traj - uniform_filter1d(traj, size=10 * out_rate, mode="nearest")
    info = {"pulse_period_ms": period * 1000, "tape_latency_ms": lag * 1000,
            "matched": float(rate_ok), "std_ms": float(np.std(traj) * 1000),
            "peak_to_peak_ms": float(np.ptp(traj) * 1000), "seconds": float(len(traj) / out_rate)}
    # 속도 변동률(%) = 지연의 시간 미분
    speed = np.gradient(traj) * out_rate
    info["speed_dev_rms_pct"] = float(np.sqrt(np.mean(speed ** 2)) * 100)
    return traj.astype(np.float32), info


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Aalto 테이프 데이터셋 → tape_profile.npz")
    ap.add_argument("root", help="tar 를 푼 폴더 (예: C:\\aalto)")
    ap.add_argument("--out", default="assets/tape_profile.npz")
    ap.add_argument("--noise-clips", type=int, default=12, help="속도별 히스 조각 수")
    ap.add_argument("--noise-sec", type=float, default=8.0)
    ap.add_argument("--analyze-sec", type=float, default=600.0, help="파일당 분석할 길이(초)")
    ap.add_argument("--traj-rate", type=int, default=500, help="흔들림 궤적 저장 간격(Hz)")
    args = ap.parse_args()

    root = Path(args.root)
    rng = np.random.default_rng(0)
    silence = find_folders(root, "Silence_")
    r2r = find_folders(root, "ReelToReel_Dataset_MiniPulse100_")
    out = {"sr": np.int32(SR), "traj_rate": np.int32(args.traj_rate)}
    report = []

    for s in SPEEDS:
        key = s.replace(".", "_")
        print(f"\n=== {s} IPS ===")
        bank, npsd = noise_bank(silence[s], args.noise_clips, args.noise_sec, rng)
        if len(bank):
            out[f"noise_{key}"] = bank
            lvl = 10 * np.log10(np.sum(npsd) * (SR / 8192) + 1e-30)
            print(f"  히스: {len(bank)}조각 × {args.noise_sec:g}초, 원래 잡음 크기 약 {lvl:.1f} dBFS")
            report.append(f"{s} IPS 히스 크기 {lvl:.1f} dBFS")
        else:
            print("  [경고] 무음 폴더를 찾지 못해 히스를 건너뜀")

        pairs = [p for f in r2r[s] for p in pairs_in(f)]
        if not pairs:
            print("  [경고] ReelToReel 짝을 찾지 못해 주파수 특성·흔들림을 건너뜀")
            continue
        print(f"  짝 파일 {len(pairs)}개 (파일당 {args.analyze_sec:g}초 분석)")

        f, g, valid = freq_response(pairs, npsd, args.analyze_sec)
        out["eq_freqs"] = f.astype(np.float32)
        out[f"eq_db_{key}"] = g.astype(np.float32)
        pts = [50, 100, 1000, 5000, 10000, 15000, 18000]
        print("  주파수 특성 (1kHz 근처 = 0dB): " +
              ", ".join(f"{p / 1000:g}k {np.interp(p, f, g):+.1f}" for p in pts))
        report.append(f"{s} IPS 특성 " + ", ".join(f"{p / 1000:g}k {np.interp(p, f, g):+.1f}dB" for p in pts))

        trajs = []
        for inp, tgt in pairs:
            tr, info = wow_trajectory(inp, tgt, args.analyze_sec, args.traj_rate)
            if len(tr):
                trajs.append(tr)
                print(f"  흔들림 {inp.parent.name}/{inp.name}: 지연 흔들림 표준편차 {info['std_ms']:.3f}ms, "
                      f"최대폭 {info['peak_to_peak_ms']:.2f}ms, 속도 변동 {info['speed_dev_rms_pct']:.3f}%, "
                      f"펄스 간격 {info['pulse_period_ms']:.2f}ms, 테이프 지연 {info['tape_latency_ms']:.1f}ms")
            else:
                print(f"  [건너뜀] {inp.name}: {info.get('error')}")
        if trajs:
            out[f"traj_{key}"] = np.concatenate(trajs)
            sp = np.gradient(out[f"traj_{key}"]) * args.traj_rate
            report.append(f"{s} IPS 속도 변동 RMS {np.sqrt(np.mean(sp ** 2)) * 100:.3f}%")

    if not any(k.startswith(("noise_", "eq_db_", "traj_")) for k in out):
        raise SystemExit("[오류] 측정된 것이 없습니다. 폴더 경로를 확인하세요.")
    out["credits"] = np.array("Derived from the Magnetic Tape Recorder Dataset (Mikkonen, Wright, Moliner, "
                              "Välimäki, DAFx 2023), CC BY 4.0, https://doi.org/10.5281/zenodo.8026271")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    size = Path(args.out).stat().st_size / 1e6
    print(f"\n저장: {args.out} ({size:.1f} MB)")
    print("요약:\n  " + "\n  ".join(report))


if __name__ == "__main__":
    main()
