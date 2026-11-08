import sys, json, subprocess, csv
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from degrade import degrade, PRESETS

T = Path(__file__).parent / "test_output"; T.mkdir(exist_ok=True)
sr = 44100
rng = np.random.default_rng(1)

# --- 합성 '피아노 + 홀' 스테레오 신호 (진짜 녹음 대신 테스트용) ---
dur = 12.0
n = int(sr * dur)
t = np.arange(n) / sr
L = np.zeros(n); R = np.zeros(n)
midi = [48, 55, 60, 64, 67, 72, 76, 79, 84, 88, 60, 64, 67, 71, 74]
for i, m in enumerate(midi):
    f0 = 440 * 2 ** ((m - 69) / 12)
    st = int(i * 0.7 * sr)
    tt = np.arange(n - st) / sr
    note = np.zeros(n - st)
    for k in range(1, 25):
        fk = k * f0 * np.sqrt(1 + 0.0004 * k * k)  # 피아노 비조화성
        if fk > sr / 2 * 0.95: break
        note += (1 / k) * np.sin(2 * np.pi * fk * tt) * np.exp(-tt * (1.5 + 0.4 * k))
    pan = (m - 48) / 40
    L[st:] += note * (1 - pan); R[st:] += note * pan
# 홀 잔향 (좌우 비상관, RT60 2초)
for ch in (L, R):
    ir = rng.standard_normal(int(sr * 2.2)) * np.exp(-6.9 * np.arange(int(sr * 2.2)) / sr / 2.0)
    ch += 0.15 * signal.fftconvolve(ch, ir)[:n] / np.sqrt(np.sum(ir**2))
x = np.stack([L, R], 1)
x = (x / np.max(np.abs(x)) * 0.8).astype(np.float32)
sf.write(T / "clean.wav", x, sr)

def band_energy_ratio(y, sr, f):
    F, P = signal.welch(y, sr, nperseg=4096)
    return 10 * np.log10(P[F > f].sum() / P.sum() + 1e-20)

results = {}
for name in PRESETS:
    y, meta = degrade(x, sr, name, seed=7)
    sf.write(T / f"degraded_{name}.wav", y, sr)
    assert y.ndim == 1, "모노가 아님"
    assert len(y) == len(x), "길이 불일치"
    assert np.isfinite(y).all(), "NaN/Inf"
    # degrade 는 최종 출력을 0.99 이하로 맞춤 (손실 압축 후 최대값이 살짝 튀기 때문). 1.0 을 넘어야 진짜 잘림
    assert np.max(np.abs(y)) <= 0.991, "클리핑"
    # 정렬 확인: 원본 모노와 교차상관 지연
    ref = x.mean(1)
    c = signal.correlate(y[:sr*4], ref[:sr*4], mode="full", method="fft")
    lag_ms = (np.argmax(c) - (sr*4 - 1)) / sr * 1000
    results[name] = {
        "lowpass_hz": round(meta["lowpass_hz"]),
        "hf_energy_above_lowpass_clean_dB": round(band_energy_ratio(ref, sr, meta["lowpass_hz"]), 1),
        "hf_energy_above_lowpass_degraded_dB": round(band_energy_ratio(y, sr, meta["lowpass_hz"]), 1),
        "alignment_lag_ms": round(lag_ms, 2),
        "max_time_shift_ms": round(meta["max_time_shift_ms"], 2),
    }
# 재현성
a, _ = degrade(x, sr, "tape_1950s", seed=3); b, _ = degrade(x, sr, "tape_1950s", seed=3)
assert np.array_equal(a, b), "같은 seed인데 결과가 다름"
print(json.dumps(results, indent=2, ensure_ascii=False, default=float))

# --- make_pairs 통합 테스트 (허용/비허용 라이선스) ---
src = T / "clean_dir"; src.mkdir(exist_ok=True)
sf.write(src / "ok.wav", x, sr)
sf.write(src / "nc.wav", x, 48000)  # 다른 샘플레이트 + 비상업 라이선스
sf.write(src / "unlisted.wav", x[:, 0], sr)
with open(T / "sources.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "title", "performer", "source_url", "license"])
    w.writerow(["ok.wav", "test", "synth", "", "CC0"])
    w.writerow(["nc.wav", "test", "synth", "", "CC-BY-NC-SA"])
r = subprocess.run([sys.executable, str(Path(__file__).parent / "make_pairs.py"), "--src", str(src),
                    "--sources", str(T / "sources.csv"), "--out", str(T / "pairs"),
                    "--segment", "4", "--variants", "3"], capture_output=True, text=True)
print(r.stdout, r.stderr)
lines = [json.loads(l) for l in open(T / "pairs" / "meta.jsonl")]
tgt, _ = sf.read(T / "pairs" / lines[0]["target"]); inp, _ = sf.read(T / "pairs" / lines[0]["input"])
print("pairs:", len(lines), "target shape", tgt.shape, "input shape", inp.shape,
      "presets", sorted({l["degrade"]["preset"] for l in lines}))
assert all(l["license"] == "CC0" for l in lines)

# --- 스펙트로그램 비교 그림 ---
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
sigs = [("Target: modern stereo (synthetic test)", x.mean(1)),
        ("Input: tape_1950s", sf.read(T / "degraded_tape_1950s.wav")[0]),
        ("Input: shellac_1925", sf.read(T / "degraded_shellac_1925.wav")[0])]
plt.rcParams["font.family"] = "DejaVu Sans"
for ax, (title, s) in zip(axes, sigs):
    f, tt, S = signal.spectrogram(s[:sr*8], sr, nperseg=2048, noverlap=1536)
    ax.pcolormesh(tt, f / 1000, 10 * np.log10(S + 1e-12), vmin=-120, vmax=-30, shading="auto", cmap="magma")
    ax.set_title(title, fontsize=10); ax.set_xlabel("time (s)")
axes[0].set_ylabel("frequency (kHz)")
plt.tight_layout(); plt.savefig(T / "spectrogram_compare.png", dpi=110)
print("figure saved")
