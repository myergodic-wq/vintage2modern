"""
measure_real.py — 실제 옛 녹음(정답 없음)의 변환 전/후를 숫자와 그림으로 비교

어떤 종류의 잡음이 남았는지 알아내기 위한 도구:
    히스(쉬~)          조용한 부분의 4kHz 이상 잡음 크기
    럼블·험(웅~)       80Hz 이하 에너지, 50/60Hz 봉우리
    크래클(지직)        초당 순간적인 튐 횟수
    주기적 스윽 소리    78rpm 음반 한 바퀴(약 1.3Hz) 주기로 잡음 크기가 오르내리는 정도

사용 예:
    python measure_real.py real_tests/rach_1920_prelude.mp3 real_tests/rach_1920_prelude_v3.wav
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import signal

from infer import load_audio
from analyze import width_db

SR = 44100


def load(path):
    """원래 채널 구성을 유지해서 읽음 (스테레오 폭을 재기 위해)."""
    import soundfile as sf
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != SR:
            from math import gcd
            g = gcd(sr, SR)
            x = signal.resample_poly(x, SR // g, sr // g, axis=0)
        return x
    except Exception:
        m, _ = load_audio(path)          # mp3/webm 등: 모노로
        return m[:, None]


def click_rate(x: np.ndarray, order: int = 24, frame: float = 0.05, k: float = 8.0) -> float:
    # 검증 (2ms 타건 피아노 합성 신호): 깨끗한 원본 1.7, 초당 10/30/60개 넣으면 10.3/25.7/44.0
    # → 깨끗한 음악은 거의 0, 밀도가 아주 높으면 약간 적게 셈
    """
    초당 클릭 수. 선형 예측(LPC) 잔차 방식 — 전문 디클리커가 쓰는 고전적 방법.
    프레임마다 '앞 샘플들로 다음 샘플을 예측'하는 필터를 맞추면, 음악(배음)은 예측되어 사라지고
    예측할 수 없는 짧은 튐(클릭)만 잔차에 크게 남음. 잔차가 그 프레임 잔차의 k배를 넘으면 클릭.
    """
    from scipy.linalg import solve_toeplitz
    L = int(frame * SR)
    count = 0
    for s in range(0, len(x) - L, L):
        seg = x[s:s + L].astype(np.float64)
        if np.max(np.abs(seg)) < 1e-6:
            continue
        w = seg * np.hanning(L)
        ac = np.correlate(w, w, "full")[L - 1:L + order]
        ac[0] *= 1.0001                                    # 수치 안정
        try:
            a = solve_toeplitz(ac[:order], ac[1:order + 1])
        except Exception:
            continue
        e = seg - signal.lfilter(np.r_[0, a], [1.0], seg)  # 예측 오차
        e = e[order:]
        sd = 1.4826 * np.median(np.abs(e)) + 1e-12         # 튐에 영향 덜 받는 크기 추정
        pk, _ = signal.find_peaks(np.abs(e), height=k * sd, distance=int(0.003 * SR))
        count += len(pk)
    return count / (len(x) / SR)


def measure(x2: np.ndarray) -> dict:
    x = x2.mean(1)
    x = x / (np.max(np.abs(x)) + 1e-12)
    # 프레임별 에너지로 '조용한 부분'(하위 10%)과 '큰 부분'(상위 10%) 구분
    f, t, S = signal.stft(x, SR, nperseg=4096, noverlap=3072)
    E = np.abs(S) ** 2
    fe = E.sum(0)
    q = fe <= np.percentile(fe, 10)
    loud = fe >= np.percentile(fe, 90)
    quiet_spec = E[:, q].mean(1)
    loud_spec = E[:, loud].mean(1)
    db = lambda a, b, spec: 10 * np.log10(spec[(f >= a) & (f < b)].sum() + 1e-20)
    total_loud = 10 * np.log10(loud_spec.sum() + 1e-20)

    r = {}
    r["음악/잡음 비 (dB)"] = 10 * np.log10(fe[loud].mean() / (fe[q].mean() + 1e-20))
    r["히스: 조용한 부분 4k↑ (dB)"] = db(4000, SR / 2, quiet_spec) - total_loud
    r["럼블: 조용한 부분 80Hz↓ (dB)"] = db(20, 80, quiet_spec) - total_loud
    # 험: 50/60Hz 주변 봉우리가 이웃보다 얼마나 튀는지
    fw, P = signal.welch(x, SR, nperseg=1 << 16)
    def peak(fr):
        on = P[(fw > fr - 1.5) & (fw < fr + 1.5)].max()
        off = np.median(P[(fw > fr - 15) & (fw < fr + 15)])
        return 10 * np.log10(on / (off + 1e-30))
    r["험: 50/60Hz 봉우리 (dB)"] = max(peak(50), peak(60))
    r["크래클 (초당)"] = click_rate(x)
    # 주기적 스윽: 2~8kHz 잡음 포락선이 1.0~1.6Hz(78rpm 근처)로 흔들리는 정도
    band = signal.sosfilt(signal.butter(4, [2000 / (SR / 2), 8000 / (SR / 2)], "band", output="sos"), x)
    envb = signal.decimate(np.abs(band), 441)                     # 100Hz
    envb = envb - envb.mean()
    fm, Pm = signal.welch(envb, 100, nperseg=min(len(envb), 2048))
    in_band = Pm[(fm >= 1.0) & (fm <= 1.6)].max()
    ref = np.median(Pm[(fm >= 0.5) & (fm <= 3.0)])
    r["주기적 스윽: 1.3Hz 근처 (dB)"] = 10 * np.log10(in_band / (ref + 1e-30))
    # 유효 대역: 큰 부분 스펙트럼이 조용한 부분보다 6dB 이상 높은 가장 높은 주파수
    # 500Hz 부터 위로 올라가며, 1/6 옥타브 창 안에서 '큰 부분이 조용한 부분보다 6dB 이상' 인
    # 비율이 절반 아래로 떨어지는 첫 주파수 (띄엄띄엄 튀는 값에 흔들리지 않도록)
    ok = 10 * np.log10(loud_spec + 1e-20) - 10 * np.log10(quiet_spec + 1e-20) > 6
    top = 0.0
    for i in np.where(f >= 500)[0]:
        win = (f >= f[i] / 2 ** (1 / 12)) & (f <= f[i] * 2 ** (1 / 12))
        if ok[win].mean() < 0.5:
            top = float(f[i])
            break
    else:
        top = float(f[-1])
    r["음악이 잡음보다 큰 최고 주파수 (Hz)"] = top
    r["스테레오 폭 (dB)"] = width_db(x2) if x2.shape[1] == 2 else -200.0
    return r, (f, quiet_spec, loud_spec)


GUIDE = {
    "음악/잡음 비 (dB)": "클수록 좋음",
    "히스: 조용한 부분 4k↑ (dB)": "낮을수록 좋음",
    "럼블: 조용한 부분 80Hz↓ (dB)": "낮을수록 좋음",
    "험: 50/60Hz 봉우리 (dB)": "0에 가까울수록 좋음 (6 이상이면 험 있음)",
    "크래클 (초당)": "낮을수록 좋음",
    "주기적 스윽: 1.3Hz 근처 (dB)": "참고용 (음악의 리듬도 섞여 나옴)",
    "음악이 잡음보다 큰 최고 주파수 (Hz)": "참고용 (조용한 구간이 적은 곡에선 흔들림)",
    "스테레오 폭 (dB)": "모노는 -200",
}


def main():
    ap = argparse.ArgumentParser(description="실제 녹음 변환 전/후 비교 (정답 없이)")
    ap.add_argument("files", nargs="+", help="입력, 출력(들) 순서로")
    args = ap.parse_args()

    res, specs = [], []
    for p in args.files:
        r, s = measure(load(p))
        res.append(r); specs.append(s)
    names = [Path(p).name for p in args.files]
    w = max(len(k) for k in GUIDE) + 2
    print(f"{'':{w}}" + "".join(f"{n[:22]:>24}" for n in names) + "   보는 법")
    for k in GUIDE:
        print(f"{k:{w}}" + "".join(f"{r[k]:>24.1f}" for r in res) + f"   {GUIDE[k]}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, len(args.files), figsize=(6 * len(args.files), 4), squeeze=False)
    for ax, n, (f, qs, ls) in zip(axes[0], names, specs):
        ax.semilogx(f[1:], 10 * np.log10(ls[1:] + 1e-20), label="loud parts", color="#2a6fd4")
        ax.semilogx(f[1:], 10 * np.log10(qs[1:] + 1e-20), label="quiet parts (noise)", color="#d4572a")
        ax.set_xlim(30, SR / 2); ax.grid(alpha=0.3); ax.set_title(n[:40], fontsize=9); ax.set_xlabel("Hz")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = Path(args.files[-1]).with_suffix(".measure.png")
    plt.savefig(out, dpi=100)
    print(f"\n그림 저장: {out}")


if __name__ == "__main__":
    main()
