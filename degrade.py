"""
degrade.py — 현대 스테레오 녹음을 '옛날 모노 녹음'처럼 만드는 열화 시뮬레이터

학습용 쌍 데이터를 만들기 위한 도구입니다.
    현대 스테레오 (정답)  --degrade-->  옛날풍 모노 (입력)

처리 순서 (실제 녹음 체인 순서를 따름):
    1. 스테레오 → 모노 합성        (당시 모노 녹음)
    2. 작은 방 잔향                (당시 스튜디오 공간, 선택)
    3. 마이크 음색                  (리본/초기 콘덴서 마이크 특성)
    4. 대역 제한                    (잘린 고음/저음)
    5. 다이내믹 압축                (당시 레벨 관리)
    6. 테이프 포화                  (아날로그 테이프 왜곡)
    6b. 테이프 주파수 특성          (실측, assets/tape_profile.npz 가 있을 때)
    7. Wow & flutter                (테이프 속도 흔들림. 실측 궤적이 있으면 그것을 사용)
    8. 테이프 잡음                  (히스(실측 또는 합성), 험)

실측 테이프 특성은 calibrate_tape.py 로 Aalto 테이프 데이터셋에서 만들며,
1925년(SP) 프리셋에는 쓰지 않습니다(테이프가 없던 시대).
    9. 재생·전사 대역 제한          (히스까지 같이 깎임)
   10. 디스크 크래클                (클릭, 재생할 때 생기는 광대역 잡음)
   10b. 78rpm 표면 잡음             (홈 잡음의 회전 주기 출렁임, 긴 팝, 매 바퀴 반복 스크래치.
                                    실제 홈 잡음 assets/disc_noise.npz 가 있으면 사용)
   --- 여기부터는 '유튜브에 올라온 옛 녹음'의 추가 열화 ---
   11. 업로더의 잡음 제거          (과한 스펙트럼 잡음 제거 → 뭉개짐, '물 흐르는' 잔여음)
   12. 손실 압축                   (유튜브 Opus/AAC, 옛 MP3 재업로드 → 두 번 압축)

모든 단계는 확률적으로 파라미터를 뽑고, 사용된 값을 dict로 돌려줍니다.
(메타데이터로 저장해 두면 분석·조건부 학습에 쓸 수 있음)

의존성: numpy, scipy, soundfile, ffmpeg(12번 손실 압축 단계)
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# 프리셋: 시대별 열화 강도 범위 (최소, 최대)
# ---------------------------------------------------------------------------

@dataclass
class Preset:
    name: str
    lowpass_hz: tuple = (7000, 12000)      # 고음 차단 주파수
    lowpass_order: tuple = (4, 8)
    highpass_hz: tuple = (40, 100)         # 저음 차단 주파수
    room_prob: float = 0.5                 # 작은 방 잔향을 넣을 확률
    room_rt60: tuple = (0.2, 0.6)          # 잔향 시간(초)
    room_wet: tuple = (0.05, 0.25)
    mic_peaks: int = 3                     # 마이크 음색용 무작위 EQ 개수
    mic_gain_db: tuple = (-4.0, 4.0)
    comp_ratio: tuple = (1.5, 4.0)
    comp_threshold_db: tuple = (-24.0, -12.0)
    tape_drive: tuple = (1.0, 3.0)         # 클수록 포화 심함
    wow_depth: tuple = (0.0005, 0.003)     # 피치 변동 비율 (0.001 = 0.1%)
    wow_rate_hz: tuple = (0.3, 1.5)
    flutter_depth: tuple = (0.0001, 0.0008)
    flutter_rate_hz: tuple = (6.0, 14.0)
    hiss_snr_db: tuple = (25.0, 45.0)      # 낮을수록 잡음 큼
    hum_prob: float = 0.3
    hum_snr_db: tuple = (35.0, 55.0)
    crackle_prob: float = 0.2              # LP/디스크 전사 잡음
    crackle_rate: tuple = (0.5, 5.0)       # 초당 클릭 수
    denoise_prob: float = 0.3              # 업로더가 잡음 제거를 했을 확률
    denoise_strength: tuple = (0.8, 2.5)
    codec_prob: float = 0.85               # 손실 압축을 거쳤을 확률 (유튜브면 사실상 항상)
    codec_double_prob: float = 0.25        # MP3 등을 다시 유튜브에 올린 '두 번 압축'
    tape_real_prob: float = 0.75           # 실측 테이프 특성(assets/tape_profile.npz)을 쓸 확률
    real_wow_scale: tuple = (0.7, 2.5)     # 실측 흔들림 배율 (가정용 데크 기준 → 더 낡은 장비까지)
    real_eq_scale: tuple = (0.5, 1.5)      # 실측 주파수 특성의 강도 배율
    # 78rpm 음반 표면 잡음 (홈 잡음 + 한 바퀴 주기의 스윽 소리 + 긴 팝 + 매 바퀴 반복되는 스크래치)
    surface_prob: float = 0.0
    surface_snr_db: tuple = (12.0, 28.0)   # 낮을수록 표면 잡음 큼
    swish_depth: tuple = (0.3, 0.9)        # 한 바퀴 주기로 잡음 크기가 출렁이는 정도
    pop_rate: tuple = (0.0, 5.0)           # 초당 긴 '팝' 수 (1~6ms)
    scratch_n: tuple = (0, 3)              # 매 바퀴 같은 자리에서 반복되는 스크래치 개수
    disc_real_prob: float = 0.7            # 실제 홈 잡음(assets/disc_noise.npz)을 쓸 확률


PRESETS = {
    # 1950년대 테이프 → LP 녹음 (목표: 호로비츠 1955년 같은 음반). LP 전사 잡음은 약하게 가끔
    "tape_1950s": Preset(name="tape_1950s", surface_prob=0.15, surface_snr_db=(30.0, 45.0),
                         swish_depth=(0.1, 0.4), pop_rate=(0.0, 1.0), scratch_n=(0, 1)),
    # 1925~1950년 전기 녹음 → SP(78rpm) 음반
    "shellac_1925": Preset(
        name="shellac_1925",
        lowpass_hz=(4000, 7000),
        highpass_hz=(80, 200),
        room_prob=0.7,
        mic_gain_db=(-6.0, 6.0),
        tape_drive=(1.5, 4.0),
        wow_depth=(0.001, 0.005),
        hiss_snr_db=(15.0, 30.0),
        hum_prob=0.2,
        crackle_prob=0.95,
        crackle_rate=(10.0, 60.0),
        tape_real_prob=0.0,                # 테이프가 아니라 원반에 직접 기록
        surface_prob=0.95,
        surface_snr_db=(12.0, 28.0),
        pop_rate=(1.0, 8.0),
        scratch_n=(0, 3),
    ),
    # 1925년 이전 '어쿠스틱' 녹음 (마이크 없이 나팔로 녹음) → 78rpm.
    # 실제 1920년 Victor 라흐마니노프 측정(검증된 LPC 클릭 검출기)과 비교해 조정:
    # 클릭 초당 약 13회 (시뮬 3~15회, 중앙 10회), 조용한 부분 4kHz 이상 히스 -14dB
    # (시뮬 범위의 시끄러운 끝 → 표면 잡음을 더 크게)
    "acoustic_pre1925": Preset(
        name="acoustic_pre1925",
        lowpass_hz=(2200, 4500),
        lowpass_order=(6, 10),
        highpass_hz=(150, 350),
        room_prob=0.8,
        room_rt60=(0.15, 0.4),
        mic_peaks=5,                       # 나팔의 강한 공명
        mic_gain_db=(-10.0, 10.0),
        comp_ratio=(1.0, 1.5),
        tape_drive=(1.0, 2.5),
        wow_depth=(0.002, 0.006),
        hiss_snr_db=(15.0, 30.0),
        hum_prob=0.05,
        crackle_prob=1.0,
        crackle_rate=(25.0, 90.0),
        tape_real_prob=0.0,
        surface_prob=1.0,
        surface_snr_db=(3.0, 15.0),
        swish_depth=(0.4, 0.95),
        pop_rate=(2.0, 12.0),
        scratch_n=(0, 4),
    ),
    # 가벼운 열화 (모델이 '거의 깨끗한' 입력도 망치지 않게 학습용으로 섞기)
    "mild": Preset(
        name="mild",
        lowpass_hz=(12000, 16000),
        highpass_hz=(20, 50),
        room_prob=0.2,
        mic_gain_db=(-2.0, 2.0),
        comp_ratio=(1.0, 2.0),
        tape_drive=(0.5, 1.5),
        wow_depth=(0.0, 0.001),
        flutter_depth=(0.0, 0.0003),
        hiss_snr_db=(40.0, 60.0),
        hum_prob=0.1,
        crackle_prob=0.05,
    ),
}


def _u(rng: np.random.Generator, lo_hi: tuple) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


# ---------------------------------------------------------------------------
# 개별 처리 단계
# ---------------------------------------------------------------------------

def to_mono(x: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    """(N, 2) 또는 (N,) → (N,). 좌우 균형을 약간 무작위로."""
    if x.ndim == 1:
        return x.copy(), {"mono_balance": 0.5}
    bal = _u(rng, (0.4, 0.6))
    return bal * x[:, 0] + (1 - bal) * x[:, 1], {"mono_balance": bal}


def small_room(x, sr, rng, p: Preset):
    """지수 감쇠 잡음으로 만든 작은 방 임펄스 응답을 섞음."""
    if rng.random() > p.room_prob:
        return x, {"room": False}
    rt60 = _u(rng, p.room_rt60)
    wet = _u(rng, p.room_wet)
    n = int(sr * rt60 * 1.2)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)  # 60dB 감쇠
    # 작은 방은 고음이 빨리 죽으니 IR을 약간 어둡게
    b, a = signal.butter(2, 4000 / (sr / 2), btype="low")
    ir = signal.lfilter(b, a, ir)
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    wet_sig = signal.fftconvolve(x, ir)[: len(x)]
    y = (1 - wet) * x + wet * wet_sig
    return y, {"room": True, "room_rt60": rt60, "room_wet": wet}


def _peaking_eq(f0, gain_db, q, sr):
    """RBJ 피킹 EQ 계수."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
    return b / a[0], a / a[0]


def mic_color(x, sr, rng, p: Preset):
    """무작위 피킹 EQ 몇 개로 옛 마이크의 울퉁불퉁한 주파수 응답을 흉내."""
    bands = []
    y = x
    for _ in range(p.mic_peaks):
        f0 = float(np.exp(rng.uniform(np.log(150), np.log(6000))))
        g = _u(rng, p.mic_gain_db)
        q = _u(rng, (0.5, 2.0))
        b, a = _peaking_eq(f0, g, q, sr)
        y = signal.lfilter(b, a, y)
        bands.append({"f0": f0, "gain_db": g, "q": q})
    return y, {"mic_eq": bands}


def bandlimit(x, sr, rng, p: Preset):
    lp = min(_u(rng, p.lowpass_hz), sr / 2 * 0.95)
    hp = _u(rng, p.highpass_hz)
    order = int(rng.integers(p.lowpass_order[0], p.lowpass_order[1] + 1))
    sos_lp = signal.butter(order, lp / (sr / 2), btype="low", output="sos")
    sos_hp = signal.butter(2, hp / (sr / 2), btype="high", output="sos")
    y = signal.sosfilt(sos_hp, signal.sosfilt(sos_lp, x))
    return y, {"lowpass_hz": lp, "lowpass_order": order, "highpass_hz": hp}


def compress(x, sr, rng, p: Preset):
    """간단한 RMS 기반 컴프레서."""
    ratio = _u(rng, p.comp_ratio)
    thr_db = _u(rng, p.comp_threshold_db)
    win = int(0.02 * sr)
    env = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, mode="same") + 1e-12)
    env_db = 20 * np.log10(env + 1e-12)
    over = np.maximum(env_db - thr_db, 0.0)
    gain_db = -over * (1 - 1 / ratio)
    # 게인 변화를 부드럽게 (어택/릴리즈 대신 저역통과)
    b, a = signal.butter(1, 10 / (sr / 2))
    gain_db = signal.filtfilt(b, a, gain_db)
    y = x * 10 ** (gain_db / 20)
    return y, {"comp_ratio": ratio, "comp_threshold_db": thr_db}


def tape_saturate(x, rng, p: Preset):
    """tanh 소프트 클리핑 + 약간의 비대칭 (짝수 배음)."""
    drive = _u(rng, p.tape_drive)
    bias = _u(rng, (0.0, 0.1))
    peak = np.max(np.abs(x)) + 1e-9
    xn = x / peak
    y = np.tanh(drive * (xn + bias)) - np.tanh(drive * bias)
    y = y / (np.max(np.abs(y)) + 1e-9) * peak
    return y, {"tape_drive": drive, "tape_bias": bias}


TAPE_PROFILE = Path(__file__).resolve().parent / "assets" / "tape_profile.npz"


def measured_enabled() -> bool:
    """
    실측 자료(테이프 특성·실제 홈 잡음)를 쓸지. 환경 변수 V2M_NO_MEASURED=1 이면 끔 (보정 효과 분리 실험용).
    끄면 실측 대신 같은 자리에 합성 방식이 들어가고, 나머지 열화(프리셋·78rpm 스윽·팝·스크래치 등)는 그대로.
    make_pairs.py --no-measured 가 이 변수를 설정 (병렬 작업자에게도 전달됨).
    """
    return os.environ.get("V2M_NO_MEASURED", "") != "1"


def real_disc_enabled() -> bool:
    """
    실제 78rpm 홈 잡음(assets/disc_noise.npz)을 쓸지. 환경 변수 V2M_NO_REAL_DISC=1 이면 끔 (2026-11-07 r38).
    이유: 홈 잡음 조각의 출처인 UCSB DAHR 이용 조건이 연구·교육·정보 제공 목적만 허용 → 공개 가중치 학습에서 제외.
    끄면 추첨은 똑같이 하고(난수 흐름 맞춤) 뽑혀도 합성 홈 잡음을 씀. 실측 테이프 특성(Aalto, CC BY 4.0)은 그대로.
    make_pairs.py --no-real-disc 가 이 변수를 설정.
    """
    return os.environ.get("V2M_NO_REAL_DISC", "") != "1"


@functools.lru_cache(maxsize=1)
def _tape_profile_file() -> dict | None:
    if not TAPE_PROFILE.exists():
        return None
    with np.load(TAPE_PROFILE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_tape_profile() -> dict | None:
    """calibrate_tape.py 가 만든 실측 테이프 특성. 없거나 실측을 끈 경우 None (합성 방식만 사용)."""
    return _tape_profile_file() if measured_enabled() else None


def choose_real_tape(rng, p: Preset) -> str | None:
    """
    이번 샘플에 실측 테이프를 쓸지, 쓴다면 어느 속도(7_5 / 3_75)인지.
    실측을 끈 실험에서도 '쓸지' 추첨은 똑같이 해서 난수 흐름을 맞춤
    → 실측이 뽑히지 않은 샘플은 두 실험에서 완전히 같은 입력이 됨.
    """
    prof = _tape_profile_file()
    if prof is None or rng.random() >= p.tape_real_prob:
        return None
    speeds = [s for s in ("7_5", "3_75") if f"traj_{s}" in prof or f"noise_{s}" in prof or f"eq_db_{s}" in prof]
    speed = str(rng.choice(speeds)) if speeds else None
    return speed if measured_enabled() else None


def tape_eq(x, sr, rng, p: Preset, speed: str | None):
    """실측 테이프 주파수 특성 (저음 '헤드 범프', 고음 감쇠 등). 지연 없는 선형 위상 FIR."""
    prof = load_tape_profile()
    if speed is None or prof is None or f"eq_db_{speed}" not in prof:
        return x, {"tape_eq": False}
    f, g = prof["eq_freqs"].astype(np.float64), prof[f"eq_db_{speed}"].astype(np.float64)
    k = _u(rng, p.real_eq_scale)
    g = np.clip(g * k, -40, 12)
    fn = np.clip(f / (sr / 2), 0, 1)
    fn, idx = np.unique(fn, return_index=True)
    gain = 10 ** (g[idx] / 20)
    if fn[0] > 0:
        fn, gain = np.r_[0.0, fn], np.r_[gain[0], gain]
    if fn[-1] < 1:
        fn, gain = np.r_[fn, 1.0], np.r_[gain, gain[-1]]
    h = signal.firwin2(1025, fn, gain)
    y = signal.fftconvolve(x, h, mode="same")          # 대칭 FIR + same → 지연 0
    return y, {"tape_eq": True, "tape_speed": speed, "tape_eq_scale": k}


def wow_flutter(x, sr, rng, p: Preset, speed: str | None = None):
    """
    테이프 속도 흔들림을 시간축 재샘플링으로 구현.
    실측 궤적(Aalto)이 있으면 그 일부를 잘라 배율을 곱해 쓰고, 없으면 사인파 합성.
    변조의 평균이 0이라 전체 길이·정렬은 대략 유지됨
    (정답과 입력의 시간 정렬이 크게 어긋나지 않도록).
    """
    n = len(x)
    prof = load_tape_profile()
    if speed is not None and prof is not None and f"traj_{speed}" in prof:
        traj, rate = prof[f"traj_{speed}"].astype(np.float64), int(prof["traj_rate"])
        need = int(np.ceil(n / sr * rate)) + 2
        while len(traj) < need:                         # 궤적이 짧으면 이어 붙임 (드묾)
            traj = np.concatenate([traj, traj[::-1]])
        st = int(rng.integers(0, len(traj) - need + 1))
        seg = traj[st:st + need]
        k = _u(rng, p.real_wow_scale)
        seg = (seg - seg.mean()) * k                    # 평균 0 → 정렬 유지
        d = np.interp(np.arange(n) / sr * rate, np.arange(need), seg)   # 초 단위 지연
        pos = np.clip(np.arange(n) - d * sr, 0, n - 1)
        y = np.interp(pos, np.arange(n), x)
        return y, {"wow_real": True, "wow_scale": k,
                   "max_time_shift_ms": float(np.max(np.abs(d)) * 1000)}

    t = np.arange(n) / sr
    wd, wr = _u(rng, p.wow_depth), _u(rng, p.wow_rate_hz)
    fd, fr = _u(rng, p.flutter_depth), _u(rng, p.flutter_rate_hz)
    ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
    # 순간 속도 비율 = 1 + 변조. 적분해서 읽기 위치를 얻음
    speed = 1 + wd * np.sin(2 * np.pi * wr * t + ph1) + fd * np.sin(2 * np.pi * fr * t + ph2)
    pos = np.cumsum(speed) - speed[0]
    pos = np.clip(pos, 0, n - 1)
    y = np.interp(pos, np.arange(n), x)
    max_shift_ms = float(np.max(np.abs(pos - np.arange(n))) / sr * 1000)
    return y, {"wow_real": False, "wow_depth": wd, "wow_rate_hz": wr, "flutter_depth": fd,
               "flutter_rate_hz": fr, "max_time_shift_ms": max_shift_ms}


def _pink_noise(n, rng):
    """1/f 잡음 (Voss 대신 주파수 영역 필터링)."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(len(spec))
    f[0] = 1
    spec /= np.sqrt(f)
    pink = np.fft.irfft(spec, n)
    return pink / (np.std(pink) + 1e-12)


def _real_hiss(n, rng, bank: np.ndarray) -> np.ndarray:
    """실측 히스 조각들을 무작위로 골라 겹쳐 이어 붙임 (이음새 50ms 교차 페이드)."""
    clip = bank.shape[1]
    fade = min(2205, clip // 4)
    ramp = np.linspace(0, 1, fade)
    out = np.zeros(n + clip)
    pos = 0
    while pos < n:
        c = bank[int(rng.integers(len(bank)))].astype(np.float64) / 32767
        st = int(rng.integers(0, max(1, clip // 2)))
        c = c[st:]
        if pos > 0:
            c[:fade] *= ramp
            out[pos - fade:pos] *= ramp[::-1]
            out[pos - fade:pos - fade + len(c)] += c
            pos = pos - fade + len(c)
        else:
            out[:len(c)] = c
            pos = len(c)
    return out[:n]


def tape_noise(x, sr, rng, p: Preset, speed: str | None = None):
    """테이프 히스와 전원 험 (녹음/재생 체인 안에서 생기는 잡음)."""
    n = len(x)
    sig_rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    info = {}

    hiss_snr = _u(rng, p.hiss_snr_db)
    prof = load_tape_profile()
    if speed is not None and prof is not None and f"noise_{speed}" in prof:
        hiss = _real_hiss(n, rng, prof[f"noise_{speed}"])      # 실측 테이프 히스
        info["hiss_real"] = True
    else:
        # 합성 히스: 백색+핑크 혼합, 고음 쪽이 강함
        hiss = 0.6 * rng.standard_normal(n) + 0.4 * _pink_noise(n, rng)
        info["hiss_real"] = False
    hiss = hiss / (np.std(hiss) + 1e-12) * sig_rms * 10 ** (-hiss_snr / 20)
    y = x + hiss
    info["hiss_snr_db"] = hiss_snr

    # 전원 험 (50 또는 60Hz + 배음)
    if rng.random() < p.hum_prob:
        f0 = 50.0 if rng.random() < 0.5 else 60.0
        hum_snr = _u(rng, p.hum_snr_db)
        t = np.arange(n) / sr
        hum = sum((1 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) for k in (1, 2, 3, 5))
        hum = hum / (np.std(hum) + 1e-12) * sig_rms * 10 ** (-hum_snr / 20)
        y = y + hum
        info.update({"hum_hz": f0, "hum_snr_db": hum_snr})

    return y, info


def transfer_filter(x, sr, rng, lowpass_hz: float):
    """
    재생·전사 체인의 대역 제한. 히스까지 같이 깎여서
    실제 옛 녹음처럼 차단 주파수 위가 '잡음만 남은 빈 공간'이 됨.
    """
    lp = min(lowpass_hz * _u(rng, (1.0, 1.6)), sr / 2 * 0.95)
    order = int(rng.integers(2, 5))
    sos = signal.butter(order, lp / (sr / 2), btype="low", output="sos")
    return signal.sosfilt(sos, x), {"transfer_lowpass_hz": lp, "transfer_order": order}


def disc_crackle(x, sr, rng, p: Preset):
    """클릭/크래클 (디스크를 재생해 디지털화할 때 생김, 광대역)."""
    n = len(x)
    sig_rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    info = {"crackle": False}
    y = x
    if rng.random() < p.crackle_prob:
        rate = _u(rng, p.crackle_rate)
        n_clicks = rng.poisson(rate * n / sr)
        clicks = np.zeros(n)
        idx = rng.integers(0, n, n_clicks)
        amps = rng.exponential(1.0, n_clicks) * rng.choice([-1, 1], n_clicks)
        clicks[idx] = amps
        # 클릭을 짧은 임펄스 모양으로
        kern = np.exp(-np.arange(int(0.0005 * sr) + 2) / (0.0001 * sr + 1e-9))
        clicks = np.convolve(clicks, kern, mode="same")
        clicks = clicks / (np.max(np.abs(clicks)) + 1e-12) * sig_rms * _u(rng, (1.0, 4.0))
        y = y + clicks
        info.update({"crackle": True, "crackle_rate": rate, "n_clicks": int(n_clicks)})

    return y, info


DISC_NOISE = Path(__file__).resolve().parent / "assets" / "disc_noise.npz"
ROT_HZ = 78.26 / 60          # 78rpm 한 바퀴 = 약 0.77초


@functools.lru_cache(maxsize=1)
def _disc_noise_file() -> np.ndarray | None:
    if not DISC_NOISE.exists():
        return None
    with np.load(DISC_NOISE, allow_pickle=False) as z:
        return z["noise"] if "noise" in z.files and len(z["noise"]) else None


def load_disc_noise() -> np.ndarray | None:
    """harvest_disc_noise.py 가 만든 실제 78rpm 홈 잡음 조각 (int16, 조각×길이). 없거나 실측을 끈 경우 None."""
    return _disc_noise_file() if measured_enabled() and real_disc_enabled() else None


def _add_at(buf: np.ndarray, pos: int, shape: np.ndarray):
    end = min(len(buf), pos + len(shape))
    if 0 <= pos < end:
        buf[pos:end] += shape[: end - pos]


def disc_surface(x, sr, rng, p: Preset):
    """
    78rpm 음반 표면 잡음. 실제 1920년 녹음 분석에서 기존 합성(짧은 클릭)에 없던 것들:
      - 홈 잡음이 한 바퀴(약 0.77초) 주기로 출렁이는 '스윽' 소리
      - 1~6ms 길이의 낮고 둔한 '팝'
      - 같은 자리 흠집이 매 바퀴 반복되는 스크래치
    """
    if rng.random() >= p.surface_prob:
        return x, {"surface": False}
    n = len(x)
    sig_rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    t = np.arange(n) / sr
    f_rot = ROT_HZ * _u(rng, (0.97, 1.03))               # 실제 회전 속도 편차
    ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
    depth = _u(rng, p.swish_depth)
    mod = 1 + depth * (0.7 * np.sin(2 * np.pi * f_rot * t + ph1) + 0.3 * np.sin(4 * np.pi * f_rot * t + ph2))
    mod = np.maximum(mod, 0.05)

    bank = _disc_noise_file()
    # 조각이 적으면 같은 잡음만 반복해서 보게 되므로 사용 비율을 줄임 (40조각 이상이면 그대로)
    real_p = p.disc_real_prob * (min(1.0, len(bank) / 40) if bank is not None else 0.0)
    # 실측을 끈 실험에서도 추첨은 똑같이 함 (난수 흐름 맞춤). 뽑혀도 합성 잡음을 씀
    if bank is not None and rng.random() < real_p and measured_enabled() and real_disc_enabled():
        g = _real_hiss(n, rng, bank)
        real = True
    else:
        # 합성 홈 잡음: 300Hz~8kHz 대역, 고음 쪽으로 약간 기울어짐
        sos = signal.butter(2, [300 / (sr / 2), min(8000, sr / 2 * 0.9) / (sr / 2)], btype="band", output="sos")
        g = signal.sosfilt(sos, rng.standard_normal(n))
        real = False
    snr = _u(rng, p.surface_snr_db)
    g = g / (np.std(g) + 1e-12) * mod
    g = g / (np.std(g) + 1e-12) * sig_rms * 10 ** (-snr / 20)

    # 긴 팝: 저역 위주로 둔하게 떨어지는 짧은 덩어리, 크기는 두꺼운 꼬리 분포
    pops = np.zeros(n)
    rate = _u(rng, p.pop_rate)
    n_pop = int(rng.poisson(rate * n / sr))
    for _ in range(n_pop):
        L = max(8, int(sr * _u(rng, (0.001, 0.006))))
        fc = _u(rng, (800.0, 4000.0))
        b, a = signal.butter(2, fc / (sr / 2))
        sh = signal.lfilter(b, a, rng.standard_normal(L)) * np.exp(-np.arange(L) / (L / 3))
        sh = sh / (np.max(np.abs(sh)) + 1e-12)
        amp = sig_rms * _u(rng, (1.0, 5.0)) * min(rng.pareto(3.0) + 1, 4)
        _add_at(pops, int(rng.integers(0, n)), sh * amp * rng.choice([-1, 1]))

    # 반복 스크래치: 회전 위상의 같은 자리에서 매 바퀴 반복 (크기는 바퀴마다 조금씩 다름)
    period = sr / f_rot
    k = int(rng.integers(p.scratch_n[0], p.scratch_n[1] + 1))
    for _ in range(k):
        phase = rng.uniform(0, period)
        L = max(4, int(sr * _u(rng, (0.0002, 0.0015))))
        sh = rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 2))
        sh = sh / (np.max(np.abs(sh)) + 1e-12)
        base = sig_rms * _u(rng, (1.5, 6.0))
        m = 0
        while phase + m * period < n:
            _add_at(pops, int(phase + m * period + rng.normal(0, 0.0003 * sr)), sh * base * _u(rng, (0.6, 1.2)))
            m += 1

    y = x + g + pops
    return y, {"surface": True, "surface_real": real, "surface_snr_db": snr, "swish_depth": depth,
               "rot_hz": f_rot, "pop_rate": rate, "n_pops": n_pop, "n_scratch": k}


def prior_denoise(x, sr, rng, p: Preset):
    """
    업로더가 쓴 스펙트럼 잡음 제거를 흉내.
    각 주파수의 하위 10% 크기를 '잡음'으로 보고 빼는 단순한 방식이라,
    실제 도구처럼 소리가 뭉개지고 '물 흐르는' 잔여음(뮤지컬 노이즈)이 생김.
    """
    if rng.random() > p.denoise_prob:
        return x, {"denoise": False}
    strength = _u(rng, p.denoise_strength)
    floor = _u(rng, (0.03, 0.2))
    _, _, X = signal.stft(x, sr, nperseg=2048)
    mag = np.abs(X)
    noise = np.percentile(mag, 10, axis=1, keepdims=True)
    gain = np.maximum(1 - strength * noise / (mag + 1e-12), floor)
    _, y = signal.istft(X * gain, sr, nperseg=2048)
    y = y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    return y, {"denoise": True, "denoise_strength": strength, "denoise_floor": floor}


# 코덱: (ffmpeg 인코더, 쓸 때 형식, 인코더가 받는 샘플레이트, 비트레이트 후보 kbps, 읽을 때 형식)
CODECS = {
    "opus": ("libopus", "ogg", 48000, (48, 64, 96, 128, 160), "ogg"),   # 유튜브 기본
    "aac": ("aac", "adts", None, (64, 96, 128, 160), "aac"),            # 유튜브 m4a
    "mp3": ("libmp3lame", "mp3", None, (64, 96, 128, 192), "mp3"),      # 옛 MP3 재업로드
}
CODEC_WEIGHTS = {"opus": 0.5, "aac": 0.3, "mp3": 0.2}


@functools.lru_cache(maxsize=1)
def available_codecs() -> tuple:
    """이 컴퓨터의 ffmpeg 로 쓸 수 있는 코덱 목록 (ffmpeg 가 없으면 빈 튜플)."""
    if shutil.which("ffmpeg") is None:
        return ()
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return tuple(name for name, (enc, *_rest) in CODECS.items() if f" {enc} " in out)


def _align(y: np.ndarray, ref: np.ndarray, max_lag: int = 8192) -> tuple[np.ndarray, int]:
    """
    코덱이 앞에 붙이는 지연(수십~천여 샘플)을 교차상관으로 찾아 제거.
    입력과 정답의 시간 정렬이 어긋나면 학습이 망가지므로 꼭 필요.
    """
    n = min(len(ref), len(y), 44100 * 3)
    a, b = ref[:n], y[:n + max_lag] if len(y) >= n + max_lag else y
    c = signal.correlate(b, a, mode="full", method="fft")
    mid = len(a) - 1                                       # 지연 0 의 위치
    lo, hi = max(0, mid - max_lag), min(len(c), mid + max_lag + 1)
    lag = int(np.argmax(c[lo:hi]) + lo - mid)              # 양수 = y 가 늦음
    y = y[lag:] if lag >= 0 else np.pad(y, (-lag, 0))
    y = y[:len(ref)] if len(y) >= len(ref) else np.pad(y, (0, len(ref) - len(y)))
    return y, lag


def codec_roundtrip(x: np.ndarray, sr: int, codec: str, kbps: int) -> tuple[np.ndarray, int]:
    """ffmpeg 로 인코딩 → 디코딩. 파일을 만들지 않고 메모리(파이프)로 처리."""
    enc, fmt, enc_sr, _, dec_fmt = CODECS[codec]
    pcm = np.ascontiguousarray(x, dtype=np.float32).tobytes()
    cmd = ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0"]
    if enc_sr:
        cmd += ["-ar", str(enc_sr)]
    cmd += ["-c:a", enc, "-b:a", f"{kbps}k", "-f", fmt, "pipe:1"]
    coded = subprocess.run(cmd, input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(["ffmpeg", "-v", "error", "-f", dec_fmt, "-i", "pipe:0",
                          "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"],
                         input=coded, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, dtype=np.float32).astype(np.float64)
    return _align(y, x)


def lossy_codec(x, sr, rng, p: Preset, enabled: bool = True):
    avail = available_codecs() if enabled else ()
    if not avail or rng.random() > p.codec_prob:
        return x, {"codec": []}
    names = list(avail)
    w = np.array([CODEC_WEIGHTS[n] for n in names]); w /= w.sum()
    n_pass = 2 if rng.random() < p.codec_double_prob else 1
    chain = []
    y = x
    peak = np.max(np.abs(y)) + 1e-9
    for _ in range(n_pass):
        c = str(rng.choice(names, p=w))
        kbps = int(rng.choice(CODECS[c][3]))
        # 코덱 입력이 0dBFS 를 넘으면 잘리므로 여유를 두고 넣었다가 되돌림
        y, lag = codec_roundtrip(y / peak * 0.9, sr, c, kbps)
        y = y / 0.9 * peak
        chain.append({"codec": c, "kbps": kbps, "aligned_lag": lag})
    return y, {"codec": chain}


# ---------------------------------------------------------------------------
# 전체 파이프라인
# ---------------------------------------------------------------------------

def degrade(x: np.ndarray, sr: int, preset: str | Preset = "tape_1950s",
            seed: int | None = None, codec: bool = True) -> tuple[np.ndarray, dict]:
    """
    x : (N,) 또는 (N, 2) float 배열, 범위 대략 [-1, 1]
    codec: False 면 손실 압축 단계를 건너뜀 (ffmpeg 가 없을 때 등)
    반환: (열화된 모노 (N,), 사용된 파라미터 dict)
    출력 길이는 입력과 같고, 시간 정렬도 거의 유지됩니다.
    """
    p = PRESETS[preset] if isinstance(preset, str) else preset
    rng = np.random.default_rng(seed)
    meta = {"preset": p.name, "seed": seed, "sr": sr}

    in_peak = float(np.max(np.abs(x)) + 1e-9)

    y, m = to_mono(x.astype(np.float64), rng); meta.update(m)
    y, m = small_room(y, sr, rng, p); meta.update(m)
    y, m = mic_color(y, sr, rng, p); meta.update(m)
    y, m = bandlimit(y, sr, rng, p); meta.update(m)
    y, m = compress(y, sr, rng, p); meta.update(m)
    speed = choose_real_tape(rng, p)                     # 실측 테이프 특성 사용 여부
    y, m = tape_saturate(y, rng, p); meta.update(m)
    y, m = tape_eq(y, sr, rng, p, speed); meta.update(m)
    y, m = wow_flutter(y, sr, rng, p, speed); meta.update(m)
    y, m = tape_noise(y, sr, rng, p, speed); meta.update(m)
    y, m = transfer_filter(y, sr, rng, meta["lowpass_hz"]); meta.update(m)
    y, m = disc_crackle(y, sr, rng, p); meta.update(m)
    y, m = disc_surface(y, sr, rng, p); meta.update(m)

    # 원본과 비슷한 레벨로 맞추고 클리핑 방지 (여기까지가 '음반')
    y = y / (np.max(np.abs(y)) + 1e-9) * min(in_peak, 0.95)

    # 유튜브 업로드 과정
    y, m = prior_denoise(y, sr, rng, p); meta.update(m)
    y, m = lossy_codec(y, sr, rng, p, enabled=codec); meta.update(m)
    pk = np.max(np.abs(y))
    if pk > 0.99:
        y = y / pk * 0.99
    return y.astype(np.float32), meta


PRESET_WEIGHTS = {"tape_1950s": 0.5, "shellac_1925": 0.15, "acoustic_pre1925": 0.15, "mild": 0.2}


def random_preset(rng: np.random.Generator) -> str:
    """학습 데이터용 프리셋 혼합 비율 (주 목표가 1950년대이므로 가장 많이, 공개 데모용 1925년 이전도 충분히)."""
    names = list(PRESET_WEIGHTS)
    return str(rng.choice(names, p=[PRESET_WEIGHTS[n] for n in names]))


# ---------------------------------------------------------------------------
# 명령줄 사용
# ---------------------------------------------------------------------------

def main():
    import soundfile as sf

    ap = argparse.ArgumentParser(description="현대 녹음을 옛날 모노 녹음처럼 열화")
    ap.add_argument("input", help="입력 오디오 (wav/flac)")
    ap.add_argument("output", help="출력 wav (모노)")
    ap.add_argument("--preset", default="tape_1950s", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--meta", help="사용한 파라미터를 저장할 json 경로")
    args = ap.parse_args()

    x, sr = sf.read(args.input, always_2d=False)
    y, meta = degrade(x, sr, args.preset, args.seed)
    sf.write(args.output, y, sr, subtype="PCM_16")
    if args.meta:
        with open(args.meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"저장: {args.output}  (preset={meta['preset']}, lowpass={meta['lowpass_hz']:.0f}Hz)")


if __name__ == "__main__":
    main()
