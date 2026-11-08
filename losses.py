"""
losses.py — 학습 손실

1) 압축 스펙트럼 손실 (주 손실)
   모델과 같은 방식(크기 0.3제곱)으로 압축한 스펙트럼끼리 비교합니다.
   조용한 대역(잘려 나간 고음 등)에서도 학습 신호가 약해지지 않게 하는 핵심 장치입니다.
   (선형 영역 손실만 쓰면 작은 소리일수록 기울기가 수십~수백 배 약해져
    고음 복원을 거의 배우지 못함)

2) 다중 해상도 STFT 손실 (보조)
   짧은 창(타건)과 긴 창(배음)을 함께 봅니다.

L/R 뿐 아니라 Side(좌우 차이)도 따로 봐서 스테레오 공간감을 학습시킵니다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompressedSpecLoss(nn.Module):
    def __init__(self, n_fft=2048, hop=512, power=0.3, mag_w=0.7, floor=1e-3):
        super().__init__()
        self.n_fft, self.hop, self.power, self.mag_w = n_fft, hop, power, mag_w
        # floor: 0 근처에서 기울기가 무한대로 튀는 것을 막는 바닥값.
        # (STFT 크기 기준 1e-3 ≈ 전체 음량 대비 약 -110dB, 들리지 않는 수준)
        self.floor = floor
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def _c(self, x):
        X = torch.stft(x, self.n_fft, self.hop, window=self.window, return_complex=True)
        # |X|=0 에서도 미분이 유한하도록 sqrt(|X|^2 + floor^2) 사용
        mag = (X.real.pow(2) + X.imag.pow(2) + self.floor ** 2).sqrt()
        scale = mag.pow(self.power - 1)
        return X * scale, mag.pow(self.power)       # 압축 복소, 압축 크기

    def forward(self, y_hat, y):          # (B, T)
        Ch, Mh = self._c(y_hat)
        C, M = self._c(y)
        complex_l = (Ch - C).abs().pow(2).mean()
        mag_l = (Mh - M).pow(2).mean()
        # 크기(위상 무관) 위주 + 복소(위상 포함) 일부
        return self.mag_w * mag_l + (1 - self.mag_w) * complex_l


class MultiResSTFTLoss(nn.Module):
    def __init__(self, ffts=(512, 1024, 2048, 4096)):
        super().__init__()
        self.ffts = ffts
        for n in ffts:
            self.register_buffer(f"w{n}", torch.hann_window(n), persistent=False)

    def _one(self, y_hat, y, n):
        w = getattr(self, f"w{n}")
        S_hat = torch.stft(y_hat, n, n // 4, window=w, return_complex=True).abs()
        S = torch.stft(y, n, n // 4, window=w, return_complex=True).abs()
        sc = torch.linalg.norm(S - S_hat) / (torch.linalg.norm(S) + 1e-7)           # 스펙트럼 수렴
        logmag = (torch.log(S + 1e-5) - torch.log(S_hat + 1e-5)).abs().mean()       # 로그 크기 L1
        return sc + logmag

    def forward(self, y_hat, y):          # (B, T)
        return sum(self._one(y_hat, y, n) for n in self.ffts) / len(self.ffts)


class MonoLoss(nn.Module):
    """
    모노 복원 모델(v7~)용: 출력 가운데 vs 정답 가운데((L+R)/2) 의 다중해상도 손실.
    StereoLoss 의 L/R 부분과 같은 손실 함수(comp_w=0, mr_w=1)를 가운데 하나에 적용. 로그 형식은 StereoLoss 와 같게 (side=0).
    """

    def __init__(self):
        super().__init__()
        self.mr = MultiResSTFTLoss()

    def forward(self, y_hat, y):          # (B, 2, T)
        y_hat, y = y_hat.float(), y.float()
        l = self.mr(y_hat.mean(1), y.mean(1))
        return l, {"lr": l.item(), "side": 0.0}


class StereoLoss(nn.Module):
    """
    전체 = 압축 손실(L/R + Side) × comp_w  +  다중해상도 손실(L/R + Side) × mr_w
    로그에 보이는 값: lr = L/R 부분, side = 좌우 차이 부분 (둘 다 두 손실의 합)
    """

    # comp_w 기본 0: 2026-10-31 비교 실험(compare_fixes.py, 1000스텝)에서 압축 손실을 켜면
    # 5-10kHz 오차 4.4→8.0dB, 10-16kHz 3.7→5.9dB 로 고음 복원이 오히려 나빠짐. 실험용으로만 남겨 둠.
    def __init__(self, side_w=0.5, comp_w=0.0, mr_w=1.0):
        super().__init__()
        self.comp = CompressedSpecLoss()
        self.mr = MultiResSTFTLoss()
        self.side_w, self.comp_w, self.mr_w = side_w, comp_w, mr_w

    def _both(self, a, b):
        return self.comp_w * self.comp(a, b) + self.mr_w * self.mr(a, b)

    def forward(self, y_hat, y):          # (B, 2, T)
        y_hat, y = y_hat.float(), y.float()
        lr = self._both(y_hat.flatten(0, 1), y.flatten(0, 1))
        side = self._both((y_hat[:, 0] - y_hat[:, 1]) / 2, (y[:, 0] - y[:, 1]) / 2)
        return lr + self.side_w * side, {"lr": lr.item(), "side": side.item()}
