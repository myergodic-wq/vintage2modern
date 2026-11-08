"""
model.py — 기준선 모델: 스펙트로그램 U-Net (모노 → 스테레오)

입력 모노의 STFT를 받아, 스테레오를 Mid/Side 형태로 예측합니다.
    Mid  = (L + R) / 2   ← 입력을 출발점으로 '고친 버전'을 예측 (잔차 학습)
    Side = (L - R) / 2   ← 입력에 없는 공간 정보를 새로 생성
    L = Mid + Side,  R = Mid - Side

복소 스펙트로그램을 크기만 0.3제곱으로 압축해서 다룹니다 (위상 유지).
큰 소리와 작은 소리의 차이를 줄여서 학습이 안정적이 됩니다.

이건 '기준선'입니다. 회귀 모델이라 잘린 고음역은 흐릿하게 채우는 경향이 있고,
본격적인 고음 생성은 다음 단계(플로우 매칭)에서 다룹니다.
크기를 작게(기본 base=32에서 약 14M 파라미터, base=16이면 약 4M) 잡아
나중에 소형 장치 배포도 고려했습니다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Spec(nn.Module):
    """STFT ↔ 압축 복소 스펙트로그램 변환."""

    def __init__(self, n_fft=2048, hop=512, power=0.3):
        super().__init__()
        self.n_fft, self.hop, self.power = n_fft, hop, power
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, wav):                      # (B, T) → (B, 2, Fq, Fr)  Fq = n_fft/2 (나이퀴스트 제외)
        X = torch.stft(wav, self.n_fft, self.hop, window=self.window, return_complex=True)
        X = X[:, :-1]                            # 1025 → 1024 bin (2의 거듭제곱으로 맞춤)
        mag = X.abs().clamp_min(1e-8)
        Xc = X * mag.pow(self.power - 1)         # 크기만 압축, 위상 유지
        return torch.stack([Xc.real, Xc.imag], 1)

    def decompress(self, Yc):                    # (B, 2, Fq, Fr) 압축 → 선형 복소 (B, Fq, Fr)
        Y = torch.complex(Yc[:, 0], Yc[:, 1])
        mag = Y.abs().clamp_min(1e-8)
        return Y * mag.pow(1 / self.power - 1)

    def istft(self, Y, length):                  # 선형 복소 (B, Fq, Fr) → (B, T)
        Y = F.pad(Y, (0, 0, 0, 1))               # 나이퀴스트 bin 복원 (0)
        return torch.istft(Y, self.n_fft, self.hop, window=self.window, length=length)


def block(cin, cout):
    g = lambda c: nn.GroupNorm(min(8, c), c)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), g(cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), g(cout), nn.GELU(),
    )


class SpecUNet(nn.Module):
    """mono=True (v7~): 복원만 하는 모노 모델. Side 가지 없이 Mid 만 예측하고, 출력은 기존 도구와 맞추려고
    좌우에 같은 Mid 를 넣은 (B, 2, T). 공간감은 별도 렌더링 단계가 담당 (2026-11-03 결정, PROGRESS r23)."""

    def __init__(self, n_fft=2048, hop=512, base=32, depth=4, mono=False):
        super().__init__()
        self.spec = Spec(n_fft, hop)
        self.depth = depth
        self.mono = mono
        ch = [base * 2 ** i for i in range(depth + 1)]       # 32, 64, 128, 256, 512

        self.inp = block(3, ch[0])                           # 실수부, 허수부, 로그 크기
        self.downs = nn.ModuleList(nn.Conv2d(ch[i], ch[i + 1], 4, stride=2, padding=1) for i in range(depth))
        self.encs = nn.ModuleList(block(ch[i + 1], ch[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(nn.ConvTranspose2d(ch[i + 1], ch[i], 4, stride=2, padding=1)
                                 for i in reversed(range(depth)))
        self.decs = nn.ModuleList(block(ch[i] * 2, ch[i]) for i in reversed(range(depth)))
        self.out = nn.Conv2d(ch[0], 2 if mono else 4, 1)     # Mid(실,허) 수정량 [+ Side 복소 이득(실,허)]
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # 시작은 '입력 그대로 모노'
        # (Side를 '이득 × Mid'로 만들기 때문에 0에서 출발해도 학습 신호가 살아 있음)

    def forward(self, x):                                    # x: (B, 1, T) → (B, 2, T)
        B, _, T = x.shape
        Xc = self.spec(x[:, 0])                              # (B, 2, 1024, Fr)
        Fr = Xc.shape[-1]
        pad = (-Fr) % (2 ** self.depth)
        Xc = F.pad(Xc, (0, pad))

        logmag = torch.log1p(Xc.pow(2).sum(1, keepdim=True).sqrt() * 10)
        h = self.inp(torch.cat([Xc, logmag], 1))
        skips = [h]
        for down, enc in zip(self.downs, self.encs):
            h = enc(down(h)); skips.append(h)
        skips.pop()
        for up, dec in zip(self.ups, self.decs):
            h = dec(torch.cat([up(h), skips.pop()], 1))
        o = self.out(h).float()

        mid = (Xc + o[:, 0:2])[..., :Fr]
        if self.mono:
            M = self.spec.istft(self.spec.decompress(mid), T)
            return torch.stack([M, M], 1)
        gain = torch.tanh(o[:, 2:4, :, :Fr])                 # 각 주파수·시간마다 |이득| ≲ 1.4
        Mlin = self.spec.decompress(mid)
        # Side = 복소 이득 × Mid. 이득이 0이어도 기울기 = Mid 이므로 처음부터 학습됨.
        # (이전 버전은 Side를 압축 영역에서 0으로 시작해 기울기가 0 → 영원히 모노였음)
        Slin = Mlin * torch.complex(gain[:, 0], gain[:, 1])
        M = self.spec.istft(Mlin, T)
        S = self.spec.istft(Slin, T)
        return torch.stack([M + S, M - S], 1)


if __name__ == "__main__":
    m = SpecUNet()
    print(f"파라미터: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M")
    y = m(torch.randn(2, 1, 44100 * 2))
    print("출력:", tuple(y.shape))
