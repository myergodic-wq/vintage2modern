"""
dataset.py — make_pairs.py 가 만든 쌍 데이터를 읽는 PyTorch Dataset

각 샘플: (input 모노 (1, T), target 스테레오 (2, T))
같은 위치에서 무작위로 잘라내므로 입력과 정답의 시간 정렬이 유지됩니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, root: str, split: str = "train", crop_sec: float | None = 6.0,
                 sr: int = 44100, gain_aug: bool = True, level_match: bool = True):
        self.level_match = level_match
        self.root = Path(root)
        with open(self.root / "meta.jsonl", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
            if not rows:
                hint = ("meta.jsonl 이 비어 있습니다. make_pairs.py 가 쌍을 0개 만들었어요.\n"
                        "make_pairs.py 를 다시 실행하고 '건너뛴 파일' 목록을 확인하세요 "
                        "(sources.csv 파일명 불일치가 가장 흔한 원인).")
            else:
                hint = (f"현재 train {counts['train']}개, val {counts['val']}개.\n"
                        "최신 make_pairs.py 로 쌍을 다시 만드세요 (파일이 적어도 양쪽이 생기도록 수정됨).")
            raise ValueError(f"'{split}' 쌍이 없습니다. {hint}")
        self.crop = int(crop_sec * sr) if crop_sec else None
        self.gain_aug = gain_aug and split == "train"
        self.split = split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x, _ = sf.read(self.root / r["input"], dtype="float32")       # (T,)
        y, _ = sf.read(self.root / r["target"], dtype="float32")      # (T, 2)
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]

        if self.crop and n > self.crop:
            # 검증은 항상 같은 위치(가운데)에서 자름 → 손실 비교가 안정적
            # (torch 난수를 써야 DataLoader 워커마다 다른 위치가 뽑힘)
            if self.split == "train":
                s = int(torch.randint(0, n - self.crop, (1,)))
            else:
                s = (n - self.crop) // 2
            x, y = x[s:s + self.crop], y[s:s + self.crop]

        # 입력 음량을 정답(가운데 채널)에 맞춤. 열화 과정의 포화·압축 때문에 입력이
        # 평균 8dB가량 크게 저장되는데, 이걸 그대로 두면 모델이 '소리 고치기' 대신
        # '음량 줄이기'를 배우느라 학습을 낭비함. (저장된 파일은 그대로, 읽을 때만 맞춤)
        rms_x = float((x ** 2).mean() ** 0.5)
        rms_t = float((y.mean(1) ** 2).mean() ** 0.5)
        if self.level_match and rms_x > 1e-6 and rms_t > 1e-6:
            x = x * (rms_t / rms_x)

        if self.gain_aug:
            # 입력·정답에 같은 배율 → 음량에 무관하게 학습
            g = float(10 ** (float(torch.empty(1).uniform_(-12, 3)) / 20))
            x, y = x * g, y * g

        return torch.from_numpy(x[None].copy()), torch.from_numpy(y.T.copy())
