"""
audio_io.py — 오디오 읽기 (torch 없이 쓸 수 있게 분리). wav/flac 은 soundfile, 그 외(mp3·webm·m4a·opus)는 ffmpeg.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

SR = 44100


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """
    모노 float32 와 샘플레이트를 돌려줌.
    wav/flac 은 soundfile 로, 손실 압축 형식(.webm/.m4a/.opus/.mp3 등)은 ffmpeg 로 읽음.
    """
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return x.mean(1), sr
    except Exception:
        pass
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        raise SystemExit(f"[오류] {path} 형식을 읽으려면 ffmpeg 가 필요합니다. (winget install Gyan.FFmpeg)")
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "1",
                        "-ar", str(SR), "pipe:1"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise SystemExit(f"[오류] {path} 를 읽지 못했습니다: {r.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy(), SR
