"""
inspect_aalto.py — Aalto 테이프 녹음기 데이터셋(압축 해제한 폴더)의 구조를 요약

보정 도구를 만들기 전에 실제 파일 구성(샘플레이트, 길이, 채널, npy 모양)을 확인하기 위한 것.
출력 전체를 복사해서 알려 주세요.

사용 예:
    python inspect_aalto.py C:\aalto\neural-tape-audio_AKAI
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser(description="Aalto 테이프 데이터셋 구조 요약")
    ap.add_argument("root", help="tar 를 푼 폴더")
    ap.add_argument("--examples", type=int, default=2, help="폴더당 자세히 볼 파일 수")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"[오류] 폴더가 없습니다: {root}")

    groups = defaultdict(list)
    for p in sorted(root.rglob("*")):
        if p.is_file():
            groups[p.parent].append(p)

    total_files = sum(len(v) for v in groups.values())
    print(f"폴더 {len(groups)}개, 파일 {total_files}개\n")

    for folder, files in groups.items():
        rel = folder.relative_to(root)
        kinds = defaultdict(int)
        for f in files:
            kinds[f.suffix.lower()] += 1
        print(f"[{rel}]  " + ", ".join(f"{k} {n}개" for k, n in kinds.items()))
        print("   예시 이름:", ", ".join(f.name for f in files[:4]))

        wavs = [f for f in files if f.suffix.lower() == ".wav"]
        total_sec, srs, chans = 0.0, set(), set()
        for f in wavs:
            try:
                info = sf.info(f)
                total_sec += info.duration
                srs.add(info.samplerate)
                chans.add(info.channels)
            except Exception as e:
                print(f"   [읽기 실패] {f.name}: {e}")
        if wavs:
            print(f"   wav 합계 {total_sec / 60:.1f}분, 샘플레이트 {sorted(srs)}, 채널 {sorted(chans)}")
            for f in wavs[:args.examples]:
                info = sf.info(f)
                x, _ = sf.read(f, frames=min(info.frames, info.samplerate * 5), always_2d=True)
                rms = 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)
                print(f"   - {f.name}: {info.duration:.1f}초, {info.subtype}, 앞 5초 RMS {rms:.1f} dBFS, "
                      f"최대 {np.abs(x).max():.3f}")

        for f in [f for f in files if f.suffix.lower() == ".npy"][:args.examples]:
            try:
                a = np.load(f, allow_pickle=False)
                print(f"   - {f.name}: 모양 {a.shape}, 자료형 {a.dtype}, "
                      f"최소 {a.min():.4g}, 최대 {a.max():.4g}, 평균 {a.mean():.4g}")
            except Exception as e:
                print(f"   [npy 읽기 실패] {f.name}: {e}")
        print()


if __name__ == "__main__":
    main()
