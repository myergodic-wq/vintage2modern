"""
convert_to_flac.py — 폴더 안의 오디오를 FLAC으로 변환 (무손실 원본만)

인터넷 아카이브의 Musopen 쇼팽 전집처럼 Apple Lossless(.m4a)로 받은 파일을
make_pairs.py 가 읽을 수 있는 FLAC으로 바꿉니다.

손실 압축(AAC, MP3, Vorbis 등)은 고음이 이미 잘려 있어 정답 데이터로 쓸 수 없으므로
변환하지 않고 목록만 보여 줍니다. (.m4a 는 손실(AAC)일 수도, 무손실(ALAC)일 수도 있어
확장자가 아니라 실제 코덱을 검사합니다.)

필요: ffmpeg (ffprobe 포함)
    윈도우: winget install Gyan.FFmpeg   (설치 후 터미널을 새로 열기)

사용 예:
    python convert_to_flac.py 다운로드/musopen-chopin data/clean/Chopin
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

LOSSLESS = {"alac", "flac", "wavpack", "ape", "tta", "mlp", "truehd"}
LOSSLESS_PREFIX = ("pcm_",)
SKIP_EXT = {".pdf", ".txt", ".jpg", ".jpeg", ".png", ".jp2", ".xml", ".torrent", ".sqlite", ".json", ".gz", ".zip"}


def codec_of(path: Path) -> str | None:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=codec_name", "-of", "json", str(path)],
                       capture_output=True, text=True)
    try:
        return json.loads(r.stdout)["streams"][0]["codec_name"]
    except (ValueError, KeyError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser(description="무손실 오디오만 FLAC으로 변환")
    ap.add_argument("src", help="받은 파일이 있는 폴더 (zip은 먼저 풀기)")
    ap.add_argument("dst", help="FLAC을 저장할 폴더 (예: data/clean/Chopin)")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"[오류] {tool} 를 찾을 수 없습니다. 'winget install Gyan.FFmpeg' 로 설치 후 터미널을 새로 여세요.")

    src, dst = Path(args.src), Path(args.dst)
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() not in SKIP_EXT)
    if not files:
        raise SystemExit(f"[오류] {src} 에 파일이 없습니다. zip이면 먼저 압축을 푸세요.")

    done, lossy, skipped, failed = 0, [], 0, []
    for p in files:
        codec = codec_of(p)
        if codec is None:
            continue                               # 오디오가 아닌 파일
        if not (codec in LOSSLESS or codec.startswith(LOSSLESS_PREFIX)):
            lossy.append((p.name, codec))
            continue
        out = dst / p.relative_to(src).with_suffix(".flac")
        if out.exists():
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-map", "0:a:0",
                            "-c:a", "flac", "-compression_level", "5", str(out)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            done += 1
            print(f"  변환: {p.name}  ({codec})")
        else:
            failed.append((p.name, r.stderr.strip()[:200]))
            out.unlink(missing_ok=True)

    print(f"\n변환 {done}개, 이미 있어서 건너뜀 {skipped}개 → {dst}")
    if lossy:
        print(f"손실 압축이라 제외한 파일 {len(lossy)}개 (예: {lossy[0][0]} = {lossy[0][1]})")
        print("  같은 곡이 무손실(FLAC/ALAC/WAV) 판으로도 있으면 그걸 받으세요.")
    for name, err in failed:
        print(f"  [실패] {name}: {err}")


if __name__ == "__main__":
    main()
