"""
make_sources.py — 폴더 안의 오디오 파일로 sources.csv 를 자동으로 채웁니다.

같은 출처에서 받은 파일들(예: Open Goldberg 32곡)을 한 번에 등록할 때 사용.
파일명을 손으로 치다 틀리는 일을 막아 줍니다.

사용 예:
    python make_sources.py data/clean/goldberg --license CC0 \
        --performer "Kimiko Ishizaka" --url https://www.opengoldbergvariations.org

이미 sources.csv 에 있는 파일은 건너뛰고, 새 파일만 아래에 추가합니다.
라이선스는 본인이 직접 확인한 값을 넣으세요. 이 도구는 확인해 주지 않습니다.
"""

import argparse
import csv
from pathlib import Path

AUDIO_EXT = {".wav", ".flac", ".aiff", ".aif", ".ogg"}
FIELDS = ["file", "title", "performer", "source_url", "license"]


def main():
    ap = argparse.ArgumentParser(description="폴더의 오디오 파일을 sources.csv 에 등록")
    ap.add_argument("folder", help="등록할 파일들이 있는 폴더")
    ap.add_argument("--license", required=True, help="CC0, PD, CC-BY, OWN 등 (직접 확인한 값)")
    ap.add_argument("--performer", default="")
    ap.add_argument("--url", default="", help="다운로드한 출처 주소")
    ap.add_argument("--csv", default="sources.csv")
    ap.add_argument("--clean-root", default="data/clean",
                    help="make_pairs.py 의 --src 폴더 (파일 경로를 이 기준으로 기록)")
    args = ap.parse_args()

    folder, root, csv_path = Path(args.folder), Path(args.clean_root), Path(args.csv)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"[오류] {folder} 에 오디오 파일(flac/wav 등)이 없습니다.")

    existing = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            existing = {r.get("file", "").strip() for r in csv.DictReader(f)}

    new_rows = []
    for p in files:
        try:
            rel = p.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            raise SystemExit(f"[오류] {p} 가 {root} 폴더 안에 있지 않습니다. 파일을 {root} 아래로 옮기세요.")
        if rel in existing:
            continue
        new_rows.append({"file": rel, "title": p.stem, "performer": args.performer,
                         "source_url": args.url, "license": args.license.upper()})

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(new_rows)
    print(f"{len(new_rows)}개 추가 (이미 있던 {len(files) - len(new_rows)}개는 건너뜀) → {csv_path}")


if __name__ == "__main__":
    main()
