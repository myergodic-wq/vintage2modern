"""
make_pairs.py — 깨끗한 현대 녹음 폴더로부터 학습용 쌍 데이터를 만듭니다.

    sources.csv 에 등록된 (라이선스가 확인된) 파일만 사용
      → 일정 길이로 자르기
      → 구간마다 여러 개의 열화 버전 생성
      → target/ (정답, 스테레오)  input/ (열화, 모노)  meta.jsonl

사용 예:
    python make_pairs.py --src data/clean --sources sources.csv --out data/pairs \
        --segment 10 --variants 4

sources.csv 형식 (헤더 필수):
    file,title,performer,source_url,license
    goldberg_aria.flac,Goldberg Variations - Aria,Kimiko Ishizaka,https://...,CC0

license 열이 ALLOWED_LICENSES 에 없는 파일은 건너뜁니다.
(상업적으로도 쓸 수 있는 데이터만 학습에 들어가도록 하는 안전장치)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from degrade import (DISC_NOISE, PRESET_WEIGHTS, TAPE_PROFILE, available_codecs, degrade, load_disc_noise,
                     load_tape_profile, measured_enabled, random_preset, real_disc_enabled)

# 상업 이용까지 가능한 라이선스만 허용
ALLOWED_LICENSES = {"CC0", "PD", "PUBLIC-DOMAIN", "CC-BY", "CC-BY-4.0", "OWN"}
AUDIO_EXT = {".wav", ".flac", ".aiff", ".aif", ".ogg"}


def load_sources(csv_path: Path) -> dict[str, dict]:
    # utf-8-sig: 윈도우 메모장·엑셀이 붙이는 BOM 문자를 자동 제거
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]
    # 키는 소문자 + '/' 구분자로 통일 (윈도우의 대소문자·역슬래시 차이 무시)
    return {_key(r["file"]): r for r in rows if r.get("file")}


def _key(p: str) -> str:
    return p.replace("\\", "/").strip().lower()


def to_stereo(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x], axis=1) if x.ndim == 1 else x[:, :2]


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g, axis=0)


def split_of_key(filename: str) -> int:
    """파일명으로 정해지는 고정 순서 (실행할 때마다 같은 파일이 검증용으로 뽑히게)."""
    return zlib.crc32(filename.encode())


def unique_id(rel: str) -> str:
    """
    폴더 경로까지 포함한 고유 이름. ('Goldberg/01.flac' → 'Goldberg__01')
    파일명만 쓰면 다른 폴더의 같은 이름 파일이 정답 파일을 덮어써서
    입력·정답이 엉뚱하게 짝지어짐.
    """
    return str(Path(rel).with_suffix("")).replace("\\", "/").replace("/", "__")


def process_file(job: dict) -> tuple[list[dict], str]:
    """파일 하나를 구간으로 자르고 열화 버전을 만듦. (병렬 작업자에서 실행)"""
    path, rel, info, lic = Path(job["path"]), job["rel"], job["info"], job["lic"]
    sr_out, seg_len, hop = job["sr"], job["seg_len"], job["hop"]
    out = Path(job["out"])
    # 파일마다 고정된 난수 → 작업자 수·처리 순서와 무관하게 같은 결과
    rng = np.random.default_rng([job["seed"], zlib.crc32(rel.encode())])

    x, sr = sf.read(path, always_2d=False, dtype="float32")
    x = to_stereo(resample(x, sr, sr_out)).astype(np.float32)
    starts = list(range(0, len(x) - seg_len + 1, hop))
    # 파일이 적을 때: 마지막 구간들을 검증용으로 (구간이 2개 이상이면 최소 1개)
    n_val_seg = max(1, round(len(starts) * job["val_ratio"])) if len(starts) >= 2 else 0
    first_val = starts[-n_val_seg] if n_val_seg else len(x)
    uid = unique_id(rel)

    rows = []
    for k, start in enumerate(starts):
        if job["by_file"]:
            split = job["file_split"]
        else:
            split = "val" if start >= first_val else "train"
            if split == "train" and start + seg_len > first_val:
                continue        # 검증 구간과 겹치는 학습 구간은 버림 (--hop 을 짧게 준 경우)
        tgt = x[start:start + seg_len]
        rms_db = 20 * np.log10(np.sqrt(np.mean(tgt ** 2)) + 1e-12)
        if rms_db < job["min_rms_db"]:
            continue
        seg_id = f"{uid}_{k:05d}"
        sf.write(out / "target" / f"{seg_id}.flac", tgt, sr_out)

        for v in range(job["variants"]):
            preset = random_preset(rng)
            seed = int(rng.integers(0, 2**31 - 1))
            y, meta = degrade(tgt, sr_out, preset, seed, codec=job["codec"])
            in_name = f"{seg_id}_v{v}.flac"
            sf.write(out / "input" / in_name, y, sr_out)
            rows.append({
                "input": f"input/{in_name}",
                "target": f"target/{seg_id}.flac",
                "split": split,
                "source_file": rel,
                "license": lic,
                "source_url": info.get("source_url", ""),
                "start_sec": start / sr_out,
                "measured": measured_enabled(),
                "real_disc": real_disc_enabled(),
                "degrade": meta,
            })
    return rows, rel


def main():
    ap = argparse.ArgumentParser(description="학습용 (열화 모노, 정답 스테레오) 쌍 생성")
    ap.add_argument("--src", required=True, help="깨끗한 녹음 폴더")
    ap.add_argument("--sources", required=True, help="출처·라이선스 기록 csv")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--segment", type=float, default=10.0, help="구간 길이(초)")
    ap.add_argument("--hop", type=float, default=None, help="구간 간격(초), 기본=segment")
    ap.add_argument("--variants", type=int, default=4, help="구간당 열화 버전 수")
    ap.add_argument("--min-rms-db", type=float, default=-45.0, help="이보다 조용한 구간은 건너뜀")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="동시에 처리할 파일 수 (기본: CPU 코어 수, 최대 8)")
    ap.add_argument("--no-codec", action="store_true",
                    help="유튜브 손실 압축 단계 끄기 (ffmpeg 가 없을 때만. 유튜브 입력 성능이 떨어짐)")
    ap.add_argument("--no-measured", action="store_true",
                    help="보정 효과 분리 실험: 실측 테이프 특성·실제 홈 잡음 대신 합성 방식 사용 (나머지는 동일)")
    ap.add_argument("--no-real-disc", action="store_true",
                    help="실제 78rpm 홈 잡음(UCSB DAHR 에서 수확)을 쓰지 않음 — 공개 가중치용 (DAHR 이용 조건). 실측 테이프 특성은 그대로")
    args = ap.parse_args()
    if args.no_measured:                               # 작업자 프로세스에도 전달되도록 환경 변수로
        os.environ["V2M_NO_MEASURED"] = "1"
    if args.no_real_disc:
        os.environ["V2M_NO_REAL_DISC"] = "1"

    src, out = Path(args.src), Path(args.out)
    sources = load_sources(Path(args.sources))

    seg_len = int(args.segment * args.sr)
    hop = int((args.hop or args.segment) * args.sr)
    n_pairs, skipped = 0, []

    # 손실 압축(유튜브) 단계에 필요한 ffmpeg 확인
    codecs = () if args.no_codec else available_codecs()
    if not args.no_codec and not codecs:
        raise SystemExit("[오류] ffmpeg 를 찾을 수 없습니다. 유튜브 손실 압축 시뮬레이션에 필요합니다.\n"
                         "  윈도우: winget install Gyan.FFmpeg  (설치 후 터미널을 새로 여세요)\n"
                         "  ffmpeg 없이 하려면 --no-codec (유튜브 입력 성능이 떨어짐)")
    print(f"손실 압축 코덱: {', '.join(codecs) if codecs else '사용 안 함'}")
    if args.no_measured:
        print("[보정 효과 분리 실험] 실측 자료 끔 (--no-measured): 테이프·홈 잡음 모두 합성 방식")
    prof = load_tape_profile()
    if args.no_measured:
        print("실측 테이프 특성: 끔 (합성 방식만 사용)")
    elif prof is None:
        print(f"실측 테이프 특성: 없음 (합성 방식만 사용). 만들려면 calibrate_tape.py → {TAPE_PROFILE}")
    else:
        parts = [k for k in ("noise_7_5", "noise_3_75", "eq_db_7_5", "eq_db_3_75", "traj_7_5", "traj_3_75") if k in prof]
        print(f"실측 테이프 특성: 사용 ({', '.join(parts)})")
    bank = load_disc_noise()
    if args.no_measured or args.no_real_disc:
        print("실제 78rpm 홈 잡음: 끔 (합성 표면 잡음만 사용)" + (" — --no-real-disc" if args.no_real_disc else ""))
    elif bank is None:
        print(f"실제 78rpm 홈 잡음: 없음 (합성 표면 잡음만 사용). 만들려면 harvest_disc_noise.py → {DISC_NOISE}")
    else:
        print(f"실제 78rpm 홈 잡음: 사용 ({len(bank)}조각)")
    print("프리셋 비율: " + ", ".join(f"{k} {v:.0%}" for k, v in PRESET_WEIGHTS.items()))
    if out.exists() and (out / "meta.jsonl").exists():
        print(f"[주의] {out} 에 이전 결과가 있습니다. 섞이지 않도록 새 폴더를 쓰거나 지우고 다시 실행하길 권합니다.")
    (out / "target").mkdir(parents=True, exist_ok=True)
    (out / "input").mkdir(parents=True, exist_ok=True)

    # 0) 폴더에 무엇이 있는지 먼저 확인
    if not src.is_dir():
        raise SystemExit(f"[오류] 폴더가 없습니다: {src.resolve()}")
    all_files = [p for p in src.rglob("*") if p.is_file()]
    audio_files = sorted(p for p in all_files if p.suffix.lower() in AUDIO_EXT)
    print(f"{src.resolve()} 에서 오디오 파일 {len(audio_files)}개 발견")
    if not audio_files:
        from collections import Counter
        exts = Counter(p.suffix.lower() or "(확장자 없음)" for p in all_files)
        if not all_files:
            raise SystemExit("[오류] 폴더가 비어 있습니다. 녹음 파일을 이 폴더에 넣으세요.")
        msg = ", ".join(f"{e} {n}개" for e, n in exts.most_common())
        tips = []
        if ".m4a" in exts:
            tips.append(".m4a 는 무손실(Apple Lossless)일 수 있습니다. convert_to_flac.py 로 변환하세요 "
                        "(손실 압축(AAC)인 파일은 도구가 알아서 제외합니다).")
        if ".mp3" in exts:
            tips.append("MP3는 고음이 이미 잘려 있어 정답 데이터로 쓸 수 없습니다. FLAC(또는 WAV)으로 받으세요.")
        if ".zip" in exts:
            tips.append("zip 파일은 먼저 압축을 풀어 주세요.")
        raise SystemExit(f"[오류] 지원하는 오디오 파일이 없습니다. 폴더 안 파일: {msg}\n" + "\n".join(tips))

    # 1) 사용할 파일 고르기 (출처·라이선스 확인)
    eligible = []
    for path in audio_files:
        rel = path.relative_to(src).as_posix()
        info = sources.get(_key(rel)) or sources.get(_key(path.name))
        if info is None:
            skipped.append((rel, "sources.csv에 없음 (파일명 철자·확장자 확인)"))
            continue
        lic = info.get("license", "").strip().upper()
        if lic not in ALLOWED_LICENSES:
            skipped.append((rel, f"허용되지 않은 라이선스: {lic}"))
            continue
        eligible.append((path, rel, info, lic))

    # 2) train/val 나누기
    #    파일이 충분하면 파일 단위 (같은 곡이 양쪽에 섞이지 않게),
    #    적으면 각 파일의 마지막 부분을 검증용으로 (양쪽 다 반드시 생기게)
    by_file = len(eligible) >= 5
    if by_file:
        order = sorted(eligible, key=lambda e: split_of_key(e[1]))
        n_val = max(1, round(len(eligible) * args.val_ratio))
        val_files = {e[1] for e in order[:n_val]}
        print(f"파일 {len(eligible)}개 → 파일 단위로 나눔 (검증용 {n_val}개)")
    elif eligible:
        print(f"파일이 {len(eligible)}개뿐이라, 각 파일의 마지막 {args.val_ratio:.0%}를 검증용으로 씁니다.")

    # 같은 고유 이름이 둘 이상이면 중단 (예: 'a.flac' 과 'a.wav' 가 같은 폴더에)
    from collections import Counter
    dup = [u for u, n in Counter(unique_id(e[1]) for e in eligible).items() if n > 1]
    if dup:
        raise SystemExit(f"[오류] 확장자만 다른 같은 이름의 파일이 있습니다 (정답 파일이 덮어써짐): {dup[:5]}")

    jobs = [{
        "path": str(path), "rel": rel, "info": info, "lic": lic,
        "by_file": by_file, "file_split": ("val" if by_file and rel in val_files else "train"),
        "sr": args.sr, "seg_len": seg_len, "hop": hop, "val_ratio": args.val_ratio,
        "min_rms_db": args.min_rms_db, "variants": args.variants, "seed": args.seed,
        "codec": bool(codecs), "out": str(out),
    } for path, rel, info, lic in eligible]

    all_rows = []
    print(f"작업자 {args.workers}개로 처리 시작 (파일 {len(jobs)}개)")
    if args.workers <= 1:
        results = (process_file(j) for j in jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = (f.result() for f in as_completed([pool.submit(process_file, j) for j in jobs]))
    for i, (rows, rel) in enumerate(results, 1):
        all_rows.extend(rows)
        print(f"  [{i}/{len(jobs)}] {rel}: {len(rows)}쌍", flush=True)
    if args.workers > 1:
        pool.shutdown()

    # 처리 순서와 무관하게 같은 파일이 되도록 정렬해서 저장
    all_rows.sort(key=lambda r: r["input"])
    split_count = {"train": 0, "val": 0}
    with open(out / "meta.jsonl", "w", encoding="utf-8") as meta_f:
        for r in all_rows:
            meta_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            split_count[r["split"]] += 1
    n_pairs = len(all_rows)

    print(f"생성된 쌍: {n_pairs}개 (train {split_count['train']}, val {split_count['val']}) → {out}")
    if skipped:
        print(f"건너뛴 파일 {len(skipped)}개:")
        for f, why in skipped:
            print(f"  - {f}: {why}")
    if n_pairs == 0:
        print("\n[주의] 쌍이 하나도 만들어지지 않았습니다.")
        print("  - sources.csv 의 file 열이 실제 파일명과 정확히 같은지 확인하세요.")
        print(f"  - 파일 길이가 --segment ({args.segment}초)보다 긴지 확인하세요.")
        print(f"  - 오디오 확장자: {sorted(AUDIO_EXT)} (mp3는 지원 안 함)")


if __name__ == "__main__":
    main()
