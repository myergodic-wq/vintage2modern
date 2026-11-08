"""
harvest_disc_noise.py — 78rpm 음반 디지털 파일에서 '음악 없는 홈 잡음'만 잘라 모음

디지털화된 78rpm 파일은 보통 음악 시작 전/끝난 뒤에 몇 초간 바늘이 홈을 긁는 소리만 있습니다.
그 부분이 진짜 음반 표면 잡음이라, 모아서 학습용 열화에 씁니다.

결과: assets/disc_noise.npz  (degrade.py 가 있으면 자동으로 사용)

사용 예:
    python harvest_disc_noise.py "C:\\disc_noise_src"

입력: 1925년 이전 발매 공유 저작물 78rpm 파일 (UCSB DAHR 등), mp3/flac/wav 아무거나.
악기 종류는 상관없음 (음악 부분은 쓰지 않음).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import signal

from infer import load_audio

SR = 44100
EXT = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".webm", ".opus"}


def noise_segments(x: np.ndarray, min_sec: float, gap_db: float) -> list[tuple[int, int, str]]:
    """
    파일 앞/뒤에서 '음악 시작 전' 과 '음악 끝난 뒤' 구간을 찾음.
    150Hz~3kHz 대역 에너지를 50ms 단위로 보고, 처음(끝) 수준보다 gap_db 이상 커지는
    지점이 0.3초 이상 이어지면 음악 시작(끝)으로 판단.
    """
    b = signal.sosfiltfilt(signal.butter(4, [150 / (SR / 2), 3000 / (SR / 2)], "band", output="sos"), x)
    fr = int(0.05 * SR)
    nfr = len(b) // fr
    if nfr < 40:
        return []
    db = 20 * np.log10(np.sqrt(np.mean(b[: nfr * fr].reshape(nfr, fr) ** 2, axis=1)) + 1e-9)
    music = np.median(db)
    out = []

    def scan(order: np.ndarray, label: str):
        live = order[db[order] > -80]                 # 디지털 무음(0) 은 건너뜀
        if len(live) < 20:
            return
        base = np.median(db[live[:10]])
        if music - base < gap_db:                     # 시작부터 음악이면 잡음 구간 없음
            return
        above = db[live] > base + gap_db * 0.6
        run = 0
        for i, a in enumerate(above):
            run = run + 1 if a else 0
            if run >= 6:                              # 0.3초 이상 이어짐 → 음악
                end = i - run - 3                     # 시작점보다 0.15초 앞에서 자름
                if end >= 1:
                    fs = sorted([live[1], live[end]])
                    s, e = fs[0] * fr, (fs[1] + 1) * fr
                    if e - s >= min_sec * SR:
                        out.append((s, e, label))
                return

    scan(np.arange(nfr), "start")
    scan(np.arange(nfr)[::-1], "end")
    return out


def file_hash(p: Path) -> str:
    """파일 내용의 SHA-256. 이름을 바꿔 복사한 파일도 같은 값."""
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def disc_id(name: str) -> str | None:
    """
    DAHR 파일 이름에서 '물리적 음반' 식별자 (회사_음반번호).
    예: cusb_victor_1008_01_b28691_05d.mp3 → 'victor_1008' (Victor 1008 음반, 면은 b28691)
    같은 음반의 앞뒷면은 표면 재질·닳은 정도가 같아 홈 잡음이 비슷하므로, 평가용과 같은 음반은 수확에서 뺌.
    DAHR 형식이 아니면 None.
    """
    import re
    m = re.match(r"cusb_([a-z]+)_([0-9a-z]+)_\d+_", name.lower())
    return f"{m.group(1)}_{m.group(2)}" if m else None


def flatness(c: np.ndarray) -> float:
    """
    300Hz~4kHz 스펙트럼 평탄도 (0~1). 잡음은 고르게 퍼져 높고(합성 0.99, 실제 1920 홈 잡음 0.94),
    음악은 배음 봉우리가 뾰족해서 낮음(피아노 0.04~0.05), 섞이면 중간(0.5). 기준 0.6.
    """
    f, P = signal.welch(c, SR, nperseg=2048)
    band = P[(f >= 300) & (f <= 4000)] + 1e-20
    return float(np.exp(np.mean(np.log(band))) / np.mean(band))


def main():
    ap = argparse.ArgumentParser(description="78rpm 파일의 앞뒤 홈 잡음 수확 → assets/disc_noise.npz")
    ap.add_argument("folder")
    ap.add_argument("--out", default="assets/disc_noise.npz")
    ap.add_argument("--clip-sec", type=float, default=1.5, help="저장할 조각 길이")
    ap.add_argument("--min-sec", type=float, default=1.0, help="이보다 짧은 잡음 구간은 버림")
    ap.add_argument("--gap-db", type=float, default=10.0, help="음악이 잡음보다 이만큼 커야 시작으로 봄")
    ap.add_argument("--min-flatness", type=float, default=0.6,
                    help="이보다 스펙트럼이 뾰족한 조각은 음악이 섞인 것으로 보고 버림")
    ap.add_argument("--eval-dir", default="real_tests",
                    help="평가 전용 폴더. 여기 있는 파일과 내용이 같은 파일은 수확에서 자동 제외")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.folder).rglob("*") if p.suffix.lower() in EXT)
    if not files:
        raise SystemExit(f"[오류] {args.folder} 에 오디오 파일이 없습니다.")

    # 평가 데이터 누수 방지: 이름이 달라도 '내용'이 같으면 제외 (파일 해시로 비교)
    # 평가 데이터 누수 방지 2: 평가용과 '같은 음반'(앞뒷면)이면 제외 (DAHR 파일 이름의 회사_음반번호)
    eval_hashes, eval_discs = {}, {}
    ev = Path(args.eval_dir)
    if ev.is_dir():
        for q in ev.rglob("*"):
            if q.is_file() and q.suffix.lower() in EXT:
                eval_hashes[file_hash(q)] = q
                d = disc_id(q.name)
                if d:
                    eval_discs[d] = q
    else:
        print(f"[주의] 평가 전용 폴더 {ev.resolve()} 가 없어 누수 검사를 하지 못했습니다.")
    kept = []
    for p in files:
        h = file_hash(p)
        d = disc_id(p.name)
        if h in eval_hashes:
            print(f"  [제외: 평가 전용 파일과 동일] {p.name}  =  {eval_hashes[h]}")
        elif d and d in eval_discs:
            print(f"  [제외: 평가 전용과 같은 음반 {d}] {p.name}  ~  {eval_discs[d].name}")
        else:
            kept.append(p)
    if len(kept) < len(files):
        print(f"평가 전용 파일과 같은 {len(files) - len(kept)}개를 제외했습니다.\n")
    files = kept
    if not files:
        raise SystemExit("[오류] 제외하고 나니 수확할 파일이 없습니다.")
    clip = int(args.clip_sec * SR)
    hp = signal.butter(2, 20 / (SR / 2), "high", output="sos")
    clips, used, rejected, source_log = [], 0, 0, []
    for p in files:
        x, sr = load_audio(str(p))
        if sr != SR:
            from math import gcd
            g = gcd(sr, SR)
            x = signal.resample_poly(x, SR // g, sr // g)
        segs = noise_segments(x.astype(np.float64), args.min_sec, args.gap_db)
        got = 0
        for s, e, label in segs:
            seg = signal.sosfiltfilt(hp, x[s:e])
            if len(seg) < clip:                       # 짧으면 앞뒤로 이어 붙여 길이 맞춤
                seg = np.concatenate([seg, seg[::-1]] * (clip // len(seg) + 1))[:clip]
            for k in range(0, len(seg) - clip + 1, clip):
                c = seg[k:k + clip]
                fl = flatness(c)
                if fl < args.min_flatness:            # 음악이 섞인 조각
                    rejected += 1
                    continue
                clips.append(np.clip(c / (np.std(c) + 1e-12) * 0.1, -1, 1))
                got += 1
        used += bool(got)
        if got:
            source_log.append(f"{got}\t{p.relative_to(args.folder).as_posix()}\t{file_hash(p)[:16]}")
        if segs:
            desc = ", ".join(f"{lab} {(e - s) / SR:.1f}초" for s, e, lab in segs)
            print(f"  {p.name[:60]}: {desc} → {got}조각")
    print(f"\n음악이 섞여 버린 조각: {rejected}개 (평탄도 {args.min_flatness} 미만)")

    if not clips:
        raise SystemExit("[오류] 잡음 구간을 하나도 찾지 못했습니다. 음악이 바로 시작하는 파일들일 수 있어요. "
                         "--gap-db 6 으로 다시 해 보거나 다른 파일을 추가하세요.")
    bank = (np.stack(clips) * 32767).astype(np.int16)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # 어떤 파일에서 몇 조각을 가져왔는지 함께 저장 (논문·감사용 기록)
    np.savez_compressed(args.out, noise=bank, sources=np.array(source_log), credits=np.array(
        "Groove noise harvested from public-domain (pre-1926 US) 78rpm transfers, e.g. UCSB DAHR."))
    log_path = Path(args.out).with_suffix(".sources.txt")
    log_path.write_text("\n".join(source_log) + "\n", encoding="utf-8")
    print(f"출처 목록: {log_path}")
    print(f"\n파일 {len(files)}개 중 {used}개에서 {len(bank)}조각 ({len(bank) * args.clip_sec:.0f}초) → {args.out} "
          f"({Path(args.out).stat().st_size / 1e6:.1f} MB)")
    if len(bank) < 40:
        print(f"[참고] 조각이 40개 미만이라 degrade.py 가 실제 홈 잡음 사용 비율을 자동으로 낮춥니다 "
              f"(현재 {len(bank)}개 → 약 {min(1, len(bank) / 40):.0%} 수준). 파일을 더 추가하면 늘어납니다.")


if __name__ == "__main__":
    main()
