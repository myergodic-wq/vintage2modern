"""
make_listening_test.py — 블라인드 청취 점검 묶음 만들기 (공개 전 품질 확인용)

실제 옛 녹음에는 정답이 없으므로 MUSHRA 의 '숨은 정답'을 쓸 수 없음 → 방법마다 1~5점 평가(ACR) + 항목별 평가.
곡마다 같은 구간을 잘라 크기를 맞추고, 방법 이름을 숨겨 A/B/C… 로 무작위 배치.

만들어지는 것 (--out 폴더, 통째로 다른 사람에게 보내면 됨):
    clips/ITEM_x.wav      잘라낸 구간 (이름만으로는 방법을 알 수 없음)
    index.html            인터넷 없이 브라우저로 열어 듣고 점수 매기기 → CSV 내려받기
    key.json              어느 파일이 어느 방법인지 (듣는 사람에게 보내지 말 것)
분석: python analyze_listening.py --key listening_test/key.json --csv 결과1.csv 결과2.csv …

사용 예:
    python make_listening_test.py --dir real_tests --systems input v5_best moliner_dsp --sec 15
    (input = 원본, 나머지는 real_tests/out/<파일이름>__<방법>.wav)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from infer import SR, load_audio

EXT = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".webm", ".opus"}
TARGET_RMS_DB = -24.0

QUESTIONS = [
    ("overall", "Overall quality / 전체 음질", "1 = bad, 5 = excellent"),
    ("noise", "Noise (hiss, crackle) / 잡음", "1 = very noisy, 5 = no audible noise"),
    ("artifacts", "Artifacts (watery, metallic, cut-off reverb) / 부작용", "1 = severe, 5 = none"),
    ("clarity", "Clarity and attack of the piano / 피아노의 선명함·타건감", "1 = dull, 5 = clear and crisp"),
    ("space", "Space and stereo naturalness / 공간감·스테레오 자연스러움", "1 = unnatural, 5 = natural hall"),
    ("preserve", "Faithfulness to the original performance (notes, phrasing, touch) / 원래 연주의 보존 (음·프레이징·터치)",
     "1 = changed or invented sounds, 5 = fully faithful"),       # 2026-11-04 추가 (SAMECFM 의 MOS-P 에 해당)
]


def load_stereo(p: Path) -> np.ndarray:
    try:
        y, sr = sf.read(p, dtype="float32", always_2d=True)
    except Exception:
        y, sr = load_audio(str(p)); y = y[:, None]
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR); y = signal.resample_poly(y, SR // g, sr // g, axis=0)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, 1)
    return y[:, :2].astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description="블라인드 청취 점검 묶음")
    ap.add_argument("--dir", default="real_tests")
    ap.add_argument("--systems", nargs="+", required=True, help="input = 원본, 그 외 = out/<파일이름>__<이름>.wav")
    ap.add_argument("--sec", type=float, default=15.0)
    ap.add_argument("--start-frac", type=float, default=0.3, help="곡 길이의 몇 %% 지점부터 자를지")   # argparse 는 % 를 서식으로 봄 (Python 3.14 는 등록 때 바로 오류)
    ap.add_argument("--out", default="listening_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=["wav", "flac"], default="wav", help="flac 이면 파일 크기 절반 정도 (브라우저 재생 가능)")
    args = ap.parse_args()

    root, out = Path(args.dir), Path(args.out)
    od = root / "out"
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in EXT and od.resolve() not in p.resolve().parents)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    key, items = {"systems": args.systems, "clips": {}}, []
    letters = "ABCDEFGH"
    for k, p in enumerate(files, 1):
        srcs = {s: (p if s == "input" else od / f"{p.stem}__{s}.wav") for s in args.systems}
        miss = [str(v) for v in srcs.values() if not v.exists()]
        if miss:
            raise SystemExit(f"[오류] 파일이 없습니다: {miss}")
        aud = {s: load_stereo(v) for s, v in srcs.items()}
        n = min(len(a) for a in aud.values())
        a0 = int(args.start_frac * n); a1 = min(n, a0 + int(args.sec * SR))
        fade = np.minimum(1, np.minimum(np.arange(a1 - a0), np.arange(a1 - a0)[::-1]) / (0.05 * SR))[:, None]
        order = list(rng.permutation(args.systems))
        clips = []
        for j, s in enumerate(order):
            c = aud[s][a0:a1] * fade
            c = c / (np.sqrt(np.mean(c ** 2)) + 1e-12) * 10 ** (TARGET_RMS_DB / 20)    # 크기 맞춤 (큰 소리가 좋게 들리는 착각 방지)
            pk = np.abs(c).max()
            if pk > 0.98:
                c = c / pk * 0.98
            name = f"{k:02d}_{letters[j]}.{args.format}"
            sf.write(out / "clips" / name, c.astype(np.float32), SR, subtype="PCM_16")
            key["clips"][name] = {"system": s, "source": p.name, "start_sec": a0 / SR}
            clips.append(name)
        items.append({"item": f"{k:02d}", "clips": clips})
        print(f"  {k:02d}: {p.name} ({a0 / SR:.0f}~{a1 / SR:.0f}초)")
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "index.html").write_text(page(items), encoding="utf-8")
    print(f"\n완성: {out}/index.html (듣는 사람에게는 clips/ 와 index.html 만 보내세요. key.json 은 보내지 않음)")


def page(items) -> str:
    q = json.dumps(QUESTIONS, ensure_ascii=False)
    it = json.dumps(items)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Listening Test</title>
<style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--acc:#2a6fd4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1b1b1a;--fg:#ecebe6;--mut:#a3a29c;--card:#252524;--line:#3a3936;--acc:#7aa7f0}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:24px 16px 80px}}
h1{{font-size:22px;margin:0 0 6px}} p{{color:var(--mut);margin:4px 0 16px}}
.item{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}}
.clip{{border-top:1px solid var(--line);padding:12px 0}} .clip:first-of-type{{border-top:0}}
.clip h3{{margin:0 0 8px;font-size:15px}} audio{{width:100%}}
.q{{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:6px 0}} .q span{{flex:1 1 260px;font-size:13px}}
.q label{{font-size:13px;color:var(--mut)}} button{{font:inherit;padding:10px 16px;border-radius:8px;border:0;background:var(--acc);color:#fff;cursor:pointer}}
input[type=text]{{font:inherit;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}}
</style></head><body><main>
<h1>Listening test / 청취 평가</h1>
<p>Headphones recommended. Each item has the same passage processed in different ways (order is random). Rate every clip 1–5.<br>
헤드폰을 권합니다. 각 항목은 같은 구간을 서로 다른 방법으로 처리한 것입니다(순서 무작위). 모든 클립을 1~5점으로 평가하세요.</p>
<p>Your name or ID / 이름 또는 번호: <input type="text" id="who"></p>
<div id="items"></div>
<button onclick="dl()">Download results (CSV) / 결과 내려받기</button>
<p id="msg"></p>
</main><script>
const Q={q}, ITEMS={it};
const box=document.getElementById('items');
ITEMS.forEach(o=>{{const d=document.createElement('div');d.className='item';
 d.innerHTML='<h2 style="font-size:17px;margin:0 0 8px">Item '+o.item+'</h2>'+o.clips.map(c=>{{
  const L=c.split('_')[1][0];
  return '<div class="clip"><h3>'+L+'</h3><audio controls preload="none" src="clips/'+c+'"></audio>'+
   Q.map(([k,t,h])=>'<div class="q"><span>'+t+' <small>('+h+')</small></span>'+[1,2,3,4,5].map(v=>
   '<label><input type="radio" name="'+c+'|'+k+'" value="'+v+'"> '+v+'</label>').join(' ')+'</div>').join('')+'</div>';
 }}).join('');box.appendChild(d);}});
function dl(){{const who=document.getElementById('who').value.trim()||'anon';const rows=[['listener','clip','question','score']];let miss=0;
 ITEMS.forEach(o=>o.clips.forEach(c=>Q.forEach(([k])=>{{const r=document.querySelector('input[name="'+c+'|'+k+'"]:checked');
  if(r)rows.push([who,c,k,r.value]);else miss++;}})));
 const csv=rows.map(r=>r.join(',')).join('\\n');const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='listening_'+who+'.csv';a.click();
 document.getElementById('msg').textContent=miss?('Missing ratings / 빠진 평가: '+miss):'Saved / 저장됨';}}
</script></body></html>"""


if __name__ == "__main__":
    main()
