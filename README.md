# Vintage2Modern

**Turn historical mono piano recordings into modern-sounding stereo.** ([한국어 안내](docs/README_ko.md))

Vintage2Modern works in two stages:
1. **Restoration network** (learned). A spectrogram U-Net removes hiss, crackle and rumble from the kind of input people actually encounter: 78 rpm and tape transfers, often re-encoded by video platforms. Its output is mono.
2. **Spatial rendering** (no learning). A fixed high-shelf EQ and a decorrelated hall reverb place the restored piano in a stereo space. You can switch it off.

An earlier version learned the stereo image too. In blind listening it did not help, so the stereo part is now physics-based (details in a forthcoming technical report).

- Model weights: Hugging Face [oceantree798/vintage2modern](https://huggingface.co/oceantree798/vintage2modern)

## Results

Measured on 11 held-out recordings published 1920–1925 (public domain in the US): Rachmaninoff, plus Victor and Edison discs, obtained from the UCSB DAHR. They were used for evaluation only, under DAHR's terms of use, and are not redistributed; no DAHR audio is part of the training data of the released weights.

**Blind listening test** (3 listeners, 11 recordings, 15 s excerpts, loudness-matched, 1–5 scale; mean). It was run on an earlier build that differs from the released weights only in its training noise (it also used groove noise harvested from DAHR); the released weights were not listened to separately.

| | Overall quality |
|---|---|
| Original | 2.00 |
| **Vintage2Modern** (listening-test build + spatial rendering) | **4.27** |
| Existing denoiser ([Moliner & Välimäki 2022](https://github.com/eloimoliner/denoising-historical-recordings)) + the same spatial rendering | 4.61 |

With the same spatial rendering, listeners rated the existing denoiser slightly higher on average (+0.33, 95% CI −0.00 to +0.70). The gap came mostly from one of the three listeners. This is a small test, so treat it as indicative.

**Noise measures on the restoration output** of the released weights (median over the 11 recordings):

| | Original | Vintage2Modern v1 | Moliner & Välimäki 2022 |
|---|---|---|---|
| Music-to-noise ratio | 17.1 dB | 28.0 dB | 29.1 dB |
| Hiss above 4 kHz (quiet passages) | −21.7 dB | **−58.9 dB** | −41.7 dB |
| Rumble below 80 Hz (quiet passages) | −32.9 dB | −38.3 dB | **−47.7 dB** |
| Clicks per second | 13.0 | 0.9 | 1.0 |

The model removes much more hiss, but listeners did not prefer it for that. These measures did not predict listening results well.

## Quick start

```bash
pip install -r requirements.txt          # PyTorch with CUDA recommended; ffmpeg for mp3/webm/m4a input
# download vintage2modern_v1.pt from https://huggingface.co/oceantree798/vintage2modern
python infer.py vintage2modern_v1.pt old_recording.mp3 converted.wav
```

Options:
- `--spatial r2` (default): the spatial rendering used in the listening test.
- `--spatial hall`: bass kept mono below 150 Hz and frequency-dependent reverb time. Listeners rated it about the same as `r2` (+0.06, not significant).
- `--spatial none`: restoration only (mono).

Long files are processed in overlapping chunks. It runs on a GPU or a CPU (the CPU is slower).

## Known limitations

- **Piano only.** Other instruments and voices are outside the training data.
- **Bass is boosted.** On 1920s acoustic recordings the listening-test build raised the band below 250 Hz by about 7 dB relative to the input, including in quiet passages. This adds warmth but can also make rumble audible. We think this explains part of the listening gap above; it has not been confirmed.
- **Rumble.** The released weights leave more rumble than the listening-test build (median +2.5 dB, all 11 recordings). Their synthetic groove noise covers only 300 Hz–8 kHz, so low-frequency disc noise is under-represented in training.
- **High frequencies.** The network is trained with a regression loss, so partials above about 8 kHz stay soft. The spatial stage adds a fixed +6 dB shelf above 6 kHz to compensate.
- **The stereo image is synthetic.** It is reverb added to a mono restoration, not recovered information about the original room.
- The model was trained once; seed-to-seed variation has not been measured.
- **44.1 kHz** output.

## Reproducing the released model

The full pipeline is included in this repository:
- the degradation simulator (`degrade.py`) and pair generation (`make_pairs.py`)
- tape calibration (`calibrate_tape.py`) and groove-noise harvesting (`harvest_disc_noise.py`; not used for the released weights)
- pairs for the released weights: `python make_pairs.py --src data/clean --sources sources.csv --out data/pairs8 --no-real-disc`
- training: `python train.py --data data/pairs8 --out runs/v8 --steps 100000 --mono --seed 0`
- spatial rendering (`spatial.py`)
- evaluation (`compare_mono.py`, `eval_real.py`, `measure_real.py`, `diagnose_suppression.py`) and blind listening tests (`make_listening_test.py`, `analyze_listening.py`)

The calibration files are not distributed. Rebuild them from the public sources below with the included tools.

Evaluation recordings, and every other side of the same physical discs, are excluded from all training-side data automatically, by content hash and by disc identifier.

## Contributing

Bug reports are welcome via Issues. Pull requests are not accepted at this time.

## Licenses

- **Code:** Apache License 2.0 (`LICENSE`, attributions in `NOTICE`).
- **Model weights:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Free for research and personal use. Commercial use is not permitted.
- **Converted audio:** you are responsible for the rights to the recordings you convert. Do not redistribute conversions of recordings that are still under copyright.

## Responsible use

Era conversion changes how a historical performance sounds. Please label converted audio as AI-converted, and do not present it as an original recording.

## Data credits

- Training audio: [Open Goldberg Variations](https://opengoldbergvariations.org/) (Kimiko Ishizaka, CC0) and the [Musopen Complete Chopin Collection](https://archive.org/details/musopen-chopin) (CC0).
- Kimiko Ishizaka, *J.S. Bach: The Well-Tempered Clavier, Book 1* ([Open Well-Tempered Clavier](https://welltemperedclavier.org/)). The project site describes the recording as released into the public domain; the Bandcamp release we downloaded was labelled [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), so we give attribution under that license. Used as training data, segmented and degraded.
- Tape calibration: O. Mikkonen, A. Wright, E. Moliner, V. Välimäki, "Neural Modeling of Magnetic Tape Recorders," DAFx 2023. The dataset is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [doi:10.5281/zenodo.8026271](https://doi.org/10.5281/zenodo.8026271). It was used to measure hiss, frequency response and wow/flutter.
- Historical recordings: [UCSB Discography of American Historical Recordings](https://adp.library.ucsb.edu), recordings published before 1926 (public domain in the US). Used for evaluation only, under DAHR's terms of use; not redistributed.
- Evaluation baseline: E. Moliner and V. Välimäki, "A Two-Stage U-Net for High-Fidelity Denoising of Historical Recordings," ICASSP 2022 (MIT license). Used only for comparison; not part of this software.

## Citation

```bibtex
@software{vintage2modern2026,
  title  = {Vintage2Modern},
  author = {myergodic-wq},
  year   = {2026},
  url    = {https://github.com/myergodic-wq/vintage2modern}
}
```
