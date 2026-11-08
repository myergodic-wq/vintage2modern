---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: audio-to-audio
tags:
  - audio-restoration
  - historical-recordings
  - denoising
  - piano
---

# Vintage2Modern v1

Vintage2Modern v1 restores historical mono piano recordings. The network removes hiss, crackle and rumble and outputs mono. The accompanying code then adds a stereo image with a fixed, physics-based spatial stage (EQ and decorrelated hall reverb), which is not part of these weights.

- **Code:** https://github.com/myergodic-wq/vintage2modern (Apache 2.0)
- **Weights license:** CC BY-NC 4.0 (research and personal use; commercial use is not permitted).

## Model

| | |
|---|---|
| Architecture | Spectrogram U-Net, 4 levels, 32 base channels, **14.2M parameters** |
| Input | Mono audio, 44.1 kHz (any length, processed in overlapping 10 s chunks) |
| Output | Mono audio, 44.1 kHz (input + learned residual in the compressed STFT domain). `infer.py` then applies the spatial stage (`--spatial r2` by default) |
| Representation | Compressed complex STFT (n_fft 2048, hop 512, magnitude^0.3) |
| Loss | Multi-resolution STFT (FFT sizes 512–4096) between the output and the mid channel ((L+R)/2) of the modern target |
| Training | 100k steps (checkpoint at 90k, lowest validation loss), AdamW 2e-4, cosine decay, batch 8 × 6 s, bf16, seed 0, one RTX 4090, about 6 h |

## Training data

The model was trained on 13,996 synthetic pairs from 161 openly licensed piano recordings (CC0 or CC BY 4.0) (about 9.7 h):
- Open Goldberg Variations (CC0)
- Musopen Chopin (CC0)
- Open Well-Tempered Clavier Book 1 (Kimiko Ishizaka; attributed under CC BY 4.0, see Credits)

Each 10 s segment was degraded 4 times by a recording-chain simulator. Presets: 1950s tape (50%), 1925 shellac (15%), pre-1925 acoustic (15%), mild (20%). The simulator is calibrated with tape measurements from the Akai 4000D dataset (CC BY 4.0). 78 rpm surface noise is fully synthetic (rotation-rate swish, pops, scratches, crackle, band-limited groove noise); no recorded disc noise is used.

## Evaluation

The evaluation set has 11 held-out recordings published 1920–1925 (public domain in the US), obtained from the UCSB DAHR: Victor 74645, 1008, 816, 943 and 972, and Edison 82187, 82197 and 82202. They are used for evaluation only, under DAHR's terms of use, and are not redistributed; neither these recordings nor the other sides of the same discs were used anywhere in training.

**Blind listening** (3 listeners × 11 recordings, 15 s excerpts, loudness-matched, 1–5 scale, mean). All processed systems use the same spatial stage. The test used an earlier build that differs from these weights only in its training noise (it also used groove noise harvested from DAHR); these weights were not listened to separately.

| | Overall | Faithfulness to the performance |
|---|---|---|
| Original | 2.00 | 2.03 |
| **Listening-test build** | **4.27** | **4.21** |
| Moliner & Välimäki 2022 denoiser | 4.61 | 4.61 |
| Build − Moliner (paired, 95% CI by recording) | −0.33 [−0.70, +0.00] | −0.39 [−0.79, −0.06] |

Most of the gap came from one listener (the other two rated the systems within 0.2 points).

**Noise measures on the mono restoration** of these weights, mean (median):

| | Original | v1 |
|---|---|---|
| Music-to-noise ratio | 17.9 (17.1) dB | 27.7 (28.0) dB |
| Hiss above 4 kHz, quiet frames | −21.4 (−21.7) dB | −59.7 (−58.9) dB |
| Rumble below 80 Hz, quiet frames | −34.2 (−32.9) dB | −38.1 (−38.3) dB |
| Clicks per second | 12.6 (13.0) | 1.7 (0.9) |

Compared with the listening-test build on the same recordings, these weights have a higher music-to-noise ratio (10 of 11 recordings, p = 0.007) and leave more rumble (11 of 11, median +2.5 dB, p = 0.001); clicks and hiss do not differ significantly.

**Synthetic validation** (300 held-out pairs, mid channel): log-spectral distance falls from 0.762 (untouched input) to **0.249**.

## Limitations

- The model was trained on piano only.
- **Bass:** on 1920s acoustic recordings the listening-test build raised the band below 250 Hz by about 7 dB relative to the input, including rumble in quiet passages. This may explain part of the listening gap; it has not been confirmed and has not been measured for these weights.
- **Rumble:** the synthetic groove noise covers 300 Hz–8 kHz only, so low-frequency disc noise is under-represented in training; these weights leave more rumble than the listening-test build.
- Partials above about 8 kHz stay soft (regression averaging). The spatial stage adds a fixed +6 dB shelf above 6 kHz.
- In listening, the large hiss reduction did not translate into a preference over an existing denoiser. Noise measures did not predict listening results.
- The model was trained once; seed-to-seed variation has not been measured.
- The listening test is small (3 listeners) and used 15 s excerpts.

## Intended use and misuse

- **Intended:** listening to and studying historical piano recordings, research, and personal use.
- **Not intended:** presenting converted audio as an original recording, or distributing conversions of recordings you do not have rights to.

Please label converted audio as AI-converted.

## Credits

Kimiko Ishizaka, *The Well-Tempered Clavier, Book 1* (the project site describes it as public domain; the Bandcamp release was labelled CC BY 4.0, under which we attribute it). O. Mikkonen, A. Wright, E. Moliner, V. Välimäki, Akai 4000D tape dataset (CC BY 4.0, doi:10.5281/zenodo.8026271). UCSB Discography of American Historical Recordings (evaluation only). Evaluation baseline: E. Moliner and V. Välimäki, ICASSP 2022 (MIT).
