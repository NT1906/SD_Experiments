# Speaker Diarization Workflow

Scripts to build, verify, predict, and evaluate the synthetic Speaker
Diarization dataset for , matching your real data layout:

- Multiple WAV folders (any number, any names, any nesting) sharing one
  common parent folder.
- Filenames like `CSN_BRK_0002.wav` (end-to-end TTS, no vocoder code) or
  `GPT_VTS_HFG_0001.wav` (TTS + vocoder).
- One shared `sentence.txt` covering all LLMs, lines formatted as
  `<LLM>_<SENT_ID> | sentence text`, e.g. `CSN_0001 | The morning sun ...`.

## Files

| File | Purpose |
|---|---|
| `build_sd_dataset.py` | Build the ~10,000-item SD dataset (5,000 single / 3,000 two / 2,000 three-speaker) from your WAV folders + sentence.txt. Resumable. |
| `verify_sd_annotations.py` | Checks every WAV/RTTM/STM triple is correct and consistent with `metadata_sd.csv`. |
| `predict_diarization.py` | Runs `pyannote/speaker-diarization-3.1` on the built conversations. Resumable. |
| `evaluate_diarization.py` | Computes DER against ground truth, per-item and per-tar. Resumable. |
| `aggregate_results.py` | Rolls everything up into a final summary report. |
| `requirements.txt` | Python dependencies. |

## Step 0 — Install dependencies

```powershell
py -m pip install -r requirements.txt
```

If `py -m pip` isn't found, use your Python interpreter path directly, e.g.:
```powershell
C:\Users\<you>\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pip install -r requirements.txt
```

## Step 1 — Arrange your data (no merging needed)

Put all your WAV folders under one parent directory, and keep your
`sentence.txt` anywhere you like — you just need its path.

```
C:/data/CoSSHI_audio/
├── DESKTOP1_BRK/           <- folder 1, any name
│   ├── CSN_BRK_0002.wav
│   └── ...
├── LAPTOP1_VTS/            <- folder 2
│   ├── GPT_VTS_HFG_0001.wav
│   └── ...
├── ...                     <- folders 3–11
└── (sentence.txt can live outside this tree, e.g. C:/data/sentence.txt)
```

The scanner recurses into every subfolder automatically (`rglob("*.wav")`),
so it doesn't matter how the 11 folders are organized underneath the
parent — one `--source-root` pointing at the common parent covers all of
them in a single run.

## Step 2 — Build the dataset

```powershell
py build_sd_dataset.py --source-root "C:/data/CoSSHI_audio" --output-root "SSD/EXPERIMENTS/SD" --sentences-file "C:/data/sentence.txt"
```

**Console output looks like:**
```
Loaded 36464 sentences from C:/data/sentence.txt
Scanning WAV files under C:/data/CoSSHI_audio (recurses into all subfolders)...
Scanned 864000 WAV files under C:/data/CoSSHI_audio: 864000 parsed OK, 0 skipped (unrecognised name).
Creating new metadata file at SSD/EXPERIMENTS/SD/metadata_sd.csv...
Building 5000 single-speaker items...
Building single: 100%|██████████| 5000/5000 [xx:xx<00:00, xx.xxitem/s]
Building 3000 two-speaker items...
Building two: 100%|██████████| 3000/3000 [xx:xx<00:00, xx.xxitem/s]
Building 2000 three-speaker items...
Building three: 100%|██████████| 2000/2000 [xx:xx<00:00, xx.xxitem/s]
Generated dataset metadata at SSD/EXPERIMENTS/SD/metadata_sd.csv
```

If `... skipped (unrecognised name)` is not 0, some files don't match either
naming pattern — worth a look before proceeding.

**Interrupted?** Just re-run the exact same command. Progress is tracked in
`SSD/EXPERIMENTS/SD/progress.json` and it will pick up from where it left off
without duplicating `conv_id`s.

### What you get

```
SSD/EXPERIMENTS/SD/
├── conversations/
│   ├── conv_00001.wav          <- 10,000 built conversation WAVs (22050 Hz mono)
│   └── ...
├── rttm/
│   ├── conv_00001.rttm         <- ground-truth speaker turn boundaries
│   └── ...
├── stm/
│   ├── conv_00001.stm          <- boundaries + transcript text
│   └── ...
├── metadata_sd.csv             <- master index, one row per conversation
└── progress.json
```

`metadata_sd.csv` columns: `conv_id | duration | num_speakers | speaker_ids |
tts_models | vocoder | overlap_percentage | source_tar`.

## Step 3 — Verify annotations

```powershell
py verify_sd_annotations.py --sd-root SSD/EXPERIMENTS/SD --metadata SSD/EXPERIMENTS/SD/metadata_sd.csv
```

**Output:** `SSD/EXPERIMENTS/SD/annotation_verification_report.txt`, ending with:
```
Checked 10000 items
Failed 0 items
```
Any failing `conv_id` is listed above that line with the specific issue
(missing file, bad timestamp, speaker-count mismatch, overlap out of the
10–15% bound, etc.).

## Step 4 — Predict diarization

```powershell
py predict_diarization.py --input-root SSD/EXPERIMENTS/SD/conversations --output-root SSD/EXPERIMENTS/SD/predicted_rttm
```

Produces `conv_XXXXX_predicted.rttm` for each conversation plus a
`predictions_metadata.csv`. Resumable — skips already-predicted items on
re-run.

## Step 5 — Evaluate DER

```powershell
py evaluate_diarization.py --predicted-rttm-root SSD/EXPERIMENTS/SD/predicted_rttm --reference-rttm-root SSD/EXPERIMENTS/SD/rttm --output-root SSD/EXPERIMENTS/SD/results
```

Produces `SSD/EXPERIMENTS/SD/results/der_results.csv` (per-item DER, missed
speech, false alarm, speaker confusion) and prints aggregated DER for
single / two / three-speaker / all combined, per CoSSHIv2-050.

## Step 6 — Aggregate final results

```powershell
py aggregate_results.py --results-csv SSD/EXPERIMENTS/SD/results/der_results.csv --output-dir SSD/EXPERIMENTS/SD/results/aggregated
```

**Final deliverable:** `SSD/EXPERIMENTS/SD/results/aggregated/SUMMARY.txt` —
overall DER plus the per-speaker-count breakdown you need for the paper.

```powershell
cat SSD/EXPERIMENTS/SD/results/aggregated/SUMMARY.txt
```

## Notes on the naming/parsing fixes in this version

The build script was updated to match your actual data (earlier versions
would silently find 0 usable files or crash on multi-speaker items):

1. **Filenames with 3 parts** (`<LLM>_<TTS>_<SENT_ID>.wav`, used by
   end-to-end TTS models like BRK/XT2/YTS) are now parsed correctly, with
   vocoder inferred as `BLT`. 4-part filenames
   (`<LLM>_<TTS>_<VOC>_<SENT_ID>.wav`) still work as before.
2. **`sentence.txt` parsing** now matches your real format —
   `<LLM>_<SENT_ID> | sentence text` — and sentences are looked up by
   `<LLM>_<SENT_ID>` (not by sentence ID alone), so every LLM's own sentence
   set resolves correctly even though they all share one file.
3. **Overlap mixing** (`mix_overlap`) had a buffer-sizing bug that crashed on
   every 2- or 3-speaker item; this is fixed and was verified end-to-end on
   a synthetic test build (0 verification failures, overlap consistently
   landing in the required 10–15% band).

## Progress tracking & resumability

All of `build_sd_dataset.py`, `predict_diarization.py`, and
`evaluate_diarization.py` are safe to interrupt and re-run:

- **Build**: tracked in `SSD/EXPERIMENTS/SD/progress.json` per source root.
- **Predict**: skips already-predicted conversations, tracked in
  `predicted_rttm/predictions_metadata.csv`.
- **Evaluate**: skips already-evaluated conversations, tracked in
  `results/der_results.csv`.
