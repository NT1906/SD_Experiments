#!/usr/bin/env python3
"""
Build the CoSSHIv2 synthetic Speaker Diarization dataset (CoSSHIv2-034).

Fixes vs the original script, based on your actual data:
  1. Your WAV files are named "<LLM>_<TTS>_<SENT_ID>.wav" (3 parts, e.g.
     CSN_BRK_0002.wav) for end-to-end TTS models (BRK/XT2/YTS) -- there is no
     vocoder code in the filename. The old parser required 4 parts and threw
     every single file away (0 entries found). This version accepts both the
     3-part end-to-end form (vocoder is inferred as "BLT") and the 4-part
     vocoder form "<LLM>_<TTS>_<VOC>_<SENT_ID>.wav".
  2. Your sentence.txt lines look like "CSN_0001 | The morning sun ..." --
     i.e. the key is "<LLM>_<SENT_ID>", separated by " | ". The old parser
     split on a single space and keyed sentences by sent_id alone, so every
     lookup missed. This version parses on " | " and keys by "<LLM>_<SENT_ID>".

Point --source-root at the PARENT folder that contains all 11 of your WAV
folders -- the scanner recurses into subfolders automatically, so you do not
need to merge them.

STEP-BY-STEP USAGE
-------------------
1) Install deps (already in requirements.txt):
   py -m pip install -r requirements.txt

2) Build the dataset (point at the parent of your 11 folders):
   py build_sd_dataset.py ^
       --source-root "C:/data/CoSSHI_audio" ^
       --output-root "SSD/EXPERIMENTS/SD" ^
       --sentences-file sentence.txt

   This writes:
     SSD/EXPERIMENTS/SD/conversations/conv_00001.wav ...
     SSD/EXPERIMENTS/SD/rttm/conv_00001.rttm ...
     SSD/EXPERIMENTS/SD/stm/conv_00001.stm ...
     SSD/EXPERIMENTS/SD/metadata_sd.csv

   It is resumable: if interrupted, re-run the exact same command and it
   picks up where it left off (tracked in SSD/EXPERIMENTS/SD/progress.json).

3) Verify annotations:
   py verify_sd_annotations.py --sd-root SSD/EXPERIMENTS/SD --metadata SSD/EXPERIMENTS/SD/metadata_sd.csv

4) Then predict_diarization.py -> evaluate_diarization.py -> aggregate_results.py
   as documented in README.md.
"""
import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import resample_poly
from tqdm import tqdm

# TTS short codes that are end-to-end (no separate vocoder stage).
# Per the CoSSHIv2 naming table: end-to-end models use vocoder code BLT.
END_TO_END_TTS = {"BRK", "XT2", "YTS"}


# --------------------------------------------------------------------------
# Progress tracking (resumability)
# --------------------------------------------------------------------------
def load_progress(progress_file):
    if progress_file.exists():
        with progress_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_sources": {}}


def save_progress(progress_file, progress):
    with progress_file.open("w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def get_source_key(source_root):
    return Path(source_root).name


# --------------------------------------------------------------------------
# Filename parsing -- FIXED to support both naming forms actually on disk
# --------------------------------------------------------------------------
def parse_audio_metadata(path):
    """
    Accepts:
      <LLM>_<TTS>_<SENT_ID>.wav            e.g. CSN_BRK_0002.wav   (end-to-end)
      <LLM>_<TTS>_<VOC>_<SENT_ID>.wav      e.g. GPT_VTS_HFG_0001.wav
    Returns None for anything that doesn't match either shape.
    """
    if isinstance(path, (bytes, bytearray)):
        path = path.decode("utf-8", errors="ignore")
    stem = Path(str(path)).stem
    parts = stem.split("_")

    if len(parts) == 3:
        llm, tts, sent_id = parts
        voc = "BLT"  # end-to-end TTS -> built-in vocoder code
    elif len(parts) >= 4:
        llm, tts, voc = parts[0], parts[1], parts[2]
        sent_id = parts[3]
    else:
        return None

    if not sent_id.isdigit():
        return None

    return {
        "llm": llm,
        "tts": tts,
        "voc": voc,
        "sent_id": sent_id.zfill(4),
        "file_name": Path(str(path)).name,
    }


def normalize_audio(audio, sr, target_sr=22050):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr == target_sr:
        return audio, sr
    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")
    factor = math.gcd(int(sr), int(target_sr))
    up = target_sr // factor
    down = sr // factor
    audio = resample_poly(audio, up, down).astype(np.float32)
    return audio, target_sr


def read_audio_file(audio_path):
    try:
        audio, sr = sf.read(audio_path)
        return audio, sr
    except Exception as primary_error:
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            return audio, sr
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to read audio file {audio_path} with soundfile and librosa. "
                f"soundfile error: {primary_error}; librosa error: {fallback_error}"
            ) from fallback_error


def scan_audio_files(source_root):
    """Recurses into every subfolder under source_root -- covers all 11 folders
    in one call as long as source_root is their common parent."""
    source_path = Path(source_root)
    if not source_path.exists():
        raise RuntimeError(f"Source path {source_root} does not exist.")
    if source_path.is_file():
        raise RuntimeError("Source root must be a directory containing WAV files.")

    audio_paths = list(source_path.rglob("*.wav"))
    entries = []
    skipped = 0
    for path in audio_paths:
        metadata = parse_audio_metadata(path.stem)
        if not metadata:
            skipped += 1
            continue
        metadata.update({"path": path, "source": "filesystem"})
        entries.append(metadata)

    print(f"Scanned {len(audio_paths)} WAV files under {source_root}: "
          f"{len(entries)} parsed OK, {skipped} skipped (unrecognised name).")
    return entries


def load_audio_from_entry(entry):
    audio, sr = read_audio_file(entry["path"])
    return normalize_audio(audio, sr)


# --------------------------------------------------------------------------
# Sentence lookup -- FIXED to match "<LLM>_<SENT_ID> | text" format
# --------------------------------------------------------------------------
def load_sentences(sentences_file):
    """
    sentence.txt lines look like:
        CSN_0001 | The morning sun warmed the quiet garden path.
        DSK_0002 | The horse swam through a forest clumsily.
    Key used for lookup is "<LLM>_<SENT_ID>" (4-digit, zero-padded).
    """
    sentences = {}
    if not sentences_file or not sentences_file.exists():
        return sentences
    with sentences_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            key, text = line.split("|", 1)
            key = key.strip()
            if "_" not in key:
                continue
            llm, sent_id = key.rsplit("_", 1)
            if not sent_id.isdigit():
                continue
            norm_key = f"{llm}_{sent_id.zfill(4)}"
            sentences[norm_key] = text.strip()
    print(f"Loaded {len(sentences)} sentences from {sentences_file}")
    return sentences


def lookup_sentence(sentences, entry):
    key = f"{entry['llm']}_{entry['sent_id']}"
    return sentences.get(key, f"Sentence {entry['sent_id']}")


# --------------------------------------------------------------------------
# Item selection
# --------------------------------------------------------------------------
def choose_items(entries, count, distinct_model=False):
    """distinct_model=True enforces: each speaker uses a different TTS/vocoder
    model, per the CoSSHIv2-034 spec (multi-speaker items only)."""
    if not distinct_model:
        return random.sample(entries, min(count, len(entries)))

    by_model = defaultdict(list)
    for entry in entries:
        by_model[(entry["tts"], entry["voc"])].append(entry)

    unique_keys = list(by_model.keys())
    if count > len(unique_keys):
        raise ValueError(
            f"Not enough distinct TTS/vocoder combinations for {count} speakers. "
            f"Found {len(unique_keys)} unique combinations."
        )
    selected_keys = random.sample(unique_keys, count)
    return [random.choice(by_model[k]) for k in selected_keys]


def mix_overlap(previous_audio, current_audio, overlap_samples, gap_samples):
    """Places current_audio after previous_audio + gap_samples of silence,
    then pulls it earlier by overlap_samples so the turn boundary overlaps."""
    overlap_samples = max(0, min(overlap_samples, len(current_audio) - 1, len(previous_audio) - 1))
    current_start = max(len(previous_audio) + gap_samples - overlap_samples, 0)
    total_length = max(len(previous_audio), current_start + len(current_audio))

    mixed = np.zeros(total_length, dtype=np.float32)
    mixed[: len(previous_audio)] = previous_audio
    mixed[current_start: current_start + len(current_audio)] += current_audio

    if overlap_samples > 0:
        o0 = max(current_start, len(previous_audio) - overlap_samples)
        o1 = min(len(previous_audio), current_start + len(current_audio))
        if o1 > o0:
            mixed[o0:o1] = np.clip(mixed[o0:o1], -1.0, 1.0)

    return mixed, current_start / 22050.0


# --------------------------------------------------------------------------
# RTTM / STM writers
# --------------------------------------------------------------------------
def write_rttm(path, conv_id, segments):
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(
                "SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n".format(
                    uri=conv_id, start=seg["start"], duration=seg["duration"], speaker=seg["speaker_id"]
                )
            )


def write_stm(path, conv_id, segments):
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            text = seg.get("text", "<no_text>").replace("\n", " ")
            f.write(
                "{uri} 1 {speaker} {start:.3f} {end:.3f} {text}\n".format(
                    uri=conv_id, speaker=seg["speaker_id"], start=seg["start"],
                    end=seg["start"] + seg["duration"], text=text,
                )
            )


def ensure_directories(root):
    root.joinpath("conversations").mkdir(parents=True, exist_ok=True)
    root.joinpath("rttm").mkdir(parents=True, exist_ok=True)
    root.joinpath("stm").mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Main build routine
# --------------------------------------------------------------------------
def build_dataset(
    source_root,
    output_root,
    sentences_file=None,
    progress_file=None,
    num_single=5000,
    num_two=3000,
    num_three=2000,
    single_duration=(6.0, 10.0),
    two_duration=(18.0, 30.0),
    three_duration=(30.0, 45.0),
    gap_range=(0.3, 0.8),
    overlap_range=(0.10, 0.15),
):
    output_root = Path(output_root)
    progress_file = Path(progress_file) if progress_file else output_root / "progress.json"
    sentences = load_sentences(Path(sentences_file) if sentences_file else None)

    progress = load_progress(progress_file)
    source_key = get_source_key(source_root)
    if source_key in progress.get("completed_sources", {}):
        print(f"Source '{source_key}' already completed. Skipping. "
              f"Delete its entry in {progress_file} to force a rebuild.")
        return

    print(f"Scanning WAV files under {source_root} (recurses into all subfolders)...")
    entries = scan_audio_files(source_root)
    if len(entries) < 100:
        raise RuntimeError(
            "Not enough source audio entries found for dataset construction. "
            "Check --source-root points at the parent of your 11 WAV folders."
        )

    ensure_directories(output_root)
    metadata_path = output_root / "metadata_sd.csv"

    csv_exists = metadata_path.exists()
    if csv_exists:
        with metadata_path.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        conv_id = int(rows[-1][0].split("_")[-1]) + 1 if len(rows) > 1 else 1
        csv_mode = "a"
        print(f"Resuming from conv_id {conv_id}. Appending to {metadata_path}...")
    else:
        conv_id = 1
        csv_mode = "w"
        print(f"Creating new metadata file at {metadata_path}...")

    with metadata_path.open(csv_mode, encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow([
                "conv_id", "duration", "num_speakers", "speaker_ids",
                "tts_models", "vocoder", "overlap_percentage", "source_tar",
            ])

        for category, count, duration_range, multi_speaker, speaker_count in [
            ("single", num_single, single_duration, False, 1),
            ("two", num_two, two_duration, True, 2),
            ("three", num_three, three_duration, True, 3),
        ]:
            print(f"Building {count} {category}-speaker items...")
            for _ in tqdm(range(count), desc=f"Building {category}", unit="item"):
                best_item = None
                attempts = 0
                while attempts < 20:
                    attempts += 1
                    try:
                        sources = choose_items(entries, count=speaker_count, distinct_model=multi_speaker)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"Cannot build {category}-speaker items: {exc}. "
                            "You need audio from more distinct TTS/vocoder combinations "
                            "(e.g. make sure all 11 folders are under --source-root)."
                        ) from exc

                    speakers, segments = [], []
                    output_audio = np.zeros((0,), dtype=np.float32)
                    overlap_pct = 0.0
                    failed_candidate = False

                    for speaker_index, entry in enumerate(sources, start=1):
                        try:
                            audio, sr = load_audio_from_entry(entry)
                        except Exception as exc:
                            print(f"Warning: skipping unreadable source {entry['file_name']}: {exc}")
                            failed_candidate = True
                            break

                        speaker_id = f"{entry['tts']}_SPK"
                        if speaker_index == 1:
                            output_audio = audio
                            start_time = 0.0
                        else:
                            gap = random.uniform(*gap_range)
                            overlap = random.uniform(*overlap_range)
                            overlap_samples = int(overlap * sr)
                            gap_samples = int(gap * sr)
                            output_audio, start_time = mix_overlap(
                                output_audio, audio, overlap_samples, gap_samples
                            )
                            overlap_pct = overlap * 100.0

                        duration = len(audio) / sr
                        segments.append({
                            "speaker_id": speaker_id,
                            "start": start_time,
                            "duration": duration,
                            "text": lookup_sentence(sentences, entry),
                        })
                        speakers.append(speaker_id)

                    if failed_candidate:
                        continue

                    item_duration = len(output_audio) / 22050.0
                    if duration_range[0] <= item_duration <= duration_range[1]:
                        best_item = (sources, output_audio, segments, speakers, overlap_pct, item_duration)
                        break
                    if attempts == 20:
                        best_item = (sources, output_audio, segments, speakers, overlap_pct, item_duration)

                if best_item is None:
                    continue

                sources, output_audio, segments, speakers, overlap_pct, item_duration = best_item
                conv_name = f"conv_{conv_id:05d}"
                sf.write(output_root / "conversations" / f"{conv_name}.wav", output_audio, 22050)
                write_rttm(output_root / "rttm" / f"{conv_name}.rttm", conv_name, segments)
                write_stm(output_root / "stm" / f"{conv_name}.stm", conv_name, segments)

                writer.writerow([
                    conv_name,
                    f"{item_duration:.3f}",
                    len(sources),
                    ";".join(speakers),
                    ";".join(sorted({e['tts'] for e in sources})),
                    ";".join(sorted({e['voc'] for e in sources})),
                    f"{overlap_pct:.2f}",
                    "local_filesystem",
                ])
                conv_id += 1

    print(f"Generated dataset metadata at {metadata_path}")

    progress["completed_sources"][source_key] = {
        "completed_at": str(Path(source_root).stat().st_mtime),
        "total_items": conv_id - 1,
    }
    save_progress(progress_file, progress)
    print(f"Progress saved. Completed sources: {list(progress['completed_sources'].keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the CoSSHIv2 synthetic Speaker Diarization dataset (CoSSHIv2-034)."
    )
    parser.add_argument("--source-root", required=True,
                         help="Parent directory containing all 11 WAV folders (scanned recursively).")
    parser.add_argument("--output-root", required=True,
                         help="Output root for SD data (conversations/rttm/stm/metadata_sd.csv).")
    parser.add_argument("--sentences-file", required=True,
                         help="Path to sentence.txt (lines: '<LLM>_<SENT_ID> | text').")
    parser.add_argument("--progress-file", help="Optional progress tracking file (default: output-root/progress.json).")
    parser.add_argument("--num-single", type=int, default=5000)
    parser.add_argument("--num-two", type=int, default=3000)
    parser.add_argument("--num-three", type=int, default=2000)
    args = parser.parse_args()

    build_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        sentences_file=args.sentences_file,
        progress_file=args.progress_file,
        num_single=args.num_single,
        num_two=args.num_two,
        num_three=args.num_three,
    )
