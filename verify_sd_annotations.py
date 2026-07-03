import argparse
import csv
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def read_metadata(metadata_path):
    items = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items[row["conv_id"]] = row
    return items


def read_rttm(path):
    segments = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            segments.append(
                {
                    "start": float(parts[3]),
                    "duration": float(parts[4]),
                    "speaker_id": parts[7],
                }
            )
    return segments


def read_stm(path):
    segments = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=5)
            if len(parts) < 6:
                continue
            segments.append(
                {
                    "speaker_id": parts[2],
                    "start": float(parts[3]),
                    "end": float(parts[4]),
                    "text": parts[5],
                }
            )
    return segments


def validate_item(conv_id, waves, rttm_path, stm_path, metadata):
    errors = []
    if not rttm_path.exists():
        errors.append("Missing RTTM")
    if not stm_path.exists():
        errors.append("Missing STM")
    if not waves.exists():
        errors.append("Missing WAV")

    if not waves.exists():
        return errors

    with sf.SoundFile(waves) as f:
        if f.samplerate != 22050:
            errors.append(f"Sampling rate is {f.samplerate}, expected 22050")
        if f.channels != 1:
            errors.append(f"Channel count is {f.channels}, expected mono")
        duration = len(f) / f.samplerate

    if conv_id not in metadata:
        errors.append("Missing metadata entry")
        return errors

    expected = metadata[conv_id]
    speaker_ids = set()
    if rttm_path.exists():
        segments = read_rttm(rttm_path)
        for seg in segments:
            if seg["start"] < 0 or seg["duration"] <= 0:
                errors.append("Invalid RTTM timestamps")
            if seg["start"] + seg["duration"] > duration + 0.01:
                errors.append("RTTM segment outside audio duration")

        if len(segments) == 0:
            errors.append("No RTTM segments")

        speaker_ids = {seg["speaker_id"] for seg in segments}
        if expected["speaker_ids"]:
            expected_ids = set(expected["speaker_ids"].split(";"))
            if speaker_ids != expected_ids:
                errors.append(
                    f"RTTM speaker IDs mismatch {speaker_ids} != {expected_ids}"
                )

    if stm_path.exists():
        stm_segments = read_stm(stm_path)
        if len(stm_segments) == 0:
            errors.append("No STM segments")
        for seg in stm_segments:
            if seg["end"] <= seg["start"]:
                errors.append("Invalid STM segment times")
            if seg["end"] > duration + 0.01:
                errors.append("STM segment outside audio duration")
            if speaker_ids and seg["speaker_id"] not in speaker_ids:
                errors.append(
                    f"STM speaker ID {seg['speaker_id']} not in RTTM speakers"
                )

    expected_count = None
    if expected["num_speakers"]:
        try:
            expected_count = int(expected["num_speakers"])
            if len(speaker_ids) != expected_count:
                errors.append(
                    f"Speaker count {len(speaker_ids)} != expected {expected_count}"
                )
        except ValueError:
            pass

    if expected["overlap_percentage"]:
        try:
            overlap_pct = float(expected["overlap_percentage"])
            if expected_count == 1 and overlap_pct != 0.0:
                errors.append("Single-speaker overlap must be 0")
            if expected_count and expected_count > 1 and not (10.0 <= overlap_pct <= 15.0):
                errors.append(
                    f"Overlap percentage {overlap_pct} out of bounds"
                )
        except ValueError:
            pass

    return errors


def main(args):
    metadata = read_metadata(Path(args.metadata))
    report = []
    root = Path(args.sd_root)
    for conv_id in tqdm(sorted(metadata.keys()), desc="Verifying items", unit="item"):
        wav_path = root / "conversations" / f"{conv_id}.wav"
        rttm_path = root / "rttm" / f"{conv_id}.rttm"
        stm_path = root / "stm" / f"{conv_id}.stm"
        errors = validate_item(conv_id, wav_path, rttm_path, stm_path, metadata)
        report.append((conv_id, errors))

    report_path = root / "annotation_verification_report.txt"
    with report_path.open("w", encoding="utf-8") as out:
        for conv_id, errors in report:
            if errors:
                out.write(f"{conv_id}: {'; '.join(errors)}\n")
        out.write(f"Checked {len(report)} items\n")
        out.write(f"Failed {sum(1 for _, errs in report if errs)} items\n")

    print(f"Verification complete. Report written to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify SD RTTM and STM annotations.")
    parser.add_argument("--sd-root", required=True, help="Root directory for SD data.")
    parser.add_argument("--metadata", required=True, help="metadata_sd.csv path.")
    args = parser.parse_args()
    main(args)
