import argparse
import json
import csv
from pathlib import Path

from pyannote.audio import Pipeline
from tqdm import tqdm


def write_rttm(path, annotation):
    with path.open("w", encoding="utf-8") as f:
        for segment, _, label in annotation.itertracks(yield_label=True):
            f.write(
                "SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n".format(
                    uri=annotation.uri,
                    start=segment.start,
                    duration=segment.duration,
                    speaker=label,
                )
            )


def main(args):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Load source metadata to track tar files
    metadata_path = input_root.parent / "metadata_sd.csv"
    metadata_by_conv = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_by_conv[row["conv_id"]] = row
    
    # Track predictions
    predictions_metadata_path = output_root / "predictions_metadata.csv"
    predictions_by_conv = {}
    if predictions_metadata_path.exists():
        with predictions_metadata_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions_by_conv[row["conv_id"]] = row

    wav_paths = sorted(input_root.glob("*.wav"))
    skipped = 0
    predicted = 0
    
    # Write/append metadata
    csv_mode = "w" if not predictions_metadata_path.exists() else "a"
    with predictions_metadata_path.open(csv_mode, encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if csv_mode == "w":
            writer.writerow(["conv_id", "source_tar", "predicted"])
        
        for wav_path in tqdm(wav_paths, desc="Predicting diarization", unit="file"):
            conv_id = wav_path.stem
            
            # Skip if already predicted
            if conv_id in predictions_by_conv:
                skipped += 1
                continue
            
            prediction = pipeline(str(wav_path))
            rttm_path = output_root / f"{conv_id}_predicted.rttm"
            write_rttm(rttm_path, prediction)
            
            # Get source tar from metadata
            source_tar = metadata_by_conv.get(conv_id, {}).get("source_tar", "unknown")
            writer.writerow([conv_id, source_tar, str(rttm_path)])
            predicted += 1

    print(f"Predicted: {predicted}, Skipped (already done): {skipped}, Total: {len(wav_paths)}")
    print(f"Predictions metadata saved to {predictions_metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pyannote speaker diarization on SD test WAV files.")
    parser.add_argument("--input-root", required=True, help="Directory containing SD test WAV files.")
    parser.add_argument("--output-root", required=True, help="Directory for predicted RTTM files.")
    args = parser.parse_args()
    main(args)
