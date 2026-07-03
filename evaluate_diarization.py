import argparse
import csv
import json
from pathlib import Path

from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate
from tqdm import tqdm


def load_ground_truth_rttm(path):
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Load source metadata to track tar files and speaker counts
    metadata_path = Path(args.reference_rttm_root).parent / "metadata_sd.csv"
    metadata_by_conv = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_by_conv[row["conv_id"]] = row
    
    # Results tracking per tar
    results_by_tar = {}
    results_path = output_root / "der_results.csv"
    
    # Check if results file exists and load existing
    existing_results = set()
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_results.add(row["conv_id"])
    
    metric = DiarizationErrorRate()
    aggregated = {
        "all": DiarizationErrorRate(),
        "single": DiarizationErrorRate(),
        "two": DiarizationErrorRate(),
        "three": DiarizationErrorRate(),
    }
    
    csv_mode = "w" if not results_path.exists() else "a"
    with open(results_path, csv_mode, encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if csv_mode == "w":
            writer.writerow([
                "conv_id",
                "num_speakers",
                "source_tar",
                "der",
                "missed_speech",
                "false_alarm",
                "speaker_confusion",
            ])

        predicted_paths = sorted(Path(args.predicted_rttm_root).glob("*_predicted.rttm"))
        skipped = 0
        evaluated = 0
        
        for rttm in tqdm(predicted_paths, desc="Evaluating DER", unit="file"):
            conv_id = rttm.stem.replace("_predicted", "")
            
            # Skip if already evaluated
            if conv_id in existing_results:
                skipped += 1
                continue
            
            ref_rttm = Path(args.reference_rttm_root) / f"{conv_id}.rttm"
            if not ref_rttm.exists():
                continue

            hypothesis = load_ground_truth_rttm(rttm)
            reference = load_ground_truth_rttm(ref_rttm)

            der = metric(reference, hypothesis)
            scores = metric.errors(reference, hypothesis)
            
            # Get speaker count from metadata
            num_speakers = metadata_by_conv.get(conv_id, {}).get("num_speakers", "unknown")
            source_tar = metadata_by_conv.get(conv_id, {}).get("source_tar", "unknown")

            writer.writerow([
                conv_id,
                num_speakers,
                source_tar,
                f"{der:.4f}",
                f"{scores['missed']:.4f}",
                f"{scores['false_alarm']:.4f}",
                f"{scores['confusion']:.4f}",
            ])
            
            # Aggregate by speaker count
            speaker_count = str(num_speakers)
            if speaker_count in aggregated:
                aggregated[speaker_count](reference, hypothesis)
            aggregated["all"](reference, hypothesis)
            
            evaluated += 1

    print(f"Evaluated: {evaluated}, Skipped (already done): {skipped}, Total: {len(predicted_paths)}")
    print(f"Wrote DER results to {results_path}")
    
    # Print aggregated results by speaker count
    print("\n=== Aggregated Results ===")
    for key in ["all", "single", "two", "three"]:
        metric_obj = aggregated[key]
        try:
            der = metric_obj.evaluate().metrics['diarization_error_rate']
            print(f"{key}: DER = {der:.4f}")
        except Exception as e:
            print(f"{key}: No results yet ({e})")
    
    # Generate per-tar summary
    tar_summary_path = output_root / "der_results_by_tar.csv"
    with tar_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_tar", "num_items", "avg_der"])
        
        tar_stats = {}
        with results_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tar = row.get("source_tar", "unknown")
                der = float(row["der"])
                if tar not in tar_stats:
                    tar_stats[tar] = []
                tar_stats[tar].append(der)
        
        for tar, ders in sorted(tar_stats.items()):
            avg_der = sum(ders) / len(ders)
            writer.writerow([tar, len(ders), f"{avg_der:.4f}"])
    
    print(f"Wrote per-tar summary to {tar_summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pyannote diarization predictions against ground truth RTTM.")
    parser.add_argument("--predicted-rttm-root", required=True, help="Predicted RTTM files root.")
    parser.add_argument("--reference-rttm-root", required=True, help="Ground truth RTTM files root.")
    parser.add_argument("--output-root", required=True, help="Directory for DER results.")
    args = parser.parse_args()
    main(args)
