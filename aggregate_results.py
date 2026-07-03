#!/usr/bin/env python3
"""
Aggregate and summarize results across all tar file subsets.
Generates combined statistics and per-tar breakdowns.
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict


def main(args):
    results_path = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    print(f"Reading results from {results_path}...")
    
    # Aggregate statistics
    tar_stats = defaultdict(lambda: {
        "count": 0,
        "der_values": [],
        "missed_values": [],
        "false_alarm_values": [],
        "confusion_values": [],
    })
    
    speaker_stats = defaultdict(lambda: {
        "count": 0,
        "der_values": [],
    })
    
    all_der = []
    all_missed = []
    all_false_alarm = []
    all_confusion = []
    
    with results_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tar = row.get("source_tar", "unknown")
            num_speakers = row.get("num_speakers", "unknown")
            der = float(row["der"])
            missed = float(row.get("missed_speech", 0))
            false_alarm = float(row.get("false_alarm", 0))
            confusion = float(row.get("speaker_confusion", 0))
            
            # Per-tar stats
            tar_stats[tar]["count"] += 1
            tar_stats[tar]["der_values"].append(der)
            tar_stats[tar]["missed_values"].append(missed)
            tar_stats[tar]["false_alarm_values"].append(false_alarm)
            tar_stats[tar]["confusion_values"].append(confusion)
            
            # Per-speaker count stats
            speaker_stats[num_speakers]["count"] += 1
            speaker_stats[num_speakers]["der_values"].append(der)
            
            # Overall stats
            all_der.append(der)
            all_missed.append(missed)
            all_false_alarm.append(false_alarm)
            all_confusion.append(confusion)
    
    # Write summary report
    summary_path = output_dir / "SUMMARY.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SPEAKER DIARIZATION EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total items: {len(all_der)}\n")
        if all_der:
            avg_der = sum(all_der) / len(all_der)
            min_der = min(all_der)
            max_der = max(all_der)
            f.write(f"DER (Diarization Error Rate):\n")
            f.write(f"  Average: {avg_der:.4f}\n")
            f.write(f"  Min:     {min_der:.4f}\n")
            f.write(f"  Max:     {max_der:.4f}\n")
            
            if all_missed:
                avg_missed = sum(all_missed) / len(all_missed)
                f.write(f"Missed Speech: {avg_missed:.4f}\n")
            if all_false_alarm:
                avg_fa = sum(all_false_alarm) / len(all_false_alarm)
                f.write(f"False Alarm: {avg_fa:.4f}\n")
            if all_confusion:
                avg_conf = sum(all_confusion) / len(all_confusion)
                f.write(f"Speaker Confusion: {avg_conf:.4f}\n")
        
        f.write("\n")
        
        # Per-tar statistics
        f.write("PER-TAR STATISTICS\n")
        f.write("-" * 80 + "\n")
        for tar in sorted(tar_stats.keys()):
            stats = tar_stats[tar]
            f.write(f"\n{tar}:\n")
            f.write(f"  Items: {stats['count']}\n")
            if stats["der_values"]:
                avg_der = sum(stats["der_values"]) / len(stats["der_values"])
                f.write(f"  Avg DER: {avg_der:.4f}\n")
                f.write(f"  Min DER: {min(stats['der_values']):.4f}\n")
                f.write(f"  Max DER: {max(stats['der_values']):.4f}\n")
        
        f.write("\n")
        
        # Per-speaker count statistics
        f.write("PER-SPEAKER-COUNT STATISTICS\n")
        f.write("-" * 80 + "\n")
        for num_speakers in sorted(speaker_stats.keys()):
            stats = speaker_stats[num_speakers]
            f.write(f"\n{num_speakers} speaker(s):\n")
            f.write(f"  Items: {stats['count']}\n")
            if stats["der_values"]:
                avg_der = sum(stats["der_values"]) / len(stats["der_values"])
                f.write(f"  Avg DER: {avg_der:.4f}\n")
                f.write(f"  Min DER: {min(stats['der_values']):.4f}\n")
                f.write(f"  Max DER: {max(stats['der_values']):.4f}\n")
    
    print(f"Summary report written to {summary_path}")
    
    # Write per-tar CSV
    tar_csv_path = output_dir / "results_by_tar.csv"
    with tar_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_tar",
            "count",
            "avg_der",
            "min_der",
            "max_der",
            "avg_missed",
            "avg_false_alarm",
            "avg_confusion",
        ])
        for tar in sorted(tar_stats.keys()):
            stats = tar_stats[tar]
            avg_der = sum(stats["der_values"]) / len(stats["der_values"]) if stats["der_values"] else 0
            min_der = min(stats["der_values"]) if stats["der_values"] else 0
            max_der = max(stats["der_values"]) if stats["der_values"] else 0
            avg_missed = sum(stats["missed_values"]) / len(stats["missed_values"]) if stats["missed_values"] else 0
            avg_fa = sum(stats["false_alarm_values"]) / len(stats["false_alarm_values"]) if stats["false_alarm_values"] else 0
            avg_conf = sum(stats["confusion_values"]) / len(stats["confusion_values"]) if stats["confusion_values"] else 0
            
            writer.writerow([
                tar,
                stats["count"],
                f"{avg_der:.4f}",
                f"{min_der:.4f}",
                f"{max_der:.4f}",
                f"{avg_missed:.4f}",
                f"{avg_fa:.4f}",
                f"{avg_conf:.4f}",
            ])
    
    print(f"Per-tar results written to {tar_csv_path}")
    
    # Write per-speaker-count CSV
    speaker_csv_path = output_dir / "results_by_speaker_count.csv"
    with speaker_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_speakers", "count", "avg_der", "min_der", "max_der"])
        for num_speakers in sorted(speaker_stats.keys()):
            stats = speaker_stats[num_speakers]
            avg_der = sum(stats["der_values"]) / len(stats["der_values"]) if stats["der_values"] else 0
            min_der = min(stats["der_values"]) if stats["der_values"] else 0
            max_der = max(stats["der_values"]) if stats["der_values"] else 0
            
            writer.writerow([
                num_speakers,
                stats["count"],
                f"{avg_der:.4f}",
                f"{min_der:.4f}",
                f"{max_der:.4f}",
            ])
    
    print(f"Per-speaker results written to {speaker_csv_path}")
    print("\nAggregation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate speaker diarization results across all tar subsets."
    )
    parser.add_argument(
        "--results-csv",
        required=True,
        help="Path to der_results.csv from evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for aggregated reports.",
    )
    args = parser.parse_args()
    main(args)
