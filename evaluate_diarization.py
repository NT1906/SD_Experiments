import argparse
import csv
from pathlib import Path

from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from tqdm import tqdm


def load_annotation(rttm_path: Path):
    """Load a single RTTM file as a pyannote Annotation."""
    annotations = load_rttm(rttm_path)

    if len(annotations) == 0:
        raise RuntimeError(f"No annotation found in {rttm_path}")

    return next(iter(annotations.values()))


def main(args):

    predicted_root = Path(args.predicted_rttm_root)
    reference_root = Path(args.reference_rttm_root)
    output_root = Path(args.output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    metadata_csv = reference_root.parent / "metadata_sd.csv"

    metadata = {}

    if metadata_csv.exists():
        with metadata_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row["conv_id"]] = row

    results_csv = output_root / "der_results.csv"

    existing = set()

    if results_csv.exists():
        with results_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row["conv_id"])

    csv_mode = "a" if results_csv.exists() else "w"

    metric = DiarizationErrorRate()

    summary = {
        "all": [],
        "single": [],
        "two": [],
        "three": [],
    }

    with results_csv.open(csv_mode, newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)

        if csv_mode == "w":
            writer.writerow([
                "conv_id",
                "num_speakers",
                "source_tar",
                "der",
            ])

        predicted_files = sorted(predicted_root.glob("*_predicted.rttm"))

        skipped = 0
        evaluated = 0

        for pred_file in tqdm(predicted_files, desc="Evaluating DER"):

            conv_id = pred_file.stem.replace("_predicted", "")

            if conv_id in existing:
                skipped += 1
                continue

            ref_file = reference_root / f"{conv_id}.rttm"

            if not ref_file.exists():
                continue

            try:
                reference = load_annotation(ref_file)
                hypothesis = load_annotation(pred_file)

                der = metric(reference, hypothesis)

            except Exception as e:
                print(f"Skipping {conv_id}: {e}")
                continue

            info = metadata.get(conv_id, {})

            num_speakers = str(info.get("num_speakers", "unknown"))
            source_tar = info.get("source_tar", "unknown")

            writer.writerow([
                conv_id,
                num_speakers,
                source_tar,
                f"{der:.6f}",
            ])

            summary["all"].append(der)

            if num_speakers == "1":
                summary["single"].append(der)
            elif num_speakers == "2":
                summary["two"].append(der)
            elif num_speakers == "3":
                summary["three"].append(der)

            evaluated += 1

    print()
    print("=====================================")
    print(f"Evaluated : {evaluated}")
    print(f"Skipped   : {skipped}")
    print("=====================================")

    print()

    for key in ["all", "single", "two", "three"]:

        values = summary[key]

        if len(values) == 0:
            print(f"{key:8s}: No samples")
        else:
            avg = sum(values) / len(values)
            print(f"{key:8s}: DER = {avg:.4f} ({len(values)} files)")

    tar_summary = {}

    with results_csv.open("r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            tar = row["source_tar"]
            der = float(row["der"])

            tar_summary.setdefault(tar, []).append(der)

    tar_csv = output_root / "der_results_by_tar.csv"

    with tar_csv.open("w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "source_tar",
            "num_items",
            "avg_der",
        ])

        for tar in sorted(tar_summary):

            ders = tar_summary[tar]

            writer.writerow([
                tar,
                len(ders),
                f"{sum(ders)/len(ders):.6f}",
            ])

    print()
    print(f"Results written to {results_csv}")
    print(f"Per-tar summary written to {tar_csv}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predicted-rttm-root",
        required=True,
    )

    parser.add_argument(
        "--reference-rttm-root",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    args = parser.parse_args()

    main(args)
