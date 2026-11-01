"""CLI for TranscriptIQ.

  python main.py generate-synthetic
  python main.py run --mode demo            # heuristic demo, no API key
  python main.py run --mode llm --limit 3   # smoke-test the LLM path first
  python main.py review-sample --mode demo
  python main.py evaluate --mode demo
"""
import argparse
import logging
import sys

from config.settings import get_settings
from src import storage
from src.evaluation import (compute_ground_truth_metrics, compute_metrics, load_ground_truth, load_review,
                            make_review_sample)
from src.llm_client import SetupError
from src.pipeline import run_pipeline
from src.synthetic import write_synthetic


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="TranscriptIQ pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate-synthetic", help="write the labeled SYNTHETIC demo dataset")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--seed", type=int, default=42)
    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("--input", help="file or folder (default: data/raw, else synthetic demo set)")
    r.add_argument("--mode", choices=["demo", "llm"], default="llm")
    r.add_argument("--limit", type=int)
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--clusters", type=int, help="force k for KMeans (default: best silhouette)")
    r.add_argument("--skip-clusters", action="store_true")
    r.add_argument("--origin", help="override data_origin (e.g. original / synthetic)")
    r.add_argument("--out")
    for name in ("review-sample", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--mode", choices=["demo", "llm"], default="demo")
        s.add_argument("--n", type=int, default=24)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    if a.cmd == "generate-synthetic":
        print(f"Wrote SYNTHETIC dataset to {write_synthetic(settings.synthetic_dir, a.n, a.seed)}")
        return 0
    if a.cmd == "run":
        try:
            meta = run_pipeline(a.input, a.mode, a.out, a.limit, not a.no_resume, settings=settings, n_clusters=a.clusters,
                                skip_clusters=a.skip_clusters, origin=a.origin)
        except SetupError as exc:
            print(f"SETUP ERROR: {exc}", file=sys.stderr)
            return 2
        print(meta["disclaimer"])
        print({k: meta[k] for k in ("records_loaded", "records_analyzable", "processed_this_run", "skipped_resumed", "failed", "runtime_seconds")})
        return 1 if meta["failed"] else 0

    run_dir = settings.processed_dir / a.mode
    data = storage.load_processed(run_dir)
    if data is None:
        print(f"No processed data in {run_dir}. Run the pipeline first.", file=sys.stderr)
        return 2
    sheet = run_dir / "review_sheet.csv"
    if a.cmd == "review-sample":
        if sheet.exists():
            print(f"{sheet} already exists; not overwriting your labels.")
        else:
            make_review_sample(data["df"], a.n).to_csv(sheet, index=False)
            print(f"Wrote {sheet}. Fill reviewer_topic / reviewer_sentiment / reviewer_renewal_risk (y/n) / unsupported_claims (y/n).")
        return 0
    review = load_review(sheet)
    metrics = compute_metrics(review) if review is not None else None
    synthetic = data["df"][data["df"]["data_origin"] == "synthetic"]
    ground_truth = load_ground_truth(settings.synthetic_dir / "ground_truth.csv")
    if not synthetic.empty and ground_truth is not None:
        truth_metrics = compute_ground_truth_metrics(synthetic, ground_truth)
        if truth_metrics is None:
            print("No synthetic ground-truth labels matched the processed transcript IDs.")
        else:
            print("Synthetic ground-truth metrics (scenario-designed labels; not human-reviewed):")
            print(truth_metrics)
    if metrics:
        print("Human-review metrics:")
        print(metrics)
    else:
        print("No reviewed labels yet: no human-review metrics computed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
