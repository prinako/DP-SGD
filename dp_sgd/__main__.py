import argparse
import csv
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
from statistics import fmean

import torch

from .data import load_data, synthetic_data
from .evaluation import evaluate
from .model import PECSC_REVISION, load_models
from .training import train_receiver


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare SGD and DP-SGD on a frozen DeepSC transmitter")
    parser.add_argument("--pecsc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local output directory")
    parser.add_argument("--synthetic", action="store_true", help="Offline smoke test with public toy data")
    inputs = ("teacher-manifest", "encoder", "decoder", "tokenizer", "train", "evaluation")
    for name in inputs:
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--initialization-provenance",
                        help="Explain the relationship between the initial assets and protected records")
    parser.add_argument("--semantic-model",
                        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    parser.add_argument("--semantic-revision", help="40-character embedding model revision")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--secure-mode", action="store_true",
                        help="Enable Opacus secure randomness with torchcsprng installed")
    parser.add_argument("--seed", type=int, default=42,
                        help="Synthetic initialization and baseline SGD seed")
    parser.add_argument("--train-snr", type=float, default=8.0)
    parser.add_argument("--eval-snrs", type=float, nargs="+", default=[2.0, 8.0, 14.0])
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args(argv)
    real_inputs = [getattr(args, name.replace("-", "_")) for name in inputs]
    if args.synthetic:
        if any(real_inputs) or args.initialization_provenance or args.semantic_revision:
            parser.error("--synthetic cannot use research inputs")
        if args.device == "cpu":
            torch.set_num_threads(1)
    elif (not all(real_inputs) or not (args.initialization_provenance or "").strip()
          or not re.fullmatch("[0-9a-f]{40}", args.semantic_revision or "")):
        parser.error("Research runs require all six input paths, provenance and a pinned embedding revision")
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.embedding_batch_size) < 1:
        parser.error("Epoch and batch sizes must be positive")
    if any(not math.isfinite(value) or value <= 0
           for value in (args.learning_rate, args.noise_multiplier, args.max_grad_norm)):
        parser.error("Learning rate, noise multiplier and maximum gradient norm must be finite and positive")
    if (any(not math.isfinite(value) or abs(value) > 100 for value in [args.train_snr, *args.eval_snrs])
            or len(set(args.eval_snrs)) != len(args.eval_snrs)
            or len(set(args.trial_seeds)) != len(args.trial_seeds)):
        parser.error("SNRs must be finite in [-100, 100] and evaluation SNRs and trial seeds must be unique")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.output.exists():
        parser.error("--output must be a new directory")
    return args


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, cosine_metric):
    groups = {}
    for row in rows:
        groups.setdefault((row["snr_db"], row["trial_seed"]), {})[row["model"]] = row
    metrics = ["chrfpp", cosine_metric]
    result = []
    for snr in sorted({row["snr_db"] for row in rows}):
        trials = [arms for (level, _), arms in groups.items() if level == snr]
        entry = {"snr_db": snr, "trials": len(trials)}
        for metric in metrics:
            for arm in ("initial", "sgd", "dp_sgd"):
                entry[f"{arm}_{metric}"] = fmean(trial[arm][metric] for trial in trials)
            entry[f"dp_minus_sgd_{metric}"] = fmean(
                trial["dp_sgd"][metric] - trial["sgd"][metric] for trial in trials
            )
        result.append(entry)
    return result


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    transmitter, initial, arch, provenance = load_models(args)
    transmitter.to(args.device)
    initial.to(args.device).eval()
    dataset = synthetic_data() if args.synthetic else load_data(
        args.train, args.evaluation, args.tokenizer, arch["max_length"], arch["vocab_size"]
    )
    n = len(dataset.train)
    args.delta = n ** -1.1 if args.delta is None else args.delta
    if not 0 < args.delta < 1 / n or args.batch_size >= n:
        raise ValueError("Require 0 < delta < 1/N and a batch size below N")
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)

    sgd, baseline = train_receiver(initial, transmitter, dataset, args, private=False)
    private_receiver, privacy = train_receiver(initial, transmitter, dataset, args, private=True)
    models = {"initial": initial, "sgd": sgd, "dp_sgd": private_receiver}
    for name in ("sgd", "dp_sgd"):
        torch.save(models[name].state_dict(), args.output / f"{name}.pt")

    metadata = {
        "mode": "synthetic" if args.synthetic else "research",
        "pecsc_revision": PECSC_REVISION,
        "architecture": arch, "initial_assets": provenance,
        "public_training_slots": n, "evaluation_records": len(dataset.evaluation),
        "baseline": baseline, "private_decoder_training": privacy,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "channel": "AWGN", "normalization": "per sentence, non-padding symbols only",
        "training_snr_db": args.train_snr, "evaluation_snrs_db": args.eval_snrs,
        "evaluation_trial_seeds": args.trial_seeds, "evaluation_batch_size": args.eval_batch_size,
        "device": args.device, "torch": str(torch.__version__), "opacus": version("opacus"),
        "privacy_scope": "RDP accounting for decoder training conditional on independent initial assets",
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    rows, metric_metadata = evaluate(transmitter, models, dataset, args)
    save_csv(args.output / "trials.csv", rows)
    save_csv(args.output / "summary.csv", summarize(rows, metric_metadata["cosine"]["metric"]))
    (args.output / "metrics.json").write_text(
        json.dumps(metric_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{args.output} | Decoder-training epsilon={privacy['epsilon']:.4g}, delta={args.delta:.4g}")


if __name__ == "__main__":
    main()
