"""Reproduce Yale HER2 treatment probes with HF MOSAIC and frozen reference folds."""

import argparse
import csv
import hashlib
import importlib
import json
import time
import sys
from pathlib import Path

import h5py
import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel
from huggingface_hub import snapshot_download

MODEL_ID = "Astaxanthin/MOSAIC"
EXAMPLE_DATA_URL = "https://drive.google.com/drive/folders/1IUbXoDB1KSfdNoSZMUHderIqItNFMSA1?usp=drive_link"


def resolve_model(model_id, revision="main"):
    """Resolve a local folder or pinned Hub snapshot, then verify model type."""
    path = Path(model_id).expanduser()
    if not path.is_dir():
        if model_id == "Astaxanthin/KEEP":
            raise ValueError(
                "KEEP extracts patch features; load Astaxanthin/MOSAIC here."
            )
        path = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                allow_patterns=["*.py", "*.json", "*.safetensors", "vocab.txt"],
            )
        )
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "mosaic":
        raise ValueError("Expected a MOSAIC model directory, not a patch encoder.")
    return str(path.resolve())


def model_identity(folder):
    """Fingerprint weights, model code and tokenizer for safe cache reuse."""
    files = [
        p
        for p in Path(folder).iterdir()
        if p.suffix in {".py", ".json", ".safetensors"} or p.name == "vocab.txt"
    ]
    result = {}
    for path in sorted(files):
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        result[path.name] = hasher.hexdigest()
    return result


def ask(label, default, integer=False):
    """Terminal wizard: press Enter to keep a default."""
    while True:
        value = (
            input(f"{label} [{default if default is not None else 'optional'}]: ")
            .strip()
            .strip('"')
        )
        if not value:
            return default
        if not integer:
            return value
        try:
            return int(value)
        except ValueError:
            print("Please enter an integer.")


def choose_interactive(args, argv):
    return args.interactive or (
        not argv and not args.no_interactive and sys.stdin.isatty()
    )


def check_paths(args, metadata_key):
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        raise ValueError("Use cpu, cuda, or cuda:N for --device")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu")
    for field in ("features", metadata_key):
        path = Path(getattr(args, field)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {field}: {path}\n"
                f"Download the example features and metadata from {EXAMPLE_DATA_URL}\n"
                "See README: Download example data, or pass your local paths explicitly."
            )
        setattr(args, field, str(path))
    args.output = str(Path(args.output).expanduser().resolve())
    if getattr(args, "reference", None):
        args.reference = str(Path(args.reference).expanduser().resolve())
        if not Path(args.reference).is_file():
            raise FileNotFoundError(args.reference)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=f"Example data: {EXAMPLE_DATA_URL}"
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--features", default=str(base / "data" / "example_yale-her2_feats")
    )
    parser.add_argument(
        "--metadata",
        default=str(base / "data" / "yale_her2_dataset.json"),
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional original predictions JSON for patient-level comparison",
    )
    parser.add_argument("--output", default=str(base / "outputs" / "yale_her2"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--feature-method",
        choices=["predict", "treatment"],
        default="treatment",
        help="treatment reproduces the original batches of spatial views; predict uses the public single-view loop",
    )
    parser.add_argument("--view-batch-size", type=int, default=8)

    parser.add_argument("--revision", default="main", help="HF revision or commit SHA")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check cohort/split without model inference",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--interactive", action="store_true", help="Prompt for paths and settings"
    )
    mode.add_argument(
        "--no-interactive", action="store_true", help="Never prompt (for batch jobs)"
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if choose_interactive(args, argv):
        print("MOSAIC | yale_her2 reproduction — Enter keeps the displayed default")
        args.model = ask("MOSAIC Hub ID or local model folder", args.model)
        args.features = ask("HDF5 feature folder", args.features)
        args.metadata = ask("Metadata JSON", args.metadata)
        args.output = ask("Output folder", args.output)
        args.device = ask("Device: cpu / cuda", args.device)
        args.reference = ask("Reference predictions JSON (optional)", args.reference)
    check_paths(args, "metadata")
    return args


def run(args):
    """Run the evaluation; also callable from the accompanying notebook."""
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "slide_embeddings"
    cache.mkdir(exist_ok=True)
    reference, old = None, {}
    if args.reference:
        reference = json.loads(Path(args.reference).read_text(encoding="utf-8-sig"))
        reference = reference.get("treatment_response_5fold", {}).get(
            "yale_her2", reference
        )
        old = {r["case_id"]: r for r in reference["oof_predictions"]}
        if len(old) != len(reference["oof_predictions"]):
            raise ValueError("Duplicate reference patients")
    if args.view_batch_size < 1:
        raise ValueError("View batch size must be positive")
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8-sig"))
    label_map = {
        "effective": 1,
        "invalid": 0,
        "responder": 1,
        "nonresponder": 0,
        "pcr": 1,
        "rd": 0,
    }
    slides = {}
    labels = {}
    audit = []
    # Match only metadata-labelled slides, excluding unrelated H5 files.
    for slide, info in metadata.items():
        case = str(info.get("case_id") or info.get("patient_id") or slide).strip()
        candidates = [
            info.get("treatment_response"),
            info.get("raw_treatment_response"),
            info.get("task", {}).get("treatment_response"),
            info.get("surv_prior", {}).get("treatment_response"),
        ]
        response = next(
            (
                str(x).strip().lower()
                for x in candidates
                if str(x).strip().lower() in label_map
            ),
            None,
        )
        path = Path(args.features) / (Path(slide).stem + ".h5")
        if response is None or not path.exists():
            audit.append(
                {
                    "slide": slide,
                    "reason": (
                        "missing_label" if response is None else "missing_feature"
                    ),
                }
            )
            continue
        if case in labels and labels[case] != label_map[response]:
            raise ValueError("Conflicting labels: " + case)
        labels[case] = label_map[response]
        slides.setdefault(case, []).append((slide, path))
    if audit:
        write_csv(out / "excluded_slides.csv", audit)
    if old and set(slides) != set(old):
        raise ValueError(
            f"Patient mismatch: missing={set(old)-set(slides)}, extra={set(slides)-set(old)}"
        )
    ids = sorted(slides)
    y = np.array([labels[k] for k in ids], dtype=np.int64)
    # This fingerprint prevents silent changes to the published 85-patient cohort.
    cohort_hash = hashlib.sha256(
        json.dumps(list(zip(ids, y.tolist())), separators=(",", ":")).encode()
    ).hexdigest()
    if (
        cohort_hash
        != "41c8677e5b95a0c6df51ba8a85887819164451aad77b1011646af8cd567c3835"
    ):
        raise ValueError(
            "Cohort differs from the validated Yale HER2 cohort (85 patients, 36 responders). Check the metadata/features."
        )
    generated = np.zeros(len(ids), dtype=int)
    for fold, (_, test) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros((len(ids), 1)), y
        ),
        1,
    ):
        generated[test] = fold
    folds = np.array([int(old[k]["fold"]) for k in ids]) if old else generated
    if old and any(labels[k] != int(old[k]["gt"]) for k in ids):
        raise ValueError("Reference label mismatch")
    split_matches = bool(np.array_equal(generated, folds))
    if not split_matches:
        raise ValueError("Reference folds differ from the validated seed-42 split")
    print(
        f"[COHORT] n={len(ids)} positive={sum(y)} negative={sum(y==0)} seed42_matches_reference={split_matches}",
        flush=True,
    )
    write_csv(
        out / "frozen_folds.csv",
        [
            dict(case_id=k, label=int(y[i]), fold=int(folds[i]))
            for i, k in enumerate(ids)
        ],
    )

    if args.validate_only:
        return {
            "patients": len(ids),
            "responders": int(sum(y)),
            "cohort_matches": True,
            "split_matches": split_matches,
        }
    args.model = resolve_model(args.model, args.revision)
    # A cache is valid only for identical model files, metadata, and feature files.
    config = dict(
        model=str(Path(args.model).resolve()),
        metadata_sha256=digest(args.metadata),
        reference_sha256=digest(args.reference) if args.reference else None,
        device=args.device,
        threads=args.threads,
        torch_version=torch.__version__,
        sklearn_version=sklearn.__version__,
        feature_method=args.feature_method,
        view_batch_size=args.view_batch_size,
        pooling="elementwise max over views, mean over slides, L2 normalization",
        model_files=model_identity(args.model),
    )
    config_path = out / "run_config.json"
    if not config_path.exists() and any(cache.iterdir()):
        raise ValueError("Cache has no run_config.json; choose an empty output folder")
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(
            "Cache configuration mismatch; choose another output directory"
        )
    config_path.write_text(json.dumps(config, indent=2))
    model = (
        AutoModel.from_pretrained(args.model, trust_remote_code=True)
        .to(args.device)
        .eval()
    )
    vectors = []
    count = 0
    for case in ids:
        case_vectors = []
        for slide, path in slides[case]:
            count += 1
            cached = cache / (hashlib.sha256(slide.encode()).hexdigest() + ".npz")
            stat = np.array(
                [path.stat().st_size, path.stat().st_mtime_ns], dtype=np.int64
            )
            start = time.time()
            if cached.exists():
                with np.load(cached, allow_pickle=False) as data:
                    if not np.array_equal(stat, data["stat"]):
                        raise ValueError("Feature file changed: " + str(path))
                    vector = data["embedding"]
                    views = int(data["num_views"])
                print(f"[CACHE {count}] {slide}", flush=True)
            else:
                print(
                    f"[RUN {count}/{sum(map(len,slides.values()))}] {slide}", flush=True
                )
                with h5py.File(path, "r") as f:
                    features, coords = f["features"][:], f["coords"][:]
                # Treatment probes use shared WSI embeddings, not survival risks.
                if args.feature_method == "predict":
                    result = model.predict(patch_features=features, patch_coords=coords)
                    vector = (
                        result["wsi_features"].numpy().reshape(-1).astype(np.float32)
                    )
                    views = result["num_views"]
                else:
                    # Use the HF package's spatial preparation and loaded backbone.
                    # Match _ssl_forward_eval_feats: batch up to eight views and use
                    # the final shared CLS token, followed by elementwise max.
                    package = model.__class__.__module__.rsplit(".", 1)[0]
                    spatial = importlib.import_module(
                        package + ".spatial"
                    ).SpatialViews()
                    feats, xy = spatial._prepare_eval_2d_features(features, coords)
                    if feats.ndim == 2:
                        feats, xy = feats[None], xy[None]
                    view_vectors = []
                    with torch.inference_mode():
                        for start_view in range(0, len(feats), args.view_batch_size):
                            fbatch = torch.as_tensor(
                                feats[start_view : start_view + args.view_batch_size],
                                dtype=torch.float32,
                                device=args.device,
                            )
                            cbatch = torch.as_tensor(
                                xy[start_view : start_view + args.view_batch_size],
                                device=args.device,
                            )
                            cls, _, _ = model.core.ssl_vision_aggregator(
                                fbatch,
                                cbatch,
                                model.core.patch_size_lv0,
                                mask_ratio=0.0,
                            )
                            view_vectors.append(cls[:, -1, :].cpu())
                    vector = (
                        torch.cat(view_vectors)
                        .max(dim=0)
                        .values.numpy()
                        .astype(np.float32)
                    )
                    views = len(feats)
                np.savez(cached, embedding=vector, stat=stat, num_views=views)
                print(
                    f"[DONE {count}] views={views} elapsed={time.time()-start:.1f}s",
                    flush=True,
                )
            case_vectors.append(vector)
        vector = np.mean(np.stack(case_vectors), axis=0).astype(np.float32)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid patient embedding: " + case)
        vectors.append(vector / norm)
    X = np.stack(vectors).astype(np.float32)
    np.savez(
        out / "patient_embeddings.npz",
        case_ids=np.array(ids),
        features=X,
        labels=y,
        folds=folds,
    )

    # Freeze reference fold membership. Fit scaler and probe on training folds only.
    predictions = []
    fold_metrics = []
    for fold in range(1, 6):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        scaler = StandardScaler()
        x_train, x_test = scaler.fit_transform(X[train]), scaler.transform(X[test])
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42 + fold - 1
        )
        clf.fit(x_train, y[train])
        score = clf.predict_proba(x_test)[:, 1]
        pred = (score >= 0.5).astype(np.int64)
        row = dict(
            fold=fold,
            auc=float(roc_auc_score(y[test], score)),
            balanced_accuracy=float(balanced_accuracy_score(y[test], pred)),
            accuracy=float(accuracy_score(y[test], pred)),
        )
        fold_metrics.append(row)
        for i, s, p in zip(test, score, pred):
            prediction = dict(
                case_id=ids[i], fold=fold, gt=int(y[i]), pred=int(p), score=float(s)
            )
            if old:
                r = old[ids[i]]
                prediction.update(
                    reference_score=float(r["score"]),
                    reference_pred=int(r["pred"]),
                    score_difference=float(s) - float(r["score"]),
                )
            predictions.append(prediction)
        print("[FOLD]", row, flush=True)
    gt = [r["gt"] for r in predictions]
    scores = [r["score"] for r in predictions]
    report = dict(
        n_patients=len(ids),
        n_positive=int(sum(y)),
        n_negative=int(sum(y == 0)),
        reference_folds_used=bool(old),
        seed42_folds_identical=split_matches,
        oof_auc=float(roc_auc_score(gt, scores)),
        folds=fold_metrics,
    )
    if old:
        old_scores = [r["reference_score"] for r in predictions]
        report.update(
            reference_oof_auc=float(roc_auc_score(gt, old_scores)),
            max_absolute_score_difference=max(
                abs(r["score_difference"]) for r in predictions
            ),
            mean_absolute_score_difference=float(
                np.mean([abs(r["score_difference"]) for r in predictions])
            ),
            changed_predictions=sum(
                r["pred"] != r["reference_pred"] for r in predictions
            ),
        )
    for metric in ["auc", "balanced_accuracy", "accuracy"]:
        values = np.array([r[metric] for r in fold_metrics], dtype=np.float32)
        report[metric + "_mean"] = float(values.mean())
        report[metric + "_std"] = float(values.std(ddof=1))
        if reference:
            report["reference_" + metric + "_mean"] = reference[metric + "_mean"]
    write_csv(out / "oof_predictions.csv", predictions)
    write_csv(out / "fold_metrics.csv", fold_metrics)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print("[METRICS]", json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
