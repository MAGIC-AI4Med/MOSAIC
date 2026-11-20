"""Evaluate four OS settings on matched HCMI patients using HF MOSAIC."""

import argparse
import csv
import hashlib
import json
import time
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
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


SETTINGS = (
    "clinical",
    "clinical_vision",
    "clinical_genomics",
    "clinical_vision_genomics",
)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cindex(rows, setting, indices):
    event = np.array([rows[i]["os_event"] for i in indices], dtype=bool)
    times = np.array([rows[i]["os_time"] for i in indices], dtype=float)
    risk = np.array([rows[i][setting] for i in indices])
    # Harrell concordance: earlier observed events are comparable with later
    # observations and censored observations at the same time. Event/event
    # time ties are excluded; risk ties within 1e-8 receive half credit.
    numerator = denominator = 0
    for i in np.flatnonzero(event):
        valid = (times > times[i]) | ((times == times[i]) & ~event)
        delta = risk[i] - risk[valid]
        tied = np.abs(delta) <= 1e-8
        numerator += np.sum((delta > 0) & ~tied) + 0.5 * np.sum(tied)
        denominator += len(delta)
    return float(numerator / denominator) if denominator else float("nan")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=f"Example data: {EXAMPLE_DATA_URL}"
    )
    here = Path(__file__).resolve().parent.parent
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--features", default=str(here / "data" / "example_hcmi_feats"))
    parser.add_argument("--data", default=str(here / "data" / "hcmi_dataset.json"))
    parser.add_argument("--output", default=str(here / "outputs" / "hcmi_os"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Smoke test only; 0 evaluates all eligible slides.",
    )

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
        print("MOSAIC | hcmi reproduction — Enter keeps the displayed default")
        args.model = ask("MOSAIC Hub ID or local model folder", args.model)
        args.features = ask("HDF5 feature folder", args.features)
        args.data = ask("Metadata JSON", args.data)
        args.output = ask("Output folder", args.output)
        args.device = ask("Device: cpu / cuda", args.device)
        args.bootstrap = ask("Bootstrap resamples", args.bootstrap, integer=True)
    check_paths(args, "data")
    return args


def run(args):
    """Run the evaluation; also callable from the accompanying notebook."""
    if args.bootstrap < 1 or args.limit < 0:
        raise ValueError("Bootstrap must be positive and limit nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "slide_cache"
    cache.mkdir(exist_ok=True)
    data = json.loads(Path(args.data).read_text(encoding="utf-8-sig"))
    records = []
    audit = []
    # Four settings share the same feature-available, OS-valid test population.
    for slide, info in sorted(data.items()):
        path = Path(args.features) / (Path(slide).stem + ".h5")
        survival = info.get("survival", {})
        event, duration = survival.get("os_state", -1), survival.get("os_time", -1)
        reason = None
        if info.get("state") != "test":
            reason = "not_test"
        elif not path.is_file():
            reason = "missing_features"
        elif (
            event not in (0, 1)
            or not isinstance(duration, (int, float))
            or not np.isfinite(duration)
            or duration <= 0
        ):
            reason = "invalid_os"
        elif not info.get("case_id") or not info.get("project"):
            reason = "missing_patient_or_project"
        if reason:
            audit.append({"slide": slide, "reason": reason})
            print("[SKIP]", slide, reason, flush=True)
        else:
            records.append((slide, info, path))
    if args.limit:
        records = records[: args.limit]
    save_csv(out / "screening_audit.csv", audit)
    if not records:
        raise ValueError("No eligible HCMI slides; check metadata and feature paths")
    print(
        f"[COHORT] {len(records)} slides; {len({r[1]['case_id'] for r in records})} patients",
        flush=True,
    )
    if args.validate_only:
        return {
            "eligible_slides": len(records),
            "patients": len({r[1]["case_id"] for r in records}),
        }
    args.model = resolve_model(args.model, args.revision)
    options = {
        k: v
        for k, v in vars(args).items()
        if k not in ("interactive", "no_interactive", "validate_only", "output")
    }
    config = dict(
        options,
        dataset_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        view_pool="max",
        patient_slide_pool="mean",
        genomics="TP53 SNV only; CNV unknown; missing SNV remains unknown",
    )
    config["model_files"] = model_identity(args.model)
    config["torch_version"] = torch.__version__
    config["numpy_version"] = np.__version__
    old = out / "run_config.json"
    if not old.exists() and any(cache.iterdir()):
        raise ValueError("Cache has no run_config.json; use an empty output folder")
    if old.exists():
        previous = json.loads(old.read_text())
        # Moving completed outputs does not change the inference configuration.
        comparable = {k: v for k, v in config.items() if k != "output"}
        previous = {k: v for k, v in previous.items() if k != "output"}
        if previous != comparable:
            raise ValueError(
                "Output folder belongs to a different run configuration. Choose a new --output."
            )
    old.write_text(json.dumps(config, indent=2))
    model = (
        AutoModel.from_pretrained(args.model, trust_remote_code=True)
        .to(args.device)
        .eval()
    )
    slides = []
    for i, (slide, info, path) in enumerate(records, 1):
        cp = cache / (hashlib.sha256(slide.encode()).hexdigest() + ".json")
        feature_stat = [path.stat().st_size, path.stat().st_mtime_ns]
        if cp.exists():
            saved = json.loads(cp.read_text())
            if saved["feature_stat"] != feature_stat:
                raise ValueError("Feature changed: " + str(path))
            row = saved["row"]
            print(f"[CACHE {i}/{len(records)}] {slide}", flush=True)
        else:
            start = time.time()
            print(f"[RUN {i}/{len(records)}] {slide}", flush=True)
            with h5py.File(path, "r") as f:
                features = f["features"][:]
                coords = f["coords"][:]
            clinical = dict(info.get("clinical", {}))
            prior = info.get("surv_prior", {})
            clinical.update({k: prior.get(k, -1) for k in ["grade", "subtype"]})
            snv = prior.get("mutation", -1)
            genomic = {
                "snv": (
                    (["TP53"] if "TP53" in snv else []) if isinstance(snv, list) else -1
                ),
                "cnv": -1,
            }
            metadata = dict(cancer_type=info.get("cancer type"), clinical=clinical)
            row = dict(
                tp53_snv_status=(
                    "positive"
                    if genomic["snv"] == ["TP53"]
                    else "negative" if genomic["snv"] == [] else "unknown"
                ),
                slide=slide,
                project=info["project"],
                case_id=info["case_id"],
                os_event=int(info["survival"]["os_state"]),
                os_time=float(info["survival"]["os_time"]),
            )
            # Cache only the frozen visual backbone outputs across CV and CVG.
            # Fusion and survival risks are recomputed independently for each setting.
            # Cache on CPU to avoid retaining all views in GPU memory.
            original = model.core.ssl_vision_aggregator.forward
            visual_cache = []
            cursor = [0]
            reuse = [False]

            def cached_vision(*a, **kw):
                if reuse[0]:
                    value = visual_cache[cursor[0]]
                    cursor[0] += 1
                    return tuple(
                        x.to(args.device) if isinstance(x, torch.Tensor) else x
                        for x in value
                    )
                value = original(*a, **kw)
                visual_cache.append(
                    tuple(
                        x.detach().cpu() if isinstance(x, torch.Tensor) else x
                        for x in value
                    )
                )
                return value

            model.core.ssl_vision_aggregator.forward = cached_vision
            try:
                for setting in SETTINGS:
                    inputs = dict(metadata)
                    if "genomics" in setting:
                        inputs["genomics"] = genomic
                    if "vision" in setting:
                        inputs.update(patch_features=features, patch_coords=coords)
                    if setting == "clinical_vision_genomics":
                        reuse[0] = True
                    result = model.predict(**inputs)
                    row[setting] = float(result["os_risk"].reshape(-1)[0])
                    if not np.isfinite(row[setting]):
                        raise ValueError("Nonfinite risk: " + slide)
                    if "vision" in setting:
                        row["num_views"] = result["num_views"]
                if cursor[0] != len(visual_cache):
                    raise ValueError("View cache mismatch")
            finally:
                model.core.ssl_vision_aggregator.forward = original
            cp.write_text(
                json.dumps({"feature_stat": feature_stat, "row": row}, indent=2)
            )
            print(
                f'[RESULT {i}/{len(records)}] {slide} views={row["num_views"]} '
                + " ".join(f"{s}={row[s]:.6f}" for s in SETTINGS)
                + f" elapsed={time.time()-start:.1f}s",
                flush=True,
            )
        slides.append(row)
        save_csv(out / "slide_predictions.csv", slides)
    groups = defaultdict(list)
    for row in slides:
        groups[(row["project"], row["case_id"])].append(row)
    patients = []
    for (project, case), values in groups.items():
        if len({(v["os_event"], v["os_time"]) for v in values}) != 1:
            raise ValueError("Inconsistent patient OS labels: " + case)
        patients.append(
            dict(
                project=project,
                case_id=case,
                n_slides=len(values),
                os_event=values[0]["os_event"],
                os_time=values[0]["os_time"],
                **{s: float(np.mean([v[s] for v in values])) for s in SETTINGS},
            )
        )
    save_csv(out / "patient_predictions.csv", patients)
    metrics = []
    for project in sorted({r["project"] for r in patients}):
        rows = [r for r in patients if r["project"] == project]
        indices = np.arange(len(rows))
        point = {s: cindex(rows, s, indices) for s in SETTINGS}
        rng = np.random.default_rng(args.seed)
        boot = {s: [] for s in SETTINGS}
        for _ in range(args.bootstrap):
            draw = rng.choice(indices, len(indices), replace=True)
            for s in SETTINGS:
                boot[s].append(cindex(rows, s, draw))
        for s in SETTINGS:
            vals = np.array(boot[s])
            valid = vals[np.isfinite(vals)]
            diff = vals - np.array(boot["clinical"])
            diff = diff[np.isfinite(diff)]
            ci = np.percentile(valid, [2.5, 97.5]) if len(valid) else [float("nan")] * 2
            di = (
                np.percentile(diff, [2.5, 97.5]) * 100
                if len(diff)
                else [float("nan")] * 2
            )
            row = dict(
                project=project,
                setting=s,
                n_patients=len(rows),
                events=sum(r["os_event"] for r in rows),
                cindex=point[s],
                ci_lower=ci[0],
                ci_upper=ci[1],
                delta_vs_clinical_pp=100 * (point[s] - point["clinical"]),
                delta_ci_lower_pp=di[0],
                delta_ci_upper_pp=di[1],
                valid_bootstraps=len(valid),
                smoke_test=bool(args.limit),
            )
            metrics.append(row)
            print("[METRIC]", row, flush=True)
    save_csv(out / "metrics.csv", metrics)
    print("Saved results to", out.resolve(), flush=True)
    return metrics


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
