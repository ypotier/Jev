"""Command line entry point.

    python -m jev_eval.cli baselines            # offline, no API key needed
    python -m jev_eval.cli dry-run              # show one real request payload
    python -m jev_eval.cli eval --per-class 100 # the actual Jev evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .baselines import majority, tfidf_logreg
from .data import load_assays
from .metrics import (
    backoff_analysis,
    hierarchical,
    by_curation_tier,
    calibration,
    classification,
    coverage_curve,
    specificity_correlation,
)
from .ontology import load_ancestors, load_terms
from .questions import build
from .sample import natural, split, stratified

RESULTS = Path(__file__).resolve().parents[2] / "results"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _load_env() -> None:
    """Read KEY=value lines from a local .env, without overriding the real environment."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _terms(assays):
    return load_terms(sorted({a.bao_format for a in assays}))


def _print_report(title: str, report) -> None:
    print(f"\n{title}")
    print(f"  accuracy {report.accuracy:.3f}   macro-F1 {report.macro_f1:.3f}")
    print(f"  {'label':<40} {'n':>5} {'prec':>6} {'rec':>6} {'f1':>6}")
    for label, scores in sorted(report.per_class.items(), key=lambda kv: -report.support.get(kv[0], 0)):
        n = report.support.get(label, 0)
        if not n:
            continue
        print(f"  {label:<40} {n:>5} {scores['precision']:>6.3f} "
              f"{scores['recall']:>6.3f} {scores['f1']:>6.3f}")


def cmd_baselines(args) -> None:
    assays = load_assays()
    train, test = split(assays, holdout=args.holdout, seed=args.seed)
    print(f"{len(train):,} train / {len(test):,} test  (natural distribution)")

    gold = [a.bao_format for a in test]
    _print_report("majority class", classification(gold, majority(train, test)))

    predicted, confidence = tfidf_logreg(train, test)
    _print_report("tf-idf + logistic regression", classification(gold, predicted))
    correct = [g == p for g, p in zip(gold, predicted)]
    print(f"  ECE {calibration(confidence, correct)['ece']:.3f}")


def cmd_dry_run(args) -> None:
    assays = load_assays()
    terms = _terms(assays)
    state, questions = build(assays[args.index], terms)
    payload = {
        "state": state,
        "model": args.model,
        "questions": {k: v.model_dump(exclude_none=True) for k, v in questions.items()},
    }
    body = json.dumps(payload, indent=2)
    print(body)
    print(f"\n~{len(body) // 4:,} tokens per request", file=sys.stderr)
    print(f"gold: bao_format={assays[args.index].bao_format} "
          f"assay_type={assays[args.index].assay_type} "
          f"curated_by={assays[args.index].curated_by}", file=sys.stderr)


def cmd_eval(args) -> None:
    _load_env()
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit(
            "TYPESAFE_API_KEY is not set. Either:\n"
            "  export TYPESAFE_API_KEY=ts-...\n"
            f"or write it to {ENV_FILE} as TYPESAFE_API_KEY=ts-..."
        )

    from .run import run  # imported here so `baselines` works without the SDK configured

    assays = load_assays()
    terms = _terms(assays)
    sample = (natural(assays, args.size, args.seed) if args.natural
              else stratified(assays, args.per_class, args.seed))
    print(f"evaluating {len(sample):,} records "
          f"({'natural distribution' if args.natural else 'stratified'})", file=sys.stderr)

    predictions = run(sample, terms, model=args.model, workers=args.workers,
                      cache_name=args.cache)
    scored = {p.chembl_id: p for p in predictions}
    paired = [(a, scored[a.chembl_id]) for a in sample if a.chembl_id in scored]
    if not paired:
        sys.exit("no predictions returned")
    assays_out, preds_out = [a for a, _ in paired], [p for _, p in paired]

    bao_gold = [a.bao_format for a in assays_out]
    bao_pred = [p.bao_format for p in preds_out]
    bao_report = classification(bao_gold, bao_pred)
    _print_report("jev — BAO format", bao_report)

    type_report = classification([a.assay_type for a in assays_out],
                                 [p.assay_type for p in preds_out])
    _print_report("jev — ChEMBL assay type", type_report)

    correct = [g == p for g, p in zip(bao_gold, bao_pred)]
    confidence = [p.bao_confidence for p in preds_out]
    calib = calibration(confidence, correct)

    print("\ncalibration (BAO format)")
    print(f"  ECE {calib['ece']:.3f}")
    print(f"  {'bin':<12}{'n':>6}{'mean conf':>11}{'accuracy':>10}")
    for row in calib["bins"]:
        print(f"  {row['bin']:<12}{row['n']:>6}{row['mean_confidence']:>11.3f}"
              f"{row['accuracy']:>10.3f}")

    print("\ncoverage (auto-accept the most confident X%)")
    print(f"  {'coverage':>9}{'accuracy':>10}{'to review':>11}")
    for row in coverage_curve(confidence, correct):
        print(f"  {row['coverage']:>9.2f}{row['accuracy']:>10.3f}{row['n_reviewed']:>11}")

    print("\nagreement by curation tier")
    for tier, stats in by_curation_tier(assays_out, preds_out).items():
        print(f"  {tier:<16} n={stats['n']:<6} agreement={stats['agreement']:.3f}")

    print("\ncurator back-off")
    for key, value in backoff_analysis(assays_out, preds_out).items():
        print(f"  {key}: {value if value is None else round(value, 4) if isinstance(value, float) else value}")

    print("\ntarget specificity vs ChEMBL confidence_score")
    correlation = specificity_correlation(assays_out, preds_out)
    rho = correlation["spearman"]
    print(f"  n={correlation['n']}  spearman={'n/a' if rho is None else f'{rho:.3f}'}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": len(paired),
        "bao": {"accuracy": bao_report.accuracy, "macro_f1": bao_report.macro_f1,
                "per_class": bao_report.per_class},
        "assay_type": {"accuracy": type_report.accuracy, "macro_f1": type_report.macro_f1},
        "calibration": calib,
        "coverage": coverage_curve(confidence, correct),
        "by_curation_tier": by_curation_tier(assays_out, preds_out),
        "backoff": backoff_analysis(assays_out, preds_out),
        "specificity_correlation": correlation,
        "input_tokens": sum(p.input_tokens for p in preds_out),
    }
    tag = f"natural{args.size}" if args.natural else f"per-class{args.per_class}"
    out = RESULTS / f"summary_{tag}_seed{args.seed}_n{len(paired)}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


def cmd_rescore(args) -> None:
    """Offline: re-score cached predictions with ontology-aware partial credit,
    against a TF-IDF baseline trained only on records outside the sample."""
    from .baselines import tfidf_logreg
    from .run import Prediction

    assays = load_assays()
    curies = sorted({a.bao_format for a in assays})
    terms, ancestors = load_terms(curies), load_ancestors(curies)
    sample = (natural(assays, args.size, args.seed) if args.natural
              else stratified(assays, args.per_class, args.seed))
    ids = {a.chembl_id for a in sample}

    cache = RESULTS / args.cache
    if not cache.exists():
        sys.exit(f"no cached predictions at {cache}; run `eval` first")
    cached = {}
    for line in cache.read_text().splitlines():
        if line.strip():
            payload = json.loads(line)
            cached[payload["chembl_id"]] = payload

    paired = [(a, cached[a.chembl_id]) for a in sample if a.chembl_id in cached]
    if not paired:
        sys.exit("no cached predictions overlap this sample")
    test = [a for a, _ in paired]
    gold = [a.bao_format for a in test]
    jev = [p["bao_format"] for _, p in paired]
    baseline, _ = tfidf_logreg([a for a in assays if a.chembl_id not in ids], test)

    print(f"{len(paired):,} matched records "
          f"(tf-idf trained on {len(assays) - len(ids):,} held-out records)\n")
    print(f"{'':<24}{'exact':>7}{'macro-F1':>10}{'on-branch':>11}{'graded':>8}")
    for name, predicted in (("tf-idf (supervised)", baseline), ("jev (zero-shot)", jev)):
        flat = classification(gold, predicted)
        graded = hierarchical(gold, predicted, ancestors)
        print(f"{name:<24}{flat.accuracy:>7.3f}{flat.macro_f1:>10.3f}"
              f"{graded['on_branch']:>11.3f}{graded['graded_score']:>8.3f}")

    print("\nerror structure")
    print(f"  {'':<14}{'jev':>14}{'tf-idf':>14}")
    jev_h, base_h = hierarchical(gold, jev, ancestors), hierarchical(gold, baseline, ancestors)
    for bucket in ("exact", "ancestor", "descendant", "sibling", "unrelated"):
        print(f"  {bucket:<14}{jev_h['counts'][bucket]:>6} ({jev_h['rates'][bucket]:.3f})"
              f"{base_h['counts'][bucket]:>6} ({base_h['rates'][bucket]:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="jev_eval")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("baselines", help="offline baselines, no API key needed")
    b.add_argument("--holdout", type=float, default=0.3)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_baselines)

    d = sub.add_parser("dry-run", help="print one request payload without sending it")
    d.add_argument("--index", type=int, default=3)
    d.add_argument("--model", default="jev-latest")
    d.set_defaults(func=cmd_dry_run)

    e = sub.add_parser("eval", help="run the evaluation against Jev")
    e.add_argument("--per-class", type=int, default=100,
                   help="records per BAO class when stratified")
    e.add_argument("--natural", action="store_true",
                   help="sample the real distribution instead of stratifying")
    e.add_argument("--size", type=int, default=1000, help="sample size when --natural")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--model", default="jev-latest")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--cache", default="predictions.jsonl")
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("rescore", help="ontology-aware re-scoring of cached predictions")
    r.add_argument("--per-class", type=int, default=100)
    r.add_argument("--natural", action="store_true")
    r.add_argument("--size", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--cache", default="predictions.jsonl")
    r.set_defaults(func=cmd_rescore)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
