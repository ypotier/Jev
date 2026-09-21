"""Scoring.

Accuracy alone is misleading here (a majority-class guess scores ~0.69 on the natural
distribution), so the report leads with macro-F1 and per-class recall, then tests the
two things that are actually specific to a calibrated model: whether the confidence
means anything, and whether it predicts where the human curator hedged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .data import Assay
from .ontology import ROOT
from .run import Prediction


@dataclass
class ClassReport:
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    support: dict[str, int]


def classification(gold: list[str], predicted: list[str]) -> ClassReport:
    labels = sorted(set(gold) | set(predicted))
    per_class: dict[str, dict[str, float]] = {}
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        if tp + fn:  # only count classes that actually occur in the gold labels
            f1s.append(f1)
    accuracy = sum(g == p for g, p in zip(gold, predicted)) / len(gold)
    support = {label: gold.count(label) for label in labels}
    return ClassReport(accuracy, float(np.mean(f1s)), per_class, support)


def calibration(confidences: list[float], correct: list[bool], bins: int = 10) -> dict:
    """Reliability curve plus expected calibration error.

    This is the claim worth testing directly: within a bin of predictions that Jev
    reports at ~0.8 confidence, roughly 80% of them should be right.
    """
    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, ece = [], 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (conf > low) & (conf <= high) if low > 0 else (conf >= low) & (conf <= high)
        if not mask.any():
            continue
        mean_conf, mean_acc, n = conf[mask].mean(), hit[mask].mean(), int(mask.sum())
        rows.append({"bin": f"{low:.1f}-{high:.1f}", "n": n,
                     "mean_confidence": float(mean_conf), "accuracy": float(mean_acc)})
        ece += n / len(conf) * abs(mean_acc - mean_conf)
    return {"bins": rows, "ece": float(ece)}


def coverage_curve(confidences: list[float], correct: list[bool], steps: int = 20) -> list[dict]:
    """Accuracy as a function of how much you auto-accept.

    Sort by confidence, accept the top X%, and report accuracy on that slice. This is
    the operational curve: it tells you what review burden buys what error rate.
    """
    order = np.argsort(-np.asarray(confidences, dtype=float))
    hit = np.asarray(correct, dtype=float)[order]
    rows, seen = [], set()
    for step in range(1, steps + 1):
        cut = max(1, round(len(hit) * step / steps))
        if cut in seen:  # small samples collide on the same cut
            continue
        seen.add(cut)
        rows.append({
            "coverage": cut / len(hit),
            "accuracy": float(hit[:cut].mean()),
            "n_reviewed": len(hit) - cut,
        })
    return rows


def backoff_analysis(assays: list[Assay], predictions: list[Prediction]) -> dict:
    """Does Jev's confidence drop where the human curator also hedged?

    ChEMBL curators fall back to the BAO root class when they will not commit to a
    specific format. If Jev is calibrated in a way that matches human hesitation, its
    confidence should be lower on exactly those records.
    """
    hedged = [p.bao_confidence for a, p in zip(assays, predictions) if a.bao_format == ROOT]
    committed = [p.bao_confidence for a, p in zip(assays, predictions) if a.bao_format != ROOT]
    return {
        "n_curator_hedged": len(hedged),
        "mean_confidence_where_curator_hedged": float(np.mean(hedged)) if hedged else None,
        "n_curator_committed": len(committed),
        "mean_confidence_where_curator_committed": float(np.mean(committed)) if committed else None,
    }


def by_curation_tier(assays: list[Assay], predictions: list[Prediction]) -> dict:
    """Agreement split by who curated the record.

    Expert-curated records are the closest thing this corpus has to a clean gold
    standard; Autocuration records were themselves machine-assigned, so disagreement
    there is not necessarily Jev being wrong.
    """
    tiers: dict[str, list[bool]] = {}
    for assay, prediction in zip(assays, predictions):
        tiers.setdefault(assay.curated_by, []).append(assay.bao_format == prediction.bao_format)
    return {
        tier: {"n": len(hits), "agreement": float(np.mean(hits))}
        for tier, hits in sorted(tiers.items())
    }


def specificity_correlation(assays: list[Assay], predictions: list[Prediction]) -> dict:
    """Rank correlation between Jev's target-specificity Score and ChEMBL's own
    confidence_score.

    ChEMBL's 0-9 scale is a taxonomy of target-assignment specificity rather than a true
    ordinal, so we check rank agreement rather than trying to predict the value.
    """
    pairs = [(p.target_specificity, a.confidence_score)
             for a, p in zip(assays, predictions) if a.confidence_score >= 0]
    if len(pairs) < 3:
        return {"n": len(pairs), "spearman": None}
    jev, chembl = zip(*pairs)
    return {"n": len(pairs), "spearman": spearman(list(jev), list(chembl))}


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation, with average ranks for ties."""

    def rank(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        order = array.argsort()
        ranks = np.empty(len(array), dtype=float)
        ranks[order] = np.arange(len(array), dtype=float)
        # average the ranks within each group of tied values
        for value in np.unique(array):
            tied = array == value
            ranks[tied] = ranks[tied].mean()
        return ranks

    a, b = rank(xs), rank(ys)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator else 0.0


# Graded credit for a prediction that lands somewhere sensible in the ontology but not
# on the exact gold class. Ontology assignment is not a flat classification problem:
# naming a term's direct parent is a materially different kind of error from naming an
# unrelated branch, and a flat accuracy score cannot tell them apart.
CREDIT_EXACT = 1.0
CREDIT_ANCESTOR = 0.5    # correct but less specific than the curator
CREDIT_DESCENDANT = 0.5  # more specific than the curator; often defensible
CREDIT_SIBLING = 0.25    # same parent, wrong child


def hierarchical(
    gold: list[str],
    predicted: list[str],
    ancestors: dict[str, list[str]],
) -> dict:
    """Score predictions with partial credit for ontologically near misses."""
    parents = {curie: (chain[0] if chain else None) for curie, chain in ancestors.items()}

    buckets = {"exact": 0, "ancestor": 0, "descendant": 0, "sibling": 0, "unrelated": 0}
    total = 0.0
    for g, p in zip(gold, predicted):
        if g == p:
            buckets["exact"] += 1
            total += CREDIT_EXACT
        elif p in ancestors.get(g, []):
            buckets["ancestor"] += 1
            total += CREDIT_ANCESTOR
        elif g in ancestors.get(p, []):
            buckets["descendant"] += 1
            total += CREDIT_DESCENDANT
        elif parents.get(g) is not None and parents.get(g) == parents.get(p):
            buckets["sibling"] += 1
            total += CREDIT_SIBLING
        else:
            buckets["unrelated"] += 1

    n = len(gold)
    return {
        "n": n,
        "counts": buckets,
        "rates": {k: v / n for k, v in buckets.items()},
        "graded_score": total / n,
        # The headline number for curation: the prediction is on the right branch, so a
        # curator is refining a term rather than correcting a mistake.
        "on_branch": (buckets["exact"] + buckets["ancestor"] + buckets["descendant"]) / n,
    }
