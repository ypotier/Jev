"""Stratified sampling.

The corpus is severely imbalanced: ~69% of records are organism-based format, so a
majority-class classifier scores ~0.69 accuracy and raw accuracy tells you nothing.
We sample evenly across BAO classes (capped by what each class actually has) so that
macro-averaged metrics are meaningful, and keep the natural-distribution sample
available for the coverage/routing analysis, which only makes sense in situ.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .data import Assay


def stratified(assays: list[Assay], per_class: int, seed: int = 0) -> list[Assay]:
    """Up to `per_class` records per BAO class, sampled without replacement."""
    by_class: dict[str, list[Assay]] = defaultdict(list)
    for assay in assays:
        by_class[assay.bao_format].append(assay)

    rng = random.Random(seed)
    picked: list[Assay] = []
    for label in sorted(by_class):
        bucket = by_class[label]
        picked.extend(rng.sample(bucket, min(per_class, len(bucket))))
    rng.shuffle(picked)
    return picked


def natural(assays: list[Assay], size: int, seed: int = 0) -> list[Assay]:
    """A plain random sample that preserves the real class distribution."""
    rng = random.Random(seed)
    return rng.sample(assays, min(size, len(assays)))


def split(assays: list[Assay], holdout: float = 0.5, seed: int = 0) -> tuple[list[Assay], list[Assay]]:
    """Train/test split for the supervised baselines.

    Jev never trains, so it only ever sees the test half; the split exists so the
    TF-IDF baseline is scored on the same records under the same conditions.
    """
    shuffled = list(assays)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - holdout))
    return shuffled[:cut], shuffled[cut:]
