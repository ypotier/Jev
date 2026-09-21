"""Baselines.

Without these the evaluation proves nothing. ChEMBL assay descriptions are formulaic,
so a bag-of-words model trained on a few thousand of them is a genuinely strong
competitor — and unlike Jev it costs nothing per call. The question is not whether Jev
can do the task; it is whether Jev is worth using over these.
"""

from __future__ import annotations

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .data import Assay


def majority(train: list[Assay], test: list[Assay], field: str = "bao_format") -> list[str]:
    """Predict the most frequent training label for everything.

    The floor. On the natural distribution this scores ~0.69 accuracy and ~0.08 macro-F1,
    which is exactly why the report leads with macro-F1.
    """
    most_common = Counter(getattr(a, field) for a in train).most_common(1)[0][0]
    return [most_common] * len(test)


def tfidf_logreg(train: list[Assay], test: list[Assay], field: str = "bao_format") -> tuple[list[str], list[float]]:
    """Word/char TF-IDF into logistic regression.

    Returns (predictions, max class probability) so it can be put through the same
    calibration and coverage analysis as Jev.
    """
    model = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit([a.description for a in train], [getattr(a, field) for a in train])
    descriptions = [a.description for a in test]
    predicted = list(model.predict(descriptions))
    confidence = [float(row.max()) for row in model.predict_proba(descriptions)]
    return predicted, confidence
