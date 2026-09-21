"""Run the evaluation against Jev, caching every answer to disk.

One request per assay record carries all five questions, which is the whole point of
the fan-out pattern: Jev ingests the state once and evaluates every question against it
in parallel, so five judgments cost barely more than one.
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from typesafe_sdk import RetryPolicy, TypeSafeClient

from .data import Assay
from .ontology import Term
from .questions import build, label_to_curie, type_to_code

RESULTS = Path(__file__).resolve().parents[2] / "results"


@dataclass
class Prediction:
    """Jev's answers for one assay, already mapped back to ChEMBL's vocabularies."""

    chembl_id: str
    bao_format: str                      # predicted BAO curie
    bao_confidence: float
    bao_probabilities: dict[str, float]  # keyed by ontology label
    assay_type: str                      # predicted ChEMBL code
    assay_type_confidence: float
    target_specificity: float
    in_vivo: float
    names_organism: float
    input_tokens: int

    def to_json(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_json(cls, payload: dict) -> "Prediction":
        return cls(**payload)


def _predict(client: TypeSafeClient, assay: Assay, terms: dict[str, Term], model: str) -> Prediction:
    state, questions = build(assay, terms)
    response = client.system_one(state=state, questions=questions, model=model)
    answers = response.answers

    bao = answers["bao_format"]
    assay_type = answers["assay_type"]
    return Prediction(
        chembl_id=assay.chembl_id,
        bao_format=label_to_curie(terms).get(bao.choice, bao.choice),
        bao_confidence=bao.confidence,
        bao_probabilities=bao.probabilities,
        assay_type=type_to_code().get(assay_type.choice, assay_type.choice),
        assay_type_confidence=assay_type.confidence,
        target_specificity=answers["target_specificity"].score,
        in_vivo=answers["in_vivo"].noul,
        names_organism=answers["names_organism"].noul,
        input_tokens=response.usage.input_tokens or 0,
    )


def run(
    assays: list[Assay],
    terms: dict[str, Term],
    *,
    model: str = "jev-latest",
    workers: int = 8,
    cache_name: str = "predictions.jsonl",
) -> list[Prediction]:
    """Predict every assay, skipping any already present in the cache file."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache_path = RESULTS / cache_name

    done: dict[str, Prediction] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                prediction = Prediction.from_json(json.loads(line))
                done[prediction.chembl_id] = prediction

    todo = [a for a in assays if a.chembl_id not in done]
    if not todo:
        print(f"all {len(assays)} records already cached", file=sys.stderr)
        return [done[a.chembl_id] for a in assays]

    print(f"{len(done)} cached, requesting {len(todo)}", file=sys.stderr)
    lock = threading.Lock()
    failures = 0

    with TypeSafeClient(retry=RetryPolicy(max_retries=5)) as client, \
            cache_path.open("a") as sink, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_predict, client, a, terms, model): a for a in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            assay = futures[future]
            try:
                prediction = future.result()
            except Exception as error:  # keep going; a partial run is still analysable
                failures += 1
                print(f"  {assay.chembl_id} failed: {error}", file=sys.stderr)
                continue
            done[assay.chembl_id] = prediction
            with lock:
                sink.write(json.dumps(prediction.to_json()) + "\n")
                sink.flush()
            if index % 100 == 0:
                print(f"  {index}/{len(todo)}", file=sys.stderr)

    if failures:
        print(f"{failures} request(s) failed; rerun to retry them", file=sys.stderr)

    fresh = sum(done[a.chembl_id].input_tokens for a in todo if a.chembl_id in done)
    total = sum(p.input_tokens for p in done.values())
    print(f"this run: {fresh:,} input tokens (~${fresh / 1e6 * 0.042:.4f})", file=sys.stderr)
    print(f"cache total: {total:,} input tokens (~${total / 1e6 * 0.042:.4f})", file=sys.stderr)
    return [done[a.chembl_id] for a in assays if a.chembl_id in done]
