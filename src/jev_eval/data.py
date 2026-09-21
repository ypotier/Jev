"""Load the matchbench/ChEMBL-SM corpus.

The dataset ships as two CSV halves of the same ChEMBL assay table, column-prefixed
`0_` and `1_`. Its advertised task is schema matching, but `matches.txt` is the
identity map over identical column names, so we ignore it and use the CSVs as a
corpus of curator-labelled assay records.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

REPO = "https://huggingface.co/datasets/matchbench/ChEMBL-SM/resolve/main"
HALVES = {"chembl-sm-1.csv": "0_", "chembl-sm-2.csv": "1_"}

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# ChEMBL assay_type codes: code -> (option name, definition).
# The option name is what Jev returns, so it stays short and meaningful; the definition
# goes in the criterion.
ASSAY_TYPES = {
    "B": ("Binding", "Measures direct binding of the compound to a molecular target, "
                     "such as a Ki, Kd or displacement of a labelled ligand."),
    "F": ("Functional", "Measures a biological or biochemical effect, such as enzyme "
                        "inhibition, receptor activation or a cellular response, rather "
                        "than binding itself."),
    "A": ("ADMET", "Measures absorption, distribution, metabolism, excretion or "
                   "toxicity, typically in an animal, tissue, plasma or microsome "
                   "preparation. Includes biodistribution and pharmacokinetics."),
    "P": ("Physicochemical", "Measures a property of the compound itself, such as "
                             "solubility, logP, pKa or chemical stability, with no "
                             "biological system involved."),
    "U": ("Unclassified", "Does not fit any of the other categories."),
}

@dataclass(frozen=True)
class Assay:
    """One ChEMBL assay record with its curator-assigned labels."""

    chembl_id: str
    description: str
    bao_format: str        # BioAssay Ontology class, e.g. "BAO_0000218"
    assay_type: str        # B / F / A / P / U
    relationship_type: str
    confidence_score: int  # ChEMBL target-assignment confidence, 0-9
    curated_by: str        # Autocuration / Intermediate / Expert
    assay_organism: str
    assay_tissue: str
    assay_cell_type: str

    @property
    def names_organism(self) -> bool:
        return self.assay_organism not in ("", "nan")


def _download(name: str) -> Path:
    path = DATA_DIR / name
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{REPO}/{name}"
    print(f"downloading {url}", file=sys.stderr)
    with urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())
    return path


def _clean(value: str) -> str:
    """ChEMBL-SM writes missing values as the literal string 'nan'."""
    value = (value or "").strip()
    return "" if value == "nan" else value


def load_assays(deduplicate: bool = True) -> list[Assay]:
    """Return every assay record that carries a description and a BAO label.

    Underscores stand in for spaces throughout the CSV, so we restore them.

    The two CSV halves deliberately overlap: this is published as a schema-matching
    benchmark, so the second table repeats assays from the first in order to give the
    matcher something to match. 7,500 of the 14,997 assays appear twice. For our purpose
    that overlap is pure contamination — it double-bills Jev, double-counts records in
    the metrics, and puts the same assay text in both halves of a supervised baseline's
    train/test split. We deduplicate on `chembl_id` by default; pass
    ``deduplicate=False`` to see the raw rows.
    """
    assays: list[Assay] = []
    for name, prefix in HALVES.items():
        with _download(name).open(newline="") as handle:
            for row in csv.DictReader(handle):
                record = {key.removeprefix(prefix): _clean(value) for key, value in row.items()}
                description = record["description"].replace("_", " ").strip()
                if not description or not record["bao_format"]:
                    continue
                try:
                    confidence = int(float(record["confidence_score"]))
                except ValueError:
                    confidence = -1
                assays.append(
                    Assay(
                        chembl_id=record["chembl_id"],
                        description=description,
                        bao_format=record["bao_format"],
                        assay_type=record["assay_type"] or "U",
                        relationship_type=record["relationship_type"],
                        confidence_score=confidence,
                        curated_by=record["curated_by"],
                        assay_organism=record["assay_organism"].replace("_", " "),
                        assay_tissue=record["assay_tissue"].replace("_", " "),
                        assay_cell_type=record["assay_cell_type"].replace("_", " "),
                    )
                )

    if deduplicate:
        seen: dict[str, Assay] = {}
        for assay in assays:
            seen.setdefault(assay.chembl_id, assay)
        assays = list(seen.values())
    return assays
