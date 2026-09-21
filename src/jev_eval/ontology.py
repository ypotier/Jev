"""Resolve BioAssay Ontology (BAO) classes to labels and definitions.

ChEMBL tags each assay with a BAO format class. We pull the human-readable label and
textual definition from EBI's Ontology Lookup Service so they can be handed to Jev as
Choice option criteria: the option name carries the term, the criterion carries the
ontology's own definition of it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

OLS = "https://www.ebi.ac.uk/ols4/api/ontologies/bao/terms"
CACHE = Path(__file__).resolve().parents[2] / "data" / "bao_terms.json"

# The BAO root class. ChEMBL curators fall back to it when they decline to commit to a
# specific format, which makes it this dataset's native "I am not sure" label.
ROOT = "BAO_0000019"


@dataclass(frozen=True)
class Term:
    curie: str        # BAO_0000218
    label: str        # organism-based format
    definition: str


def _fetch(curie: str) -> Term:
    obo_id = curie.replace("_", ":")
    url = f"{OLS}?obo_id={quote(obo_id)}"
    with urlopen(url, timeout=60) as response:
        payload = json.load(response)
    term = payload["_embedded"]["terms"][0]
    definition = (term.get("description") or [""])[0]
    return Term(curie=curie, label=term["label"], definition=definition)


def load_terms(curies: list[str]) -> dict[str, Term]:
    """Return {curie: Term}, cached on disk so we hit OLS once."""
    cached: dict[str, dict] = {}
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())

    missing = [c for c in curies if c not in cached]
    for curie in missing:
        print(f"resolving {curie} from OLS", file=sys.stderr)
        term = _fetch(curie)
        cached[curie] = {"label": term.label, "definition": term.definition}

    if missing:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cached, indent=2, sort_keys=True))

    return {c: Term(c, cached[c]["label"], cached[c]["definition"]) for c in curies}


def _fetch_ancestors(curie: str) -> list[str]:
    """Ancestor BAO classes for a term, from OLS.

    We follow the `hierarchicalAncestors` link on the term itself rather than building
    the double-encoded IRI by hand, which is easy to get subtly wrong.
    """
    obo_id = curie.replace("_", ":")
    with urlopen(f"{OLS}?obo_id={quote(obo_id)}", timeout=60) as response:
        term = json.load(response)["_embedded"]["terms"][0]
    href = (term.get("_links") or {}).get("hierarchicalAncestors", {}).get("href")
    if not href:
        return []  # a root class has no ancestors and no link
    with urlopen(href, timeout=60) as response:
        payload = json.load(response)
    ancestors = payload.get("_embedded", {}).get("terms", [])
    return [t["obo_id"].replace(":", "_") for t in ancestors
            if t.get("obo_id", "").startswith("BAO:")]


ANCESTOR_CACHE = Path(__file__).resolve().parents[2] / "data" / "bao_ancestors.json"


def load_ancestors(curies: list[str]) -> dict[str, list[str]]:
    """Return {curie: [ancestor curies]}, cached on disk."""
    cached: dict[str, list[str]] = {}
    if ANCESTOR_CACHE.exists():
        cached = json.loads(ANCESTOR_CACHE.read_text())
    missing = [c for c in curies if c not in cached]
    for curie in missing:
        print(f"resolving ancestors of {curie}", file=sys.stderr)
        cached[curie] = _fetch_ancestors(curie)
    if missing:
        ANCESTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ANCESTOR_CACHE.write_text(json.dumps(cached, indent=2, sort_keys=True))
    return {c: cached[c] for c in curies}
