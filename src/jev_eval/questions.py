"""Build the fan-out request: one state, five independent questions, one round trip.

State is the assay description alone. That is deliberate — it is what a ChEMBL curator
reads, and Jev loses accuracy when the state carries material the question does not
need. It also keeps the organism and tissue columns clean as gold labels for the
`names_organism` question rather than leaking them into the input.
"""

from __future__ import annotations

from typesafe_sdk import Choice, Noul, Score

from .data import ASSAY_TYPES, Assay
from .ontology import Term

# Ordered levels for target specificity. ChEMBL's own confidence_score is a taxonomy of
# target-assignment specificity rather than a true ordinal, so we do not try to predict
# it directly; we ask a genuinely ordered question and check rank correlation instead.
TARGET_SPECIFICITY = [
    "No molecular target is named. The assay reads out an effect in a whole animal, "
    "tissue or organism, such as biodistribution, survival or blood pressure.",
    "Only a broad biological system or cell type is named, such as 'in 1321N1 cells' "
    "or 'in rat liver microsomes', with no protein identified.",
    "A protein family, class or pathway is named, such as 'lipoxygenase' or "
    "'adenosine receptor', without identifying which specific member.",
    "One specific molecular target is named unambiguously, such as "
    "'A2 adenosine receptor' or 'platelet 12-lipoxygenase'.",
]


def build(assay: Assay, terms: dict[str, Term]) -> tuple[dict, dict]:
    """Return (state, questions) for one assay record."""
    state = {"assay_description": assay.description}

    # Option names are the ontology labels rather than the opaque BAO_ accession, and
    # each criterion is the ontology's own definition of that class. Jev reads text, so
    # "organism-based format" carries meaning that "BAO_0000218" does not.
    bao_criteria = {
        term.label: term.definition or term.label
        for term in (terms[curie] for curie in sorted(terms))
    }

    questions = {
        "bao_format": Choice(
            instructions=(
                "Which BioAssay Ontology assay format does `assay_description` describe? "
                "Choose the format of the biological system the assay is run in, not the "
                "kind of measurement taken. Pick the most specific format the description "
                "actually supports; choose 'assay format' only when the description gives "
                "no indication of the experimental system at all."
            ),
            criteria=bao_criteria,
        ),
        "assay_type": Choice(
            instructions=(
                "Which ChEMBL assay type does `assay_description` describe? Classify what "
                "the assay measures."
            ),
            criteria={name: definition for name, definition in ASSAY_TYPES.values()},
        ),
        "target_specificity": Score(
            instructions=(
                "How specifically does `assay_description` identify the molecular target "
                "the compound is acting on?"
            ),
            criteria=TARGET_SPECIFICITY,
        ),
        # Speculative: not a gold column, but it should track assay_type 'A' and the
        # organism-based format, so it is a cheap consistency probe on the same pass.
        "in_vivo": Noul(
            instructions="Is the assay in `assay_description` performed in a living animal?",
            criteria={
                "true": "Performed in an intact, living multicellular organism.",
                "false": "Performed in vitro, ex vivo, or on isolated cells, tissue, "
                         "protein or subcellular fractions.",
            },
        ),
        # Verifiable against the assay_organism column, which is never shown to the model.
        "names_organism": Noul(
            instructions=(
                "Does `assay_description` name the species the assay material came from, "
                "either by scientific or common name?"
            ),
            criteria={
                "true": "A species is named or clearly implied by a common name such as "
                        "'rat', 'mouse', 'human', 'dog' or 'rabbit'.",
                "false": "No species is named.",
            },
        ),
    }
    return state, questions


def label_to_curie(terms: dict[str, Term]) -> dict[str, str]:
    """Map the ontology label Jev returns back to its BAO accession."""
    return {term.label: term.curie for term in terms.values()}


def type_to_code() -> dict[str, str]:
    """Map the assay-type option text back to its single-letter ChEMBL code."""
    return {name: code for code, (name, _) in ASSAY_TYPES.items()}
