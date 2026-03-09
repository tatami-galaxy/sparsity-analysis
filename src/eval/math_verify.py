"""
Math answer extraction and equivalence checking using HuggingFace's math_verify.

Install: pip install math-verify[antlr4_13_2]
"""

from math_verify import parse, verify
from math_verify.parser.extraction_config import (
    ExprExtractionConfig,
    LatexExtractionConfig,
)

# For model outputs: prioritize \boxed{}, then fall back to other latex / expressions
PRED_EXTRACTION_CONFIG = [
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
]

# For gold answers: typically a clean latex string
GOLD_EXTRACTION_CONFIG = [
    LatexExtractionConfig(),
    ExprExtractionConfig(),
]


def parse_answer(text: str, is_gold: bool = False) -> list | None:
    """Parse a math answer string into a verified representation.

    Returns the parsed result or None if parsing fails.
    """
    config = GOLD_EXTRACTION_CONFIG if is_gold else PRED_EXTRACTION_CONFIG
    try:
        return parse(text, extraction_config=config)
    except Exception:
        return None


def is_equiv(pred: str, gold: str) -> bool:
    """Check if predicted and gold answers are mathematically equivalent."""
    gold_parsed = parse_answer(gold, is_gold=True)
    pred_parsed = parse_answer(pred, is_gold=False)
    if gold_parsed is None or pred_parsed is None:
        return False
    try:
        return verify(gold_parsed, pred_parsed)
    except Exception:
        return False
