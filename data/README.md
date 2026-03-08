# Data

MATH dataset preparation and loading utilities.

## Files

- `prepare_math.py` — Downloads MATH, stratifies by difficulty level, formats for each training method
- `formatting.py` — Prompt templates: standard prompts, teacher prompts (with/without trace), chat templates

## Data Splits

The MATH dataset is split and organized by difficulty level:

```
data/
├── math_level_1/  — Easiest (base model ~80%+ accuracy)
├── math_level_2/
├── math_level_3/  — Medium (base model ~50-60%)
├── math_level_4/
├── math_level_5/  — Hardest (base model ~20-35%)
└── math_all/      — Combined for primary experiments
```

Each split contains `train.jsonl` and `test.jsonl` with fields:
- `problem`: The math question
- `solution`: Full reasoning trace
- `answer`: Extracted final answer (boxed)
- `level`: Difficulty level (1-5)
- `subject`: Math subject area
