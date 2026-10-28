#!/usr/bin/env python3
"""
Automated Quality Assurance & Validation Suite for Cross-Lingual Syntactic Bootstrapping (EN vs. RO).

Validates `data/benchmark_main.jsonl` and `data/sanity_check.jsonl` against all
psycholinguistic and structural integrity criteria:
  1. Zero unreplaced placeholders (`{`, `}`, `None`, `null`) in `context_text`, `target_valid`, `target_invalid`.
  2. Schema & Factorial Balance Verification (all 14 schema keys present, exceeding 11-key minimum;
     exact row-count symmetry across `en` vs `ro`, 3 verb classes, 7 conditions, and 2 evaluation types).
  3. Condition Distinctness (`word_scrambled != syntax_rich_informative`,
     `constituent_scrambled != syntax_rich_informative`, and in Romanian `bow_inflected != bow_lemmatized`
     on 100% of paired items, proving morphological lemmatization worked).
  4. Stray Function-Word Check on Bag-of-Words conditions (`bow_inflected`, `bow_lemmatized`).
  5. Unique `trial_id` and non-degenerate targets (`target_valid != target_invalid`).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REQUIRED_KEYS = [
    "trial_id",
    "language",
    "verb_mode",
    "verb_class",
    "template_id",
    "condition",
    "num_shots",
    "verb_lemma",
    "verb_surface",
    "noun_set_id",
    "eval_type",
    "context_text",
    "target_valid",
    "target_invalid",
]

EXPECTED_LANGUAGES = {"en", "ro"}
EXPECTED_VERB_CLASSES = {
    "mental_epistemic",
    "physical_ditransitive",
    "physical_transitive_causative",
}
EXPECTED_CONDITIONS = {
    "syntax_rich_informative",
    "syntax_rich_ambiguous",
    "cue_conflict",
    "constituent_scrambled",
    "word_scrambled",
    "bow_inflected",
    "bow_lemmatized",
}
EXPECTED_EVAL_TYPES = {
    "semantic_forced_choice",
    "held_out_frame",
}

FORBIDDEN_SUBSTRINGS = ["{", "}"]
FORBIDDEN_WORD_PATTERN = re.compile(r"\b(None|null)\b", re.IGNORECASE)

FORBIDDEN_BOW_FUNCTION_WORDS = {
    "en": {"the", "a", "an", "that", "to"},
    "ro": {"pe", "că", "să", "i-", "l-", "i-a", "l-a", "s-a", "a", "un", "o"},
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {exc}") from exc
    return rows


def pair_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """Key that uniquely identifies a trial across experimental conditions."""
    return (
        row["language"],
        row["verb_mode"],
        row["verb_class"],
        row["template_id"],
        row["num_shots"],
        row["verb_lemma"],
        row["noun_set_id"].split("_")[-1],  # 01..05 index
        row["eval_type"],
    )


def validate_split(name: str, path: Path) -> Tuple[bool, List[Tuple[str, bool, str]], Dict[str, Any]]:
    rows = load_jsonl(path)
    checks: List[Tuple[str, bool, str]] = []

    # Check 1: Non-empty & unique trial_id
    trial_ids = [r.get("trial_id", "") for r in rows]
    unique_ids = len(set(trial_ids)) == len(rows) and len(rows) > 0
    checks.append((
        f"[{name}] Unique trial_id across all rows",
        unique_ids,
        f"{len(set(trial_ids)):,}/{len(rows):,} unique trial_ids",
    ))

    # Check 2: Schema Verification (all 14 required keys present, >= 11 required keys)
    missing_key_rows = 0
    for r in rows:
        if not all(k in r for k in REQUIRED_KEYS):
            missing_key_rows += 1
    schema_ok = missing_key_rows == 0
    checks.append((
        f"[{name}] Schema completeness (all {len(REQUIRED_KEYS)} required keys >= 11 min)",
        schema_ok,
        f"{len(rows) - missing_key_rows:,}/{len(rows):,} rows valid",
    ))

    # Check 3: Zero unreplaced placeholders ({, }, None, null)
    placeholder_errors = 0
    empty_text_errors = 0
    identical_targets = 0
    for r in rows:
        for field in ("context_text", "target_valid", "target_invalid"):
            val = str(r.get(field, ""))
            if not val.strip():
                empty_text_errors += 1
            if any(sub in val for sub in FORBIDDEN_SUBSTRINGS) or FORBIDDEN_WORD_PATTERN.search(val):
                placeholder_errors += 1
        if r.get("target_valid") == r.get("target_invalid"):
            identical_targets += 1

    checks.append((
        f"[{name}] Zero unreplaced placeholders ({{, }}, None, null)",
        placeholder_errors == 0 and empty_text_errors == 0,
        f"placeholder_errors={placeholder_errors}, empty_fields={empty_text_errors}",
    ))
    checks.append((
        f"[{name}] Target distinctness (target_valid != target_invalid)",
        identical_targets == 0,
        f"identical_target_pairs={identical_targets}",
    ))

    # Check 4: Language, Verb Class, Condition, Eval Type Balance Verification
    by_lang = collections.Counter(r["language"] for r in rows)
    by_class = collections.Counter(r["verb_class"] for r in rows)
    by_cond = collections.Counter(r["condition"] for r in rows)
    by_eval = collections.Counter(r["eval_type"] for r in rows)

    lang_balanced = (
        set(by_lang.keys()) == EXPECTED_LANGUAGES
        and len(set(by_lang.values())) == 1
    )
    class_balanced = (
        set(by_class.keys()) == EXPECTED_VERB_CLASSES
        and len(set(by_class.values())) == 1
    )
    cond_balanced = (
        set(by_cond.keys()) == EXPECTED_CONDITIONS
        and len(set(by_cond.values())) == 1
    )
    eval_balanced = (
        set(by_eval.keys()) == EXPECTED_EVAL_TYPES
        and len(set(by_eval.values())) == 1
    )

    # Also check cross-tab balance: (language, verb_class, condition, eval_type)
    cell_counts = collections.Counter(
        (r["language"], r["verb_class"], r["condition"], r["eval_type"])
        for r in rows
    )
    cells_balanced = (
        len(cell_counts) == 2 * 3 * 7 * 2
        and len(set(cell_counts.values())) == 1
    )

    checks.append((
        f"[{name}] Language balance (en == ro)",
        lang_balanced,
        f"en={by_lang['en']:,}, ro={by_lang['ro']:,}",
    ))
    checks.append((
        f"[{name}] Verb class balance (3 classes equal)",
        class_balanced,
        ", ".join(f"{k}={v:,}" for k, v in sorted(by_class.items())),
    ))
    checks.append((
        f"[{name}] Condition balance (7 conditions equal)",
        cond_balanced,
        f"all 7 conditions have {next(iter(by_cond.values())):,} rows each",
    ))
    checks.append((
        f"[{name}] Evaluation type balance (2 eval_types equal)",
        eval_balanced,
        ", ".join(f"{k}={v:,}" for k, v in sorted(by_eval.items())),
    ))
    checks.append((
        f"[{name}] Full 84-cell factorial symmetry (lang x class x cond x eval)",
        cells_balanced,
        f"84/84 cells populated with {next(iter(cell_counts.values())):,} rows/cell",
    ))

    # Check 5: Condition Distinctness
    # Group rows by condition and pair_key
    by_cond_map: Dict[str, Dict[Tuple[Any, ...], str]] = collections.defaultdict(dict)
    for r in rows:
        by_cond_map[r["condition"]][pair_key(r)] = r["context_text"]

    # 5a: word_scrambled != syntax_rich_informative
    ws_pairs = 0
    ws_diffs = 0
    for k, info_ctx in by_cond_map["syntax_rich_informative"].items():
        ws_ctx = by_cond_map["word_scrambled"].get(k)
        if ws_ctx is not None:
            ws_pairs += 1
            if ws_ctx.lower() != info_ctx.lower():
                ws_diffs += 1
    ws_ok = ws_pairs > 0 and ws_diffs == ws_pairs
    checks.append((
        f"[{name}] Condition distinctness: word_scrambled != syntax_rich_informative",
        ws_ok,
        f"{ws_diffs:,}/{ws_pairs:,} paired contexts strictly differ",
    ))

    # 5b: constituent_scrambled != syntax_rich_informative
    cs_pairs = 0
    cs_diffs = 0
    for k, info_ctx in by_cond_map["syntax_rich_informative"].items():
        cs_ctx = by_cond_map["constituent_scrambled"].get(k)
        if cs_ctx is not None:
            cs_pairs += 1
            if cs_ctx.lower() != info_ctx.lower():
                cs_diffs += 1
    cs_ok = cs_pairs > 0 and cs_diffs == cs_pairs
    checks.append((
        f"[{name}] Condition distinctness: constituent_scrambled != syntax_rich_informative",
        cs_ok,
        f"{cs_diffs:,}/{cs_pairs:,} paired contexts strictly differ",
    ))

    # 5c: In Romanian, bow_inflected != bow_lemmatized
    ro_bow_pairs = 0
    ro_bow_diffs = 0
    for k, inf_ctx in by_cond_map["bow_inflected"].items():
        if k[0] != "ro":
            continue
        lem_ctx = by_cond_map["bow_lemmatized"].get(k)
        if lem_ctx is not None:
            ro_bow_pairs += 1
            if inf_ctx != lem_ctx:
                ro_bow_diffs += 1
    ro_bow_ok = ro_bow_pairs > 0 and ro_bow_diffs == ro_bow_pairs
    checks.append((
        f"[{name}] Condition distinctness (RO): bow_inflected != bow_lemmatized",
        ro_bow_ok,
        f"{ro_bow_diffs:,}/{ro_bow_pairs:,} RO paired contexts strictly differ",
    ))

    # Check 6: Zero stray function words in bow_inflected and bow_lemmatized
    stray_fn_count = 0
    for r in rows:
        if r["condition"] in ("bow_inflected", "bow_lemmatized"):
            lang = r["language"]
            forbidden_words = FORBIDDEN_BOW_FUNCTION_WORDS[lang]
            tokens = [t.strip(".") for t in r["context_text"].lower().split()]
            if any(t in forbidden_words for t in tokens):
                stray_fn_count += 1
    checks.append((
        f"[{name}] Zero function words/clitics in bow_inflected & bow_lemmatized",
        stray_fn_count == 0,
        f"stray_function_word_rows={stray_fn_count}",
    ))

    all_passed = all(passed for _, passed, _ in checks)
    stats = {
        "total_rows": len(rows),
        "by_language": dict(by_lang),
        "by_verb_class": dict(by_class),
        "by_condition": dict(by_cond),
        "by_eval_type": dict(by_eval),
        "by_num_shots": dict(collections.Counter(r["num_shots"] for r in rows)),
    }
    return all_passed, checks, stats


def print_summary_table(name: str, stats: Dict[str, Any]) -> None:
    print(f"\n--- Summary Breakdown: {name} (Total: {stats['total_rows']:,} rows) ---")
    print(f"  Languages  : " + ", ".join(f"{k}={v:,}" for k, v in sorted(stats["by_language"].items())))
    print(f"  Num Shots  : " + ", ".join(f"{k}-shot={v:,}" for k, v in sorted(stats["by_num_shots"].items())))
    print(f"  Eval Types : " + ", ".join(f"{k}={v:,}" for k, v in sorted(stats["by_eval_type"].items())))
    print("  Verb Classes:")
    for vc, count in sorted(stats["by_verb_class"].items()):
        print(f"    - {vc:<32} : {count:>6,} rows")
    print("  Experimental Conditions:")
    for cond, count in stats["by_condition"].items():
        print(f"    - {cond:<32} : {count:>6,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate benchmark_main.jsonl and sanity_check.jsonl."
    )
    parser.add_argument(
        "--main-file",
        type=Path,
        default=Path("data/benchmark_main.jsonl"),
        help="Path to main benchmark JSONL.",
    )
    parser.add_argument(
        "--sanity-file",
        type=Path,
        default=Path("data/sanity_check.jsonl"),
        help="Path to sanity check JSONL.",
    )
    args = parser.parse_args()

    print("=" * 96)
    print("CROSS-LINGUAL SYNTACTIC BOOTSTRAPPING BENCHMARK — AUTOMATED QA VALIDATOR")
    print("=" * 96)

    main_ok, main_checks, main_stats = validate_split("benchmark_main.jsonl", args.main_file)
    sanity_ok, sanity_checks, sanity_stats = validate_split("sanity_check.jsonl", args.sanity_file)

    all_checks = main_checks + sanity_checks
    for label, passed, detail in all_checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {label:<72} | {detail}")

    print_summary_table("benchmark_main.jsonl", main_stats)
    print_summary_table("sanity_check.jsonl", sanity_stats)

    overall_ok = main_ok and sanity_ok
    print("\n" + "=" * 96)
    if overall_ok:
        print("OVERALL VALIDATION RESULT: PASS (100% of integrity & psycholinguistic checks succeeded)")
        print("=" * 96)
        sys.exit(0)
    else:
        print("OVERALL VALIDATION RESULT: FAIL (One or more checks failed)")
        print("=" * 96)
        sys.exit(1)


if __name__ == "__main__":
    main()
