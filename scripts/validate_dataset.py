#!/usr/bin/env python3
"""
Automated Quality Assurance & Validation Suite for Cross-Lingual Syntactic Bootstrapping (EN vs. RO).

Validates `data/benchmark_main.jsonl` and `data/sanity_check.jsonl` against all
psycholinguistic and structural integrity criteria:
  1. Zero unreplaced placeholders (`{`, `}`, `None`, `null`) in `context_text`, `target_valid`, `target_invalid`.
  2. Schema & Factorial Balance Verification (all 14 schema keys present, exceeding 11-key minimum;
     exact row-count symmetry across `en` vs `ro`, 3 verb classes, 11 conditions, and 2 evaluation types).
  3. Condition Distinctness (`word_scrambled != syntax_rich_informative`,
     `constituent_scrambled != syntax_rich_informative`, `bow_inflected != bow_lemmatized` in RO,
     and the exact same guarantees across all `_ambiguous` counterparts + informative vs ambiguous sets).
  4. Stray Function-Word Check on all 4 Bag-of-Words conditions (`bow_inflected`, `bow_lemmatized`,
     and their `_ambiguous` variants).
  5. Unique `trial_id` and non-degenerate targets (`target_valid != target_invalid`).
  6. Exact Symmetric Mirror-Pair Counterbalancing (for `benchmark_main`) and Verb-Lemma Anchoring
     + zero control-verb stem leakage inside `semantic_forced_choice`.
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
    "constituent_scrambled_ambiguous",
    "word_scrambled",
    "word_scrambled_ambiguous",
    "bow_inflected",
    "bow_inflected_ambiguous",
    "bow_lemmatized",
    "bow_lemmatized_ambiguous",
}
BOW_CONDITIONS = {
    "bow_inflected",
    "bow_inflected_ambiguous",
    "bow_lemmatized",
    "bow_lemmatized_ambiguous",
}
EXPECTED_EVAL_TYPES = {
    "semantic_forced_choice",
    "held_out_frame",
}

CONTROL_VERB_FORBIDDEN_DEFINITIONAL_STEMS = {
    "en": re.compile(r"\b(think|thinks|thought|believe|believes|believed|give|gives|gave|send|sends|sent|push|pushes|pushed|hit|hits|hitting)\b", re.IGNORECASE),
    "ro": re.compile(r"\b(crede|crezut|creadă|afirma|afirmă|afirmat|afirme|oferi|oferă|oferit|ofere|trimite|trimis|trimită|trimiterea|împinge|împins|împingă|împingerea|lovi|lovește|lovit|lovească|lovitura|lovituri)\b", re.IGNORECASE),
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


def _check_pair_distinctness(
    by_cond_map: Dict[str, Dict[Tuple[Any, ...], str]],
    cond_a: str,
    cond_b: str,
    lang_filter: str | None = None,
    case_insensitive: bool = True,
) -> Tuple[bool, str]:
    """Compare paired context strings across two conditions on all shared pair_keys."""
    pairs = 0
    diffs = 0
    for k, ctx_b in by_cond_map[cond_b].items():
        if lang_filter is not None and k[0] != lang_filter:
            continue
        ctx_a = by_cond_map[cond_a].get(k)
        if ctx_a is not None:
            pairs += 1
            a_cmp = ctx_a.lower() if case_insensitive else ctx_a
            b_cmp = ctx_b.lower() if case_insensitive else ctx_b
            if a_cmp != b_cmp:
                diffs += 1
    ok = pairs > 0 and diffs == pairs
    return ok, f"{diffs:,}/{pairs:,} paired contexts strictly differ"


def _check_mirror_symmetry(rows: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Verify exact 1-to-1 (valid, invalid) <-> (invalid, valid) mirror balance across classes."""
    grouped: Dict[Tuple[Any, ...], List[Tuple[str, str]]] = collections.defaultdict(list)
    for r in rows:
        noun_idx = r["noun_set_id"].split("_")[-1]
        gkey = (r["language"], r["condition"], r["num_shots"], r["verb_lemma"], noun_idx, r["eval_type"])
        grouped[gkey].append((r["target_valid"], r["target_invalid"]))

    unmatched = 0
    total_groups = 0
    for pairs in grouped.values():
        total_groups += 1
        counts = collections.Counter(pairs)
        for (v, inv), c in counts.items():
            if counts.get((inv, v), 0) != c:
                unmatched += c
    return unmatched == 0 and total_groups > 0, f"{total_groups:,} groups, unmatched_rows={unmatched}"


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
    expected_cells = 2 * 3 * len(EXPECTED_CONDITIONS) * 2
    cell_counts = collections.Counter(
        (r["language"], r["verb_class"], r["condition"], r["eval_type"])
        for r in rows
    )
    cells_balanced = (
        len(cell_counts) == expected_cells
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
        f"[{name}] Condition balance ({len(EXPECTED_CONDITIONS)} conditions equal)",
        cond_balanced,
        f"all {len(EXPECTED_CONDITIONS)} conditions have {next(iter(by_cond.values())):,} rows each",
    ))
    checks.append((
        f"[{name}] Evaluation type balance (2 eval_types equal)",
        eval_balanced,
        ", ".join(f"{k}={v:,}" for k, v in sorted(by_eval.items())),
    ))
    checks.append((
        f"[{name}] Full {expected_cells}-cell factorial symmetry (lang x class x cond x eval)",
        cells_balanced,
        f"{len(cell_counts)}/{expected_cells} cells populated with {next(iter(cell_counts.values())):,} rows/cell",
    ))

    # Check 5: Condition Distinctness across informative & ambiguous ablations
    by_cond_map: Dict[str, Dict[Tuple[Any, ...], str]] = collections.defaultdict(dict)
    for r in rows:
        by_cond_map[r["condition"]][pair_key(r)] = r["context_text"]

    distinctness_specs = [
        ("word_scrambled", "syntax_rich_informative", None, True),
        ("constituent_scrambled", "syntax_rich_informative", None, True),
        ("bow_inflected", "bow_lemmatized", "ro", False),
        ("word_scrambled_ambiguous", "syntax_rich_ambiguous", None, True),
        ("constituent_scrambled_ambiguous", "syntax_rich_ambiguous", None, True),
        ("bow_inflected_ambiguous", "bow_lemmatized_ambiguous", "ro", False),
        ("syntax_rich_ambiguous", "syntax_rich_informative", None, True),
        ("constituent_scrambled_ambiguous", "constituent_scrambled", None, True),
        ("word_scrambled_ambiguous", "word_scrambled", None, True),
        ("bow_inflected_ambiguous", "bow_inflected", None, True),
        ("bow_lemmatized_ambiguous", "bow_lemmatized", None, True),
    ]
    for cond_a, cond_b, lang_filter, ci in distinctness_specs:
        ok, detail = _check_pair_distinctness(by_cond_map, cond_a, cond_b, lang_filter, ci)
        scope = f" ({lang_filter.upper()})" if lang_filter else ""
        checks.append((
            f"[{name}] Condition distinctness{scope}: {cond_a} != {cond_b}",
            ok,
            detail,
        ))

    # Check 6: Zero stray function words across all 4 Bag-of-Words conditions
    stray_fn_count = 0
    for r in rows:
        if r["condition"] in BOW_CONDITIONS:
            lang = r["language"]
            forbidden_words = FORBIDDEN_BOW_FUNCTION_WORDS[lang]
            tokens = [t.strip(".") for t in r["context_text"].lower().split()]
            if any(t in forbidden_words for t in tokens):
                stray_fn_count += 1
    checks.append((
        f"[{name}] Zero function words/clitics across all 4 BoW conditions",
        stray_fn_count == 0,
        f"stray_function_word_rows={stray_fn_count}",
    ))

    # Check 7: Semantic forced-choice verb-lemma anchoring & zero control-verb stem leakage
    missing_anchor = 0
    stem_leaks = 0
    for r in rows:
        if r["eval_type"] != "semantic_forced_choice":
            continue
        lemma = r["verb_lemma"].lower()
        lang = r["language"]
        stem_pat = CONTROL_VERB_FORBIDDEN_DEFINITIONAL_STEMS[lang]
        for field in ("target_valid", "target_invalid"):
            text = r[field]
            tokens = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
            if lemma not in tokens:
                missing_anchor += 1
            # Inspect definitional body after masking any occurrences of the target verb_lemma itself
            masked = re.sub(rf"\b{re.escape(lemma)}\b", " ", text, flags=re.IGNORECASE)
            if stem_pat.search(masked):
                stem_leaks += 1

    checks.append((
        f"[{name}] Semantic verb-lemma anchoring (lemma in both targets)",
        missing_anchor == 0,
        f"missing_anchor_targets={missing_anchor}",
    ))
    checks.append((
        f"[{name}] Zero control-verb stem leakage in semantic definitions",
        stem_leaks == 0,
        f"control_stem_leak_targets={stem_leaks}",
    ))

    # Check 8: Exact 1-to-1 mirror-pair symmetry on pseudoverb split (50.0% no-context chance baseline)
    if all(r["verb_mode"] == "pseudoverb" for r in rows):
        mirror_ok, mirror_detail = _check_mirror_symmetry(rows)
        checks.append((
            f"[{name}] Exact 1-to-1 mirror symmetry (50.0% no-context baseline)",
            mirror_ok,
            mirror_detail,
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


def validate_pseudoverb_audit(
    audit_path: Path,
    main_path: Path,
) -> List[Tuple[str, bool, str]]:
    """Verify all EN & RO pseudoverbs in benchmark_main.jsonl pass Wuggy + Appendix B.3 audit."""
    if not audit_path.exists():
        return [
            (
                "[audit] Wuggy Appendix B.3 pseudoverb audit file exists",
                False,
                f"Missing {audit_path}",
            )
        ]

    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    main_rows = load_jsonl(main_path)
    dataset_lemmas = {
        lang: {r["verb_lemma"].lower() for r in main_rows if r["language"] == lang}
        for lang in ("en", "ro")
    }

    checks: List[Tuple[str, bool, str]] = []
    certified = audit.get("certified_v2_1", {})
    for lang in ("en", "ro"):
        entries = certified.get(lang, [])
        all_passed = len(entries) == 10 and all(
            e.get("passed")
            and all(e.get(f"rule{i}_ok") for i in range(1, 6))
            and e.get("min_common_edit_dist", 0) >= 2
            and not e.get("substring_5plus_hits")
            and e.get("min_ctrl_stem_dist", 0) >= 2
            and e.get("min_ctrl_past_dist", 0) >= 3
            for e in entries
        )
        certified_lemmas = {e["forms"]["lemma"].lower() for e in entries}
        matches_dataset = certified_lemmas == dataset_lemmas[lang]
        checks.append((
            f"[audit] {lang.upper()} 10/10 pseudoverbs pass Wuggy + B.3 checks",
            all_passed,
            f"certified={len(entries)}, passed={sum(1 for e in entries if e.get('passed'))}",
        ))
        checks.append((
            f"[audit] {lang.upper()} dataset pseudoverbs match certified B.3 set",
            matches_dataset,
            f"dataset_lemmas={sorted(dataset_lemmas[lang])}",
        ))
    return checks


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
    parser.add_argument(
        "--audit-file",
        type=Path,
        default=Path("data/pseudoverb_wuggy_audit.json"),
        help="Path to Wuggy Appendix B.3 pseudoverb audit JSON.",
    )
    args = parser.parse_args()

    print("=" * 96)
    print("CROSS-LINGUAL SYNTACTIC BOOTSTRAPPING BENCHMARK — AUTOMATED QA VALIDATOR")
    print("=" * 96)

    main_ok, main_checks, main_stats = validate_split("benchmark_main.jsonl", args.main_file)
    sanity_ok, sanity_checks, sanity_stats = validate_split("sanity_check.jsonl", args.sanity_file)
    audit_checks = validate_pseudoverb_audit(args.audit_file, args.main_file)
    audit_ok = all(passed for _, passed, _ in audit_checks)

    all_checks = main_checks + sanity_checks + audit_checks
    for label, passed, detail in all_checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {label:<72} | {detail}")

    print_summary_table("benchmark_main.jsonl", main_stats)
    print_summary_table("sanity_check.jsonl", sanity_stats)

    overall_ok = main_ok and sanity_ok and audit_ok
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
