#!/usr/bin/env python3
"""
Deterministic Benchmark Generator for Cross-Lingual Syntactic Bootstrapping (EN vs. RO).

Combines `configs/lexicon_en_ro.json` and `configs/templates_en_ro.json` to generate:
  - `data/benchmark_main.jsonl` (pseudoverb evaluation across 7 conditions, 4-shot & 1-shot)
  - `data/sanity_check.jsonl` (real control verb ceiling evaluation across all 7 conditions)
  - `data/dataset_summary_stats.json` (comprehensive counts and breakdown statistics)
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_SEED = 42

CONDITIONS = [
    "syntax_rich_informative",
    "syntax_rich_ambiguous",
    "cue_conflict",
    "constituent_scrambled",
    "word_scrambled",
    "bow_inflected",
    "bow_lemmatized",
]

VERB_CLASSES = [
    "mental_epistemic",
    "physical_ditransitive",
    "physical_transitive_causative",
]

EVAL_TYPES = [
    "semantic_forced_choice",
    "held_out_frame",
]


def capitalize_sentence(text: str) -> str:
    """Capitalize the first character of a sentence while preserving the rest."""
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def deterministic_shuffle_not_equal(items: List[str], rng: random.Random) -> List[str]:
    """
    Deterministically shuffle a list of tokens using `rng`, guaranteeing that the
    resulting order is strictly different from the input order whenever possible.
    """
    if len(items) <= 1 or len(set(items)) <= 1:
        return list(items)
    shuffled = list(items)
    for _ in range(25):
        rng.shuffle(shuffled)
        if shuffled != items:
            return shuffled
    # Deterministic fallback rotation if random draws matched original
    return items[1:] + items[:1]


def build_slot_values(
    lang: str,
    verb_entry: Dict[str, str],
    noun_set: Dict[str, Any],
    slot_mapping: Dict[str, int],
) -> Dict[str, str]:
    """
    Resolve all morphological slot placeholders for a single exposure shot.
    """
    shots_pool = noun_set["shots"]
    np1 = shots_pool[slot_mapping["np1_shot_index"]][lang]["np1"]
    np2 = shots_pool[slot_mapping["np2_shot_index"]][lang]["np2"]
    np3 = shots_pool[slot_mapping["np3_shot_index"]][lang]["np3"]
    emb_v = shots_pool[slot_mapping["emb_verb_shot_index"]][lang]["emb_verb"]

    if lang == "en":
        return {
            "np1_det_def": np1["det_def"],
            "np1_bare": np1["bare"],
            "np1_to_dat": np1["to_dat"],
            "np1_lemma": np1["lemma"],
            "np2_det_def": np2["det_def"],
            "np2_bare": np2["bare"],
            "np2_to_dat": np2["to_dat"],
            "np2_lemma": np2["lemma"],
            "np3_det_def": np3["det_def"],
            "np3_det_indef": np3["det_indef"],
            "np3_bare": np3["bare"],
            "np3_lemma": np3["lemma"],
            "emb_verb_past": emb_v["past"],
            "emb_verb_lemma": emb_v["lemma"],
            "verb_lemma": verb_entry["lemma"],
            "verb_past": verb_entry["past"],
            "verb_pres_3sg": verb_entry["pres_3sg"],
            "verb_participle": verb_entry["participle"],
        }
    else:
        return {
            "np1_nom_def": np1["nom_def"],
            "np1_acc_pe": np1["acc_pe"],
            "np1_acc_bare": np1["acc_bare"],
            "np1_dat_def": np1["dat_def"],
            "np1_lemma": np1["lemma"],
            "np2_nom_def": np2["nom_def"],
            "np2_acc_pe": np2["acc_pe"],
            "np2_acc_bare": np2["acc_bare"],
            "np2_dat_def": np2["dat_def"],
            "np2_lemma": np2["lemma"],
            "np3_nom_acc_def": np3["nom_acc_def"],
            "np3_acc_indef": np3["acc_indef"],
            "np3_lemma": np3["lemma"],
            "emb_verb_past_comp": emb_v["past_comp"],
            "emb_verb_participle": emb_v["participle"],
            "emb_verb_lemma": emb_v["lemma"],
            "verb_lemma": verb_entry["lemma"],
            "verb_past_part": verb_entry["past_part"],
            "verb_pres_3sg": verb_entry["pres_3sg"],
            "verb_subj_3sg": verb_entry["subj_3sg"],
            "verb_participle": verb_entry["participle"],
        }


def render_shot_sentence(
    lang: str,
    condition: str,
    shot_spec: Dict[str, Any],
    slot_values: Dict[str, str],
    rng_seed_str: str,
) -> str:
    """
    Deterministically construct a single exposure sentence for any of the 7 conditions.
    """
    constituents = [c.format(**slot_values) for c in shot_spec["constituents"]]
    canonical_sentence = capitalize_sentence(" ".join(constituents)) + "."

    if condition in ("syntax_rich_informative", "syntax_rich_ambiguous", "cue_conflict"):
        return canonical_sentence

    if condition == "constituent_scrambled":
        order = shot_spec["scrambled_constituent_order"]
        reordered = [constituents[idx] for idx in order]
        return capitalize_sentence(" ".join(reordered)) + "."

    if condition == "word_scrambled":
        # Tokenize canonical sentence into lowercase words (stripping final period)
        words = " ".join(constituents).lower().split()
        rng = random.Random(f"word_scrambled::{rng_seed_str}")
        shuffled_words = deterministic_shuffle_not_equal(words, rng)
        return " ".join(shuffled_words) + "."

    if condition in ("bow_inflected", "bow_lemmatized"):
        inflected_tokens = [
            tok.format(**slot_values).lower() for tok in shot_spec["bow_inflected_slots"]
        ]
        lemmatized_tokens = [
            tok.format(**slot_values).lower() for tok in shot_spec["bow_lemmatized_slots"]
        ]
        # Use identical deterministic index permutation for bow_inflected and bow_lemmatized
        # so the contrast isolates morphological suffixes without word-order noise.
        indices = list(range(len(inflected_tokens)))
        rng = random.Random(f"bow_order::{rng_seed_str}")
        perm = deterministic_shuffle_not_equal([str(i) for i in indices], rng)
        perm_idx = [int(i) for i in perm]

        if condition == "bow_inflected":
            chosen = [inflected_tokens[i] for i in perm_idx]
        else:
            chosen = [lemmatized_tokens[i] for i in perm_idx]
        return " ".join(chosen) + "."

    raise ValueError(f"Unknown condition: {condition}")


def select_noun_set(
    lexicon: Dict[str, Any],
    verb_class: str,
    condition: str,
    noun_idx: int,
) -> Dict[str, Any]:
    """
    Select the appropriate noun set for a given verb class, condition, and index (0..4).
    """
    class_sets = lexicon["noun_sets"][verb_class]
    if condition == "syntax_rich_ambiguous":
        return class_sets["ambiguous"][noun_idx]
    elif condition == "cue_conflict":
        return class_sets["conflict"][noun_idx]
    else:
        return class_sets["informative"][noun_idx]


def generate_rows_for_mode(
    lexicon: Dict[str, Any],
    templates: Dict[str, Any],
    verb_mode: str,
) -> List[Dict[str, Any]]:
    """
    Generate dataset rows for either `pseudoverb` (`benchmark_main.jsonl`)
    or `real_control` (`sanity_check.jsonl`).
    """
    rows: List[Dict[str, Any]] = []

    for lang in ["en", "ro"]:
        for verb_class in VERB_CLASSES:
            class_templates = templates["templates"][lang][verb_class]

            if verb_mode == "pseudoverb":
                verbs_pool = lexicon["pseudoverbs"][lang]
                # 4-shot across all 10 pseudoverb/noun instantiations;
                # 1-shot ablation across 5 instantiations
                shot_configs = [(4, 10), (1, 5)]
            else:
                verbs_pool = lexicon["real_control_verbs"][verb_class][lang]
                # Real control verbs: 2 control verbs per class, evaluated in 4-shot
                shot_configs = [(4, 2)]

            for template_item in class_templates:
                template_id = template_item["template_id"]
                item_short = template_id.replace("_", "")

                for condition in CONDITIONS:
                    for num_shots, num_instantiations in shot_configs:
                        for inst_idx in range(num_instantiations):
                            verb_entry = verbs_pool[inst_idx % len(verbs_pool)]
                            noun_idx = inst_idx % 5
                            noun_set = select_noun_set(lexicon, verb_class, condition, noun_idx)
                            noun_set_id = noun_set["id"]

                            # Build context sentences for 1..num_shots
                            shot_sentences: List[str] = []
                            for shot_spec in template_item["shots"][:num_shots]:
                                s_num = shot_spec["shot_number"]
                                slot_vals = build_slot_values(
                                    lang=lang,
                                    verb_entry=verb_entry,
                                    noun_set=noun_set,
                                    slot_mapping=shot_spec["slot_mapping"],
                                )
                                seed_key = (
                                    f"{lang}_{verb_class}_{template_id}_{verb_entry['id']}_"
                                    f"n{noun_idx + 1:02d}_shot{s_num}"
                                )
                                sent = render_shot_sentence(
                                    lang=lang,
                                    condition=condition,
                                    shot_spec=shot_spec,
                                    slot_values=slot_vals,
                                    rng_seed_str=seed_key,
                                )
                                shot_sentences.append(sent)

                            context_text = " ".join(shot_sentences)

                            # Format evaluation targets
                            verb_format_map = (
                                {
                                    "verb_lemma": verb_entry["lemma"],
                                    "verb_past": verb_entry["past"],
                                    "verb_pres_3sg": verb_entry["pres_3sg"],
                                    "verb_participle": verb_entry["participle"],
                                }
                                if lang == "en"
                                else {
                                    "verb_lemma": verb_entry["lemma"],
                                    "verb_past_part": verb_entry["past_part"],
                                    "verb_pres_3sg": verb_entry["pres_3sg"],
                                    "verb_subj_3sg": verb_entry["subj_3sg"],
                                    "verb_participle": verb_entry["participle"],
                                }
                            )
                            verb_surface = (
                                verb_entry["past"] if lang == "en" else verb_entry["past_part"]
                            )

                            for eval_type in EVAL_TYPES:
                                eval_short = (
                                    "semantic" if eval_type == "semantic_forced_choice" else "frame"
                                )
                                trial_id = (
                                    f"{lang}_{verb_class}_{item_short}_{condition}_"
                                    f"{verb_entry['id']}_n{noun_idx + 1:02d}_s{num_shots}_{eval_short}"
                                )
                                eval_spec = template_item["evaluations"][eval_type]
                                target_valid = eval_spec["target_valid"].format(**verb_format_map)
                                target_invalid = eval_spec["target_invalid"].format(**verb_format_map)

                                row = {
                                    "trial_id": trial_id,
                                    "language": lang,
                                    "verb_mode": verb_mode,
                                    "verb_class": verb_class,
                                    "template_id": template_id,
                                    "condition": condition,
                                    "num_shots": num_shots,
                                    "verb_lemma": verb_entry["lemma"],
                                    "verb_surface": verb_surface,
                                    "noun_set_id": noun_set_id,
                                    "eval_type": eval_type,
                                    "context_text": context_text,
                                    "target_valid": target_valid,
                                    "target_invalid": target_invalid,
                                }
                                rows.append(row)

    return rows


def compute_summary_stats(
    main_rows: List[Dict[str, Any]],
    sanity_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def summarize_split(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_lang = collections.Counter(r["language"] for r in rows)
        by_class = collections.Counter(r["verb_class"] for r in rows)
        by_cond = collections.Counter(r["condition"] for r in rows)
        by_eval = collections.Counter(r["eval_type"] for r in rows)
        by_shots = collections.Counter(str(r["num_shots"]) for r in rows)
        by_lang_shots = collections.Counter(f"{r['language']}_s{r['num_shots']}" for r in rows)
        return {
            "total_rows": len(rows),
            "by_language": dict(by_lang),
            "by_verb_class": dict(by_class),
            "by_condition": dict(by_cond),
            "by_eval_type": dict(by_eval),
            "by_num_shots": dict(by_shots),
            "by_language_and_shots": dict(by_lang_shots),
        }

    return {
        "benchmark_name": "Cross-Lingual Syntactic Bootstrapping Benchmark (EN vs. RO)",
        "random_seed": RANDOM_SEED,
        "benchmark_main": summarize_split(main_rows),
        "sanity_check": summarize_split(sanity_rows),
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically generate EN-RO syntactic bootstrapping benchmark datasets."
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=Path("configs/lexicon_en_ro.json"),
        help="Path to lexicon JSON config.",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("configs/templates_en_ro.json"),
        help="Path to templates JSON config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for generated JSONL and stats files.",
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    with open(args.lexicon, "r", encoding="utf-8") as f:
        lexicon = json.load(f)
    with open(args.templates, "r", encoding="utf-8") as f:
        templates = json.load(f)

    main_rows = generate_rows_for_mode(lexicon, templates, verb_mode="pseudoverb")
    sanity_rows = generate_rows_for_mode(lexicon, templates, verb_mode="real_control")
    summary_stats = compute_summary_stats(main_rows, sanity_rows)

    main_path = args.out_dir / "benchmark_main.jsonl"
    sanity_path = args.out_dir / "sanity_check.jsonl"
    stats_path = args.out_dir / "dataset_summary_stats.json"

    write_jsonl(main_path, main_rows)
    write_jsonl(sanity_path, sanity_rows)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("DATASET GENERATION COMPLETE (random.seed = 42)")
    print("=" * 80)
    print(f"1. Main Benchmark:  {main_path} ({len(main_rows):,} rows)")
    print(f"   - 4-shot rows:   {summary_stats['benchmark_main']['by_num_shots'].get('4', 0):,}")
    print(f"   - 1-shot rows:   {summary_stats['benchmark_main']['by_num_shots'].get('1', 0):,}")
    print(f"   - Per language:  EN={summary_stats['benchmark_main']['by_language']['en']:,}, RO={summary_stats['benchmark_main']['by_language']['ro']:,}")
    print(f"2. Sanity Check:    {sanity_path} ({len(sanity_rows):,} rows)")
    print(f"   - Per language:  EN={summary_stats['sanity_check']['by_language']['en']:,}, RO={summary_stats['sanity_check']['by_language']['ro']:,}")
    print(f"3. Summary Stats:   {stats_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
