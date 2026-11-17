#!/usr/bin/env python3
"""
Build `plugins/orthographic_romanian/orthographic_romanian.txt` for the Romanian
Wuggy orthographic language plugin from `wordfreq`.

Each line in the generated lexicon follows Wuggy's standard tab-separated format:
    <plain_word>\t<syllabified_word>\t<frequency_per_million>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from wordfreq import top_n_list, word_frequency

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLUGIN_DIR.parent.parent))

from plugins.orthographic_romanian.orthographic_romanian import OfficialLanguagePlugin

VOWEL_CHARS = set("aeiouăâî")
# Exclude non-native loan letters (k, q, w, y) and foreign consonant clusters/geminates
# so Wuggy BigramChain learns strictly authentic Romanian phonotactics.
VALID_RO_WORD_RE = re.compile(r"^[a-zăâîșț]+$")
FOREIGN_CHARS_RE = re.compile(r"[kqwy]|th|ph|sh|ck|tz|cz|sz|[cg]h(?![ei])")
FOREIGN_GEMINATE_RE = re.compile(r"([bcdfghjlmpqrstvwxz])\1")
OBSTRUENTS_LIQUID_FIRST = set("bcdfghptv")
LIQUIDS = set("lr")
SPECIAL_CCC_TWO_LEFT = {"lpt", "mpt", "nct", "ncț", "ncs", "ndv", "rct", "rtf", "stm"}

TRIPHTHONGS = set(OfficialLanguagePlugin.triple_letters)
DIPHTHONGS = set(OfficialLanguagePlugin.double_letters)

# Sentinels for velar digraphs ch/gh before e/i so they are never split across syllables
_CH_SENTINEL = "\u0001"
_GH_SENTINEL = "\u0002"
_SENTINEL_TO_DIGRAPH = {_CH_SENTINEL: "ch", _GH_SENTINEL: "gh"}


def normalize_ro_diacritics(word: str) -> str:
    return word.lower().strip().replace("ş", "ș").replace("ţ", "ț")


def _encode_digraphs(word: str) -> str:
    word = re.sub(r"ch(?=[ei])", _CH_SENTINEL, word)
    word = re.sub(r"gh(?=[ei])", _GH_SENTINEL, word)
    return word


def _decode_digraphs(text: str) -> str:
    for sentinel, digraph in _SENTINEL_TO_DIGRAPH.items():
        text = text.replace(sentinel, digraph)
    return text


def _segment_vowel_run(vrun: str) -> List[str]:
    """Split a contiguous run of vowels into Romanian vocalic nuclei (handling hiatus)."""
    nuclei: List[str] = []
    i = 0
    n = len(vrun)
    while i < n:
        if i + 3 <= n and vrun[i : i + 3] in TRIPHTHONGS:
            nuclei.append(vrun[i : i + 3])
            i += 3
        elif i + 2 <= n and vrun[i : i + 2] in DIPHTHONGS:
            nuclei.append(vrun[i : i + 2])
            i += 2
        else:
            nuclei.append(vrun[i])
            i += 1
    return nuclei


def syllabify_romanian(word: str) -> str:
    """
    Deterministically syllabify a lowercase Romanian word such that:
      - Standard Romanian onset/coda rules apply (V-CV, VC-CV, V-CCV for muta cum liquida,
        preserving ch/gh before e/i).
      - Vowel hiatus breaks cleanly so every syllable contains a nucleus matched by
        OfficialLanguagePlugin.oncpattern.
    """
    enc = _encode_digraphs(word)
    # Tokenize into alternating consonant runs and vowel runs
    tokens: List[Tuple[str, str]] = []
    i = 0
    while i < len(enc):
        if enc[i] in VOWEL_CHARS:
            j = i
            while j < len(enc) and enc[j] in VOWEL_CHARS:
                j += 1
            for nuc in _segment_vowel_run(enc[i:j]):
                tokens.append(("V", nuc))
            i = j
        else:
            j = i
            while j < len(enc) and enc[j] not in VOWEL_CHARS:
                j += 1
            tokens.append(("C", enc[i:j]))
            i = j

    # Locate indices of vocalic nuclei
    v_indices = [idx for idx, (kind, _) in enumerate(tokens) if kind == "V"]
    if len(v_indices) <= 1:
        return word

    syllables: List[str] = []
    cur = ""
    # Leading consonant onset before first vowel
    if v_indices[0] > 0:
        cur += tokens[0][1]

    for k, v_idx in enumerate(v_indices):
        cur += tokens[v_idx][1]
        if k == len(v_indices) - 1:
            # Trailing coda after last vowel
            if v_idx + 1 < len(tokens):
                cur += tokens[v_idx + 1][1]
            syllables.append(cur)
            break

        next_v_idx = v_indices[k + 1]
        if next_v_idx == v_idx + 1:
            # Adjacent nuclei (hiatus) -> split directly between nuclei
            syllables.append(cur)
            cur = ""
            continue

        # Consonant cluster between tokens[v_idx] and tokens[next_v_idx]
        cluster = tokens[v_idx + 1][1]
        decoded_cluster = _decode_digraphs(cluster)
        if len(cluster) == 1:
            split_at = 0
        elif len(cluster) == 2:
            if cluster[0] in OBSTRUENTS_LIQUID_FIRST and cluster[1] in LIQUIDS:
                split_at = 0
            else:
                split_at = 1
        elif len(cluster) >= 3:
            if decoded_cluster[:3] in SPECIAL_CCC_TWO_LEFT and len(cluster) >= 3:
                split_at = 2
            else:
                split_at = 1

        left_c = cluster[:split_at]
        right_c = cluster[split_at:]
        syllables.append(cur + left_c)
        cur = right_c

    return _decode_digraphs("-".join(syllables))


def build_lexicon(output_path: Path, max_words: int = 25000) -> int:
    plugin = OfficialLanguagePlugin()
    freq_by_word: Dict[str, float] = {}

    for raw in top_n_list("ro", 60000):
        w = normalize_ro_diacritics(raw)
        if len(w) < 2 or not VALID_RO_WORD_RE.match(w):
            continue
        if FOREIGN_CHARS_RE.search(w) or FOREIGN_GEMINATE_RE.search(w):
            continue
        if not any(ch in VOWEL_CHARS for ch in w):
            continue
        freq_pm = word_frequency(raw, "ro") * 1_000_000.0
        freq_by_word[w] = freq_by_word.get(w, 0.0) + freq_pm

    sorted_words = sorted(freq_by_word.items(), key=lambda x: (-x[1], x[0]))[:max_words]

    written = 0
    skipped = 0
    lines: List[str] = []
    for word, freq_pm in sorted_words:
        syll = syllabify_romanian(word)
        try:
            # Verify compatibility with Wuggy BaseLanguagePlugin segmentation
            plugin.transform(syll, frequency=freq_pm)
        except Exception:
            skipped += 1
            continue
        lines.append(f"{word}\t{syll}\t{freq_pm:.4f}\n")
        written += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(
        f"Built Romanian Wuggy lexicon at {output_path}: "
        f"{written:,} entries written ({skipped} skipped)."
    )
    return written


if __name__ == "__main__":
    build_lexicon(PLUGIN_DIR / "orthographic_romanian.txt")
