# Syntactic vs. Morphosyntactic Bootstrapping of Novel Verb Meanings in Pretrained Language Models (English vs. Romanian)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark Validation](https://img.shields.io/badge/validation-100%25%20PASS-brightgreen.svg)](scripts/validate_dataset.py)

This repository contains the official dataset generation pipeline, lexical and psycholinguistic resources, validation suites, and generated benchmark stimuli for evaluating **cross-lingual syntactic bootstrapping** in pretrained multilingual language models (English vs. Romanian).

---

## 1. Theoretical Background & Motivation

* **Theoretical Foundation:** Built upon Lila Gleitman's (1990) **Syntactic Bootstrapping Hypothesis**, which posits that structural frames constrain and guide fast-mapping of novel verb meanings during language acquisition.
* **Base Paper:** Zhu, McCoy & Frank (2026), *"The Structural Sources of Verb Meanings Revisited"* (*Open Mind*), which established that language models exploit sentence structure to infer verb meanings even when lexical cues are ambiguous.
* **The Cross-Lingual Dimension (English vs. Romanian):**
  * **English (Analytic, strict SVO word order, sparse inflectional morphology):** Argument structure relies heavily on linear order. Scrambling word order (`constituent_scrambled` or `word_scrambled`) breaks linear phrase structure and inverts or obscures semantic roles (e.g., reversing agent and patient in OVS).
  * **Romanian (Fusional/Synthetic, flexible word order, rich morphosyntax):** Thematic roles are redundantly encoded via morphological case inflections (Dative `-ei` / `-ului`), Differential Object Marking (`pe` + Accusative), clitic doubling (`l-a`, `i-a`), and complementizers/mood particles (`că` + indicative).
  * **Central Hypothesis:** In English, word-order scrambling severely degrades novel verb fast-mapping. In Romanian, rich morphosyntactic markers preserve thematic role assignment under constituent and word scrambling, and even partially under inflected bag-of-words (`bow_inflected`), degrading completely only when both syntax and inflectional suffixes are stripped (`bow_lemmatized`).

---

## 2. Repository Structure

```text
├── README.md                        # Project documentation and reproduction guide
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Standard ignore rules
├── configs/
│   ├── lexicon_en_ro.json           # Morphological paradigms, pseudoverbs, controls, and inflected noun sets
│   └── templates_en_ro.json         # 48 parallel base items (24 EN + 24 RO), 4 shots/item, 2 eval pairs/item
├── scripts/
│   ├── verify_tokenization.py       # Subword tokenization inspector (live HF AutoTokenizer + offline fallback)
│   ├── generate_dataset.py          # Deterministic (seed=42) benchmark generator
│   └── validate_dataset.py          # Automated QA and factorial balance verification suite
└── data/
    ├── benchmark_main.jsonl         # Main novel-verb benchmark (10,080 rows: 6,720 4-shot + 3,360 1-shot)
    ├── sanity_check.jsonl           # Real control verb ceiling dataset (1,344 rows)
    ├── dataset_summary_stats.json   # Factorial summary statistics and cell counts
    └── tokenization_report.json     # Subword tokenization report across target multilingual models
```

---

## 3. Experimental Design

### 3.1 Verb Classes

The benchmark evaluates fast-mapping across **3 distinct verb classes** (each with 10 phonotactically valid pseudoverbs and 2 real control verbs per language):

1. **`mental_epistemic` (Belief / Attitude):** Sentential complement frame.
   - *EN:* `NP1 [V_past] that NP2 [VP_past]` (*"The detective daxed that the witness lied."*)
   - *RO:* `NP1_nom a [V_part] că NP2_nom [VP_past]` (*"Detectivul a daxat că martorul a mințit."*)
2. **`physical_ditransitive` (Physical Transfer of Possession):** Double-object / Dative frame.
   - *EN:* `NP1 [V_past] NP2_recipient NP3_theme` (*"The doctor daxed the assistant the folder."*)
   - *RO:* `NP1_nom i-a [V_part] NP2_dat NP3_acc` (*"Doctorul i-a daxat asistentei dosarul."*)
3. **`physical_transitive_causative` (Physical Force / Impact on Animate Patient):** Transitive frame.
   - *EN:* `NP1 [V_past] NP2_patient` (*"The chef daxed the waiter."*)
   - *RO:* `NP1_nom l-a [V_part] pe NP2_acc` (*"Bucătarul l-a daxat pe ospătar."*)

---

### 3.2 The 7 Experimental Conditions

Each template item provides exposure contexts constructed under 7 systematic structural manipulations:

| # | Condition | Linguistic Manipulation | English Example (`en`) | Romanian Example (`ro`) |
| :- | :--- | :--- | :--- | :--- |
| **1** | `syntax_rich_informative` | Intact canonical syntax + domain-specific nouns | *The doctor daxed the assistant the folder.* | *Doctorul i-a daxat asistentei dosarul.* |
| **2** | `syntax_rich_ambiguous` | Intact canonical syntax + domain-general neutral nouns | *The person daxed the neighbor the object.* | *Persoana i-a daxat vecinului obiectul.* |
| **3** | `cue_conflict` | Intact syntax of Class A + domain nouns from contrasting Class B | *The detective daxed the witness the alibi.* | *Detectivul i-a daxat martorului alibiul.* |
| **4** | `constituent_scrambled` | Major constituents permuted (OVS / VOS / OSV) | *The assistant the folder daxed the doctor.* | *Asistentei dosarul i-a daxat doctorul.* |
| **5** | `word_scrambled` | Words randomly shuffled within each sentence | *the daxed doctor folder the the assistant.* | *doctorul asistentei dosarul i-a daxat.* |
| **6** | `bow_inflected` | Bag-of-words without function words; **inflections preserved** | *assistant doctor folder daxed.* | *asistentei doctorul dosarul daxat.* |
| **7** | `bow_lemmatized` | Bag-of-words with all words converted to **dictionary lemmas** | *assistant doctor folder dax.* | *asistentă doctor dosar daxa.* |

> **Note on BoW Permutation Control:** In `scripts/generate_dataset.py`, `bow_inflected` and `bow_lemmatized` use the exact same deterministic index permutation for each shot, ensuring that any performance differential is driven strictly by morphological suffixes rather than word-order noise.

---

### 3.3 Evaluation Tasks

Each trial presents the language model with an in-context exposure text (`context_text`) followed by a binary forced-choice evaluation between a valid continuation (`target_valid`) and a foil (`target_invalid`):

1. **`semantic_forced_choice` (Semantic Paraphrase):** Evaluates verb meaning comprehension.
   - *Valid:* *"In other words, someone physically transferred a concrete object to a recipient."*
   - *Foil:* *"In other words, someone silently held a mental belief about a proposition."*
2. **`held_out_frame` (Syntactic Generalization):** Evaluates structural generalization to an unseen grammatical alternation for that class.
   - *Valid (Prepositional Dative):* *"Later, someone daxed a package to the assistant."*
   - *Foil (Epistemic Complement):* *"Later, someone daxed that the situation was complicated."*

---

## 4. Dataset Schema & Statistics

### 4.1 Schema Specification (14 Fields)

Every line in `data/benchmark_main.jsonl` and `data/sanity_check.jsonl` is a JSON object with the following fields:

```json
{
  "trial_id": "ro_physical_ditransitive_item01_word_scrambled_v01_n01_s4_semantic",
  "language": "ro",
  "verb_mode": "pseudoverb",
  "verb_class": "physical_ditransitive",
  "template_id": "item_01",
  "condition": "word_scrambled",
  "num_shots": 4,
  "verb_lemma": "daxa",
  "verb_surface": "daxat",
  "noun_set_id": "info_01",
  "eval_type": "semantic_forced_choice",
  "context_text": "doctorul asistentei dosarul i-a daxat. i-a chirurgul daxat infirmierei fișa...",
  "target_valid": "Cu alte cuvinte, cineva a transferat fizic un obiect concret către un destinatar.",
  "target_invalid": "Cu alte cuvinte, cineva a păstrat în tăcere o convingere mintală despre o idee."
}
```

### 4.2 Benchmark Splits

* **`data/benchmark_main.jsonl` (10,080 rows total):**
  * `language`: 5,040 `en` rows, 5,040 `ro` rows (perfectly balanced).
  * `verb_class`: 3,360 rows per verb class.
  * `condition`: 1,440 rows per experimental condition.
  * `eval_type`: 5,040 `semantic_forced_choice` rows, 5,040 `held_out_frame` rows.
  * `num_shots`: 6,720 rows in `4-shot` (10 instantiations), 3,360 rows in `1-shot` (5 instantiations).
  * Factorial coverage: 84 orthogonal cells (`2 languages × 3 classes × 7 conditions × 2 eval types`), each containing exactly 120 trials.
* **`data/sanity_check.jsonl` (1,344 rows total):**
  * Uses 2 real control verbs per class (e.g., `think`/`crede`, `give`/`oferi`, `push`/`împinge`) in 4-shot contexts to verify ceiling baseline performance (~85–100%) before evaluating pseudoverbs.

---

## 5. Quickstart & Reproduction Pipeline

### 5.1 Environment Setup

```bash
git clone https://github.com/<your-username>/Syntactic-Bootstrapping_en_ro.git
cd Syntactic-Bootstrapping_en_ro
pip install -r requirements.txt
```

### 5.2 Reproducing the Benchmark Datasets

```bash
# 1. Deterministically generate data/benchmark_main.jsonl and data/sanity_check.jsonl
python scripts/generate_dataset.py

# 2. Inspect pseudoverb subword tokenization across target LLM tokenizers
python scripts/verify_tokenization.py

# 3. Execute the automated QA verification suite (verifies 100% test passage)
python scripts/validate_dataset.py
```

---

## 6. Evaluation Protocol (For Downstream Modeling)

To score an autoregressive language model on the benchmark:

1. For each trial, construct the candidate prompt sequences:
   $$\text{seq}_{\text{valid}} = \text{context\_text} + \text{" "} + \text{target\_valid}$$
   $$\text{seq}_{\text{invalid}} = \text{context\_text} + \text{" "} + \text{target\_invalid}$$
2. Compute the conditional log-probability of target tokens given the context:
   $$\log P(\text{target} \mid \text{context}) = \sum_{t=1}^{T} \log P(y_t \mid \text{context}, y_{<t})$$
3. Apply length normalization to mitigate token-length variance:
   $$\text{score}_{\text{norm}} = \frac{1}{T} \sum_{t=1}^{T} \log P(y_t \mid \text{context}, y_{<t})$$
4. Accuracy is scored as binary correct when $\text{score}(\text{target\_valid}) > \text{score}(\text{target\_invalid})$.

---

## 7. License & Acknowledgments

This benchmark is released under the [MIT License](LICENSE).  
Theoretical framework inspired by Gleitman (1990) and Zhu, McCoy & Frank (2026).
