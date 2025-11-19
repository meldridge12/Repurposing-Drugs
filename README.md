# Repurposing-Drugs

CS484 capstone exploring whether machine learning can recommend new disease indications for existing drugs using publicly available chemistry and literature metadata. The repository now hosts three research phases that build from quick baselines to an optimized deep-learning approach.

---

## Dataset Snapshot

- **Rows:** 73,807 drug–disease association records after cleaning
- **Classes:** 245 unique diseases (multi-class classification)
- **Drugs:** 8 unique chemical entities represented by names, CAS numbers, SMILES strings, and molecular descriptors
- **Excluded leakage:** `inferencescore`, `directevidence`, and other curated signals are removed in every “honest” experiment

Key feature families we engineer across notebooks:

1. **Text (TF–IDF):** 200 uni/bi-gram features extracted from curated drug descriptions
2. **Structure (SMILES):** 8 rule-based counts for length, aromaticity, rings, double/triple bonds, etc.
3. **Molecular descriptors:** 5 physico-chemical properties (logP variants, solubility, RO5, etc.)
4. **Categorical embeddings (Phase 3):** Learned 32-d vectors for drug identities instead of hash-encoding

---

## Repository Layout

- `finaldataset.csv` — consolidated training table consumed by all notebooks.
- `CTD_chemicals_diseases.csv`, `drugbank_clean.csv`, `drug_disease_merged.csv` — raw public sources that feed the cleaning/merge steps documented in the notebooks.
- `disease_prediction_lightgbm.ipynb` — Phase 1 exploratory LightGBM with known leakage for ceiling estimates.
- `honest_drug_disease_prediction.ipynb` — Phase 2 “honest” LightGBM pipeline with engineered features only.
- `phase3_deep_learning_optimized.ipynb` — Phase 3 PyTorch workflow that combines embeddings and dense features.
- `best_model.pth` — automatically saved checkpoint from Phase 3 whenever validation Top‑20 improves.
- `drug_repurposing_project.md` — long-form project notes and presentation scaffolding.
- `README.md` (this file) — high-level overview plus talking points for reviewers.

Feel free to add new datasets or notebooks, but keep the naming aligned with their role (phase number, purpose, or data stage) so reviewers can retrace the chronology quickly.

---

## Phase Overview

| Phase | Notebook | Core Idea | Top‑1 Acc. | Top‑20 Acc. |
|-------|----------|-----------|------------|-------------|
| 1 | `disease_prediction_lightgbm.ipynb` | Straight LightGBM with leakage present (“inferencescore”) to understand upper bound | 73.7 %† | 99 %† |
| 2 | `honest_drug_disease_prediction.ipynb` | Honest LightGBM using only engineered text/structure/molecular features, no categorical memorization | **0.66 %** | **14.55 %** |
| 3 | `phase3_deep_learning_optimized.ipynb` | Deep neural network with embeddings, label smoothing, LR scheduling, and balanced sampling | **4.28 %** | **62.29 %** |

> †Phase 1 deliberately shows the “cheating” ceiling when curated inference scores are allowed; these numbers are not used for conclusions.

---

## End-to-End Workflow

1. **Ingest & Inspect Data:** Start from the raw CTD/DrugBank CSVs to confirm schema, drop duplicates, and align disease codes.
2. **Curate Honest Dataset:** Build `finaldataset.csv` by removing leakage columns, mapping categorical IDs, and engineering TF–IDF, SMILES counts, and simple molecular descriptors.
3. **Establish Ceiling (Phase 1):** Run `disease_prediction_lightgbm.ipynb` with the leaky `inferencescore` to understand the unrealistic upper bound every future phase should be compared against.
4. **Baseline Honest Model (Phase 2):** Execute `honest_drug_disease_prediction.ipynb` to train vanilla LightGBM purely on engineered continuous features, logging stratified metrics for every drug.
5. **Deep Learning Upgrade (Phase 3):** Move to `phase3_deep_learning_optimized.ipynb`, which introduces embeddings, label smoothing, cosine LR scheduling, and a class-balanced sampler to tackle the multi-class imbalance head-on.
6. **Evaluate & Report:** Compare Top‑K metrics across phases, capture plots/checkpoints, and summarize learnings in `README.md` and `drug_repurposing_project.md` for stakeholders.

This ordered workflow keeps the historical context intact: each notebook builds on artifacts produced earlier, so anyone can replay the progression without guessing which file to open first.

---

## Phase 3 Highlights (Deep Learning)

Phase 3 delivers the largest jump in practical usefulness by turning the problem into “rank the most likely diseases for each drug.”

- **Architecture:** 32-d drug embeddings concatenated with 213 continuous features → 4 hidden blocks (512→64 neurons) with BatchNorm + 40 % dropout → 245-class softmax head.
- **Training tricks:** label smoothing (0.1), AdamW optimizer, cosine annealing warm restarts, gradient clipping, and a WeightedRandomSampler so rare diseases appear as frequently as common ones.
- **Outcome:** Test Top‑1 jumps from 0.66 % → 4.28 % and Top‑20 from 14.55 % → 62.29 %. The model now places the true disease in a list of 20 suggestions ~2 out of 3 times, a 4.3× improvement over Phase 2.
- **Artifacts:** Best checkpoint saved to `best_model.pth` whenever validation Top‑20 improves; notebook also plots learning curves, confidence histograms, and per-class F1 statistics for reporting.

Use this section directly in presentations/reporting to describe how “honest” neural networks can beat traditional tabular models once embeddings, regularization, and balanced sampling are introduced.

---

## Accuracy Trajectory & Interpretation

- **Phase 1 (Leaky ceiling):** 73.7 % / 99 % Top‑1/Top‑20 while memorizing curation hints; serves only as a sanity check that the dataset can encode useful signals.
- **Phase 2 (Honest baseline):** Accuracy collapses to 0.66 % / 14.55 % once leakage is removed, revealing the true difficulty when relying solely on simple engineered descriptors.
- **Phase 3 (Optimized NN):** Embeddings + dense features + balanced training push scores to 4.28 % / 62.29 %, proving that representation learning and class weighting reclaim much of the lost ranking power.
- **Key takeaways:** (a) categorical embeddings let the network learn drug similarity directly, (b) label smoothing and dropout limit overfitting despite tiny class support, and (c) Top‑K metrics should guide decisions because repurposing is a ranking workflow.

When presenting results, emphasize the relative lift (6.5× Top‑1, 4.3× Top‑20 over Phase 2) and highlight that every improvement stemmed from a documented architectural or training change, making the roadmap reproducible.

---

## Running the Notebooks

1. Create/activate a Python 3.12 environment with the packages listed in the VS Code kernel (PyTorch 2.9+, scikit-learn 1.5+, LightGBM 4.5+, seaborn/matplotlib, etc.).
2. Open each notebook in VS Code or Jupyter Lab in the order shown in the table above.
3. Ensure `finaldataset.csv` is in the repository root (already tracked). The notebooks expect the file path relative to their location.
4. Run all cells sequentially. Each notebook prints intermediate status (dataset sizes, train/val/test splits, hyperparameters, validation curves). Phase 3 will save `best_model.pth` automatically.

Optional CLI launch commands:

```bash
cd Repurposing-Drugs
code phase3_deep_learning_optimized.ipynb
```

---

## Suggested Talking Points

- The project demonstrates the gap between “leaky” and “honest” experiments and documents every fix (removing inference scores, dropping categorical hashes, balancing classes).
- Reporting should emphasize Top‑K metrics (Top‑5, Top‑10, Top‑20) because drug repurposing is a ranking/filtering workflow rather than single-label prediction.
- Future work: increase embedding dimensions, add attention over text descriptors, or ensemble Phase 2 (trees) + Phase 3 (NN) for even better coverage.

---

Feel free to adapt this README as you iterate on new notebooks or datasets; keep the phase table and metric comparisons updated so reviewers immediately see the progression.
