# microGEM User Guide

**Version 1.0.0** ｜ Microbial-enzyme scoring from protein sequence ｜ English

> 中文版: [USER-GUIDE.md](USER-GUIDE.md)

---

## 1. What this is

microGEM answers one question: **given a protein sequence nobody has characterised,
is it one of these 17 enzyme chemistries?**

It is not a retrieval tool. UniProt, BRENDA, MetaCyc and KEGG tell you what a
*characterised* enzyme is. microGEM tells you what an *uncharacterised* protein
might be, and whether it is worth an experiment. 60–80% of predicted ORFs in a
metagenome have no reliable annotation, and those databases are silent on all of
them — that gap is what microGEM is built for.

**The model is deliberately simple**: frozen ESM-2 650M protein language model →
1280-d vector → standardise → logistic regression → sigmoid. Seventeen panels
share one embedding, each with its own linear probe. All weights together are
244 KB, and inference needs only NumPy.

**The evaluation is deliberately strict**: every number comes from *cluster-aware*
cross-validation — positives and negatives clustered together at bitscore 200,
whole clusters assigned to one fold, so the same protein family never appears on
both sides of a split. That is stricter than most published enzyme-mining work,
and it is why some numbers look lower than you might expect.

---

## 2. Installation

### 2.1 The short path

```bash
git clone <repo> && cd microgem-skill
./install.sh
```

The installer does two independent things:

1. **the command** — installs `microgem` into a Python environment (`pip install --user` by default)
2. **the skill** — copies `skill/` into the agent's skill directory so the
   assistant knows when and how to call it

### 2.2 Options

| Option | Effect |
|---|---|
| `--cli-only` | command only, no skill |
| `--skill-only` | skill only, no command |
| `--venv PATH` | install into a dedicated virtualenv (recommended) |
| `--agent NAME` | skill root: `dsh` \| `claude` \| `agents` \| any path |
| `--download-model` | also fetch the 2.6 GB ESM-2 weights |
| `--with-torch` | also install PyTorch + fair-esm (5–15× faster on GPU) |
| `--dry-run` | print what would happen, write nothing |

### 2.3 Manual installation

```bash
pip install .                       # command only; ~30 s on a good connection

# or into a dedicated environment
python3 -m venv ~/.venvs/microgem
~/.venvs/microgem/bin/pip install .
```

The skill is just a directory copy:

```bash
# DSH
cp -R skill ~/.dsh/skills/microgem
# Claude Code
cp -R skill ~/.claude/skills/microgem
# generic agents root
cp -R skill ~/.agents/skills/microgem
```

### 2.4 Model weights (required, one-off)

The model is 2.6 GB and **not bundled**. Three options:

```bash
# 1. download (needs torch + fair-esm first)
microgem download

# 2. you already have the .pt file
export MICROGEM_ESM2=/path/to/esm2_t33_650M_UR50D.pt

# 3. torch-free path: convert once
microgem download --convert     # torch is not needed for scoring afterwards
```

**No GPU is required.** There are two inference paths:

| Path | Depends on | Speed | Notes |
|---|---|---|---|
| `torch` | torch + fair-esm + the `.pt` | 0.2 s/seq on GPU, ~1.5 s on CPU | fast, recommended |
| `numpy` | NumPy only | ~3 s/seq on CPU | slow but light; runs on a laptop |

Both paths produce **the same vector** (measured max deviation 9.5e-7). After
`microgem convert`, PyTorch can be uninstalled.

---

## 3. Five-minute start

### 3.1 Confirm the install

```bash
microgem info
```

```
microGEM 1.0.0
  embedding backend : torch
  model cache       : /home/<user>/.cache/microgem
  ESM-2 checkpoint  : /home/<user>/models/esm2/esm2_t33_650M_UR50D.pt
  NumPy weights     : .../esm2_t33_650M_UR50D_numpy (not converted)
  panels            : 17
```

`backend: none` means the model is missing — see §2.4.

### 3.2 Self-test (~15 s on GPU, ~2.5 min on CPU)

```bash
microgem selftest
```

```
worst deviation: 4.95e-07 (AATE-lyase|Q13YI7/AATE-lyase)   tolerance 1e-03
cross-panel false positives: 27/752 (3.6%)
PASS — this installation reproduces the exported model
```

This verifies that **the scores this install produces match the ones the export
host produced**. 47 reference proteins cover all 17 panels, each as either a
per-panel positive control or an un-saturated drift probe. A changed embedding,
a wrong weight file, a truncated download — all fail here.

### 3.3 Score

```bash
microgem score proteins.faa
```

```
microGEM 1.0.0 | 17 sequences | 17 panels | per-panel calibrated cut-offs

> FATE-L-modifying|Q96GR2  (724 aa)
    CALL FATE-L-modifying  1.000 [####################] reliable AUPRC 0.894
    CALL BA                0.959 [###################.] reliable AUPRC 0.959
    CALL FATE-S            0.702 [##############......] reliable AUPRC 0.891
         ARO               0.063 [#...................] reliable AUPRC 0.890
         ... 13 further panels, all <= 0.015 (--verbose for all)
    -> assigned: FATE-L-modifying  (Fatty Acyl Transformation Enzyme — Long chain)
    -> also above cut-off: FATE-S, BA
```

Each row reads: **called? ｜ panel ｜ probability ｜ bar ｜ reliability tier ｜ that panel's AUPRC**.

### 3.4 Three output formats

```bash
microgem score proteins.faa                    # human-readable
microgem score proteins.faa --format tsv       # table, readable by Excel/R
microgem score proteins.faa --format json      # for downstream code (best for agents)
```

The JSON record per sequence carries `id`, `length`, `sha1`, `called` (panels above
their cut-off), `best`, `best_probability`, `margin`, and `scores` (all 17).

### 3.5 Python API

```python
from microgem import score, PANELS

result = score("proteins.faa")                    # all 17 panels
result = score(["MKVL...", "MSQE..."])            # raw sequences
result = score({"my_protein": "MKVL..."})         # named

for item in result:
    best = item.best
    print(f"{item.identifier}: {best.panel} {best.probability:.3f} "
          f"(tier {best.tier}, margin {item.margin:.3f})")
    print("  all calls:", item.called)

result.to_json()      # full report
```

---

## 4. The 17 panels

### 4.1 Overview

| Panel | Family | What it decides | AUPRC | Tier |
|---|---|---|---|---|
| `BA` | bile acid | a bile acid is the substrate | 0.959 | reliable |
| `AATE-ligase` | amino acid | ATP-dependent bond formation (aminoacyl-tRNA synthetase…) | 0.925 | reliable |
| `PLY` | polyamine | putrescine/spermidine/spermine metabolism | 0.919 | reliable |
| `RDX` | oxidative stress | glutathione / thioredoxin / peroxiredoxin / catalase | 0.909 | reliable |
| `FATE-L-releasing` | long-chain fatty acid | lipid-ester hydrolysis releasing a long-chain fatty acid | 0.899 | reliable |
| `FATE-L-modifying` | long-chain fatty acid | free long-chain fatty acid or its CoA is modified | 0.894 | reliable |
| `FATE-S` | short-chain fatty acid | C2–C6 fatty acid / acyl-CoA | 0.891 | reliable |
| `ARO` | aromatic | aromatic ring cleavage and modification | 0.890 | reliable |
| `AATE-hydrolysis` | amino acid | hydrolytic (deaminase, amidase) | 0.871 | reliable |
| `GLY` | carbohydrate | glycoside hydrolysis (CAZy class GH) | 0.863 | reliable |
| `SATE` | SAM | SAM transfers a methyl (produces SAH) | 0.861 | reliable |
| `VIT` | vitamin | de-novo vitamin/cofactor biosynthesis | 0.819 | moderate |
| `TRP` | tryptophan | tryptophan → indole / AhR-ligand branch | 0.783 | moderate |
| `AATE-redox` | amino acid | redox (dehydrogenase, oxidase) | 0.751 | moderate |
| `AATE-lyase` | amino acid | lyase/isomerase (decarboxylase, aminotransferase) | 0.715 | moderate |
| `SALE` | SAM | SAM consumed without methyl transfer (radical SAM) | 0.628 | **weak** |
| `AATE-CoA` | amino acid | acyl-CoA-dependent N-acylation | 0.503 | **weak** |

Tier cut-offs: AUPRC ≥ 0.85 reliable / 0.65–0.85 moderate / < 0.65 weak.

### 4.2 How panels were defined

Each panel is defined by **what the enzyme acts on**, not by EC number. Two
consequences: enzymes under one EC can land in different panels (different
chemistry), and one panel can span several EC classes.

Three decisions worth knowing:

- **FATE-L split in two**: is the substrate a free fatty acid (modifying) or a
  lipid ester whose hydrolysis releases a fatty acid (releasing)? Opposite
  chemistry. AUPRC 0.607 before the split, 0.894 / 0.899 after.
- **SALE split out of SATE**: enzymes that bind SAM but do not transfer a methyl
  (radical SAM) are a different structural family. Merged, 40.8% of positives
  scored below 0.5.
- **Aldehyde dehydrogenases removed from FATE-S**: acetaldehyde dehydrogenase
  starts from an **aldehyde**, not a fatty acid. A regex match on `acetaldehyde`
  made the panel call all 111 wrongly; AUPRC 0.752 → 0.901.
- **AATE split into five by reaction chemistry**: 131 of 348 reactions carry no EC
  number, so an EC-based split would drop 40%.

Full definitions and rationale: `skill/references/panels.md`.

### 4.3 Claim priority

When a reaction matches several panels, the earlier one wins and the reaction is
removed from the others:

```
BA > GLY > SALE > SATE > PLY > TRP > ARO > VIT > RDX
   > FATE-L-modifying > FATE-L-releasing > FATE-S
   > AATE-CoA > AATE-ligase > AATE-lyase > AATE-redox > AATE-hydrolysis
```

45 pairwise cross-checks, zero reaction overlap. That is what makes multi-label
scoring coherent.

---

## 5. Reading the scores (the important chapter)

### 5.1 The probability is not a posterior probability

The panels were fitted with `class_weight="balanced"` on ~10% positives, so the
output behaves like a likelihood ratio. The practical consequence: **the same 0.5
means very different precision on different data.**

| Panel | cut-off | CV precision | precision at 0.1% prevalence |
|---|---|---|---|
| SATE | 0.30 | 0.900 | 0.075 |
| AATE-ligase | 0.60 | 0.882 | 0.064 |
| TRP | 0.40 | 0.723 | 0.020 |
| GLY | 0.20 | 0.833 | 0.045 |

The third column is what you see when you throw a whole metagenome at it. The
second is not a lie — it describes the cross-validation set (~10% positives).

To recompute for your data:

```bash
microgem score proteins.faa --prevalence 0.001 --show-operating-points
```

### 5.2 Panels are not comparable to each other

A 0.85 on `SALE` is **weaker** evidence than a 0.65 on `GLY`. The two probes were
trained against different negatives with different class balances; their outputs
are not on one ruler. What *is* comparable is the reliability tier, and the
ranking within one sequence.

Rank by: **① tier > ② probability > ③ margin**.

### 5.3 A margin below 0.1 is not a call

margin = best − runner-up. Below 0.1, two probes are firing together and the model
cannot separate them. Three recurring cases:

- `FATE-L-modifying` + `FATE-L-releasing` both high: the protein sits on the
  ester/free-acid boundary; a promiscuous esterase looks like both
- two neighbouring `AATE-*` panels: the reaction is at the chemistry-split boundary
- `SATE` + `SALE`: it binds SAM, but the product chemistry is ambiguous in the representation

**Report both.** Do not pick one silently.

### 5.4 State the weak panels up front

- **`AATE-CoA`**: 57.6% of positives come from one reaction (EC 2.3.1.1); the
  succinyl-CoA subclass is called wrong in every fold; 30-seed AUPRC 0.62 ± 0.068.
  **Not usable for screening.**
- **`SALE`**: chemically coherent, but the positive set spans several unrelated
  folds. A high score means "SAM-consuming, not transmethylating" and nothing more.

If the user asks about these directions specifically, say the panel is weak
*before* quoting a number.

### 5.5 RDX needs taxonomy first

`RDX` (glutathione, thioredoxin, peroxiredoxin, catalase) is heavily homologous to
host enzymes. A human peroxiredoxin scores like a microbial one. For gut samples,
confirm the candidate is actually bacterial before interpreting an RDX score.

### 5.6 BA has a small sample

Only 54 positives (277 clusters), so the AUPRC confidence interval is wide, and
the positives mix eukaryotic with bacterial enzymes. **For bile acids, prefer
BEAUT** (Ding et al., *Cell* 2025) — far more data and experimental validation.

### 5.7 Always quote the AUPRC with the probability

> ✗ "This protein is a glycoside hydrolase with 0.91 confidence."

> ✓ "This protein scores 0.91 on `GLY` (AUPRC 0.863, reliable tier), with no
> competing panel above its cut-off."

A probability without its AUPRC is a number with no scale.

---

## 6. Command reference

```bash
# one direction
microgem score proteins.faa --panel GLY

# a family (expands to 2 or 5 sub-panels)
microgem score proteins.faa --panel FATE-L      # FATE-L-modifying + FATE-L-releasing
microgem score proteins.faa --panel AATE        # all 5 AATE sub-panels
microgem score proteins.faa --panel SAM         # SATE + SALE

# several panels (comma-separated, case-insensitive)
microgem score proteins.faa --panel GLY,SATE,PLY

# your own cut-off (takes the panels off their calibration)
microgem score proteins.faa --threshold 0.5

# score only the first 100 (to estimate cost before committing)
microgem score big.faa --limit 100

# explain sequences with no call
microgem score proteins.faa --verbose

# panel details
microgem panels
microgem panels --format json

# install state
microgem info --format json
```

### 6.1 Flags

| Flag | Meaning |
|---|---|
| `--panel` | panel name / family / `all` (default) |
| `--threshold` | force one cut-off; default is per-panel calibrated |
| `--format` | `text` \| `tsv` \| `json` |
| `--prevalence` | what fraction of *your* input is this enzyme class |
| `--show-operating-points` | print precision/recall at the applied cut-offs |
| `--limit N` | score only the first N records |
| `--model` | explicit `.pt` path |
| `--device` | `cuda` \| `cpu` \| `mps` |
| `--backend` | `auto` \| `torch` \| `numpy` |
| `--batch-size` | batch size (default 16; lower it for small GPUs) |
| `--verbose` | all 17 rows + explanations for sequences with no call |
| `--quiet` | suppress progress and caveats |

---

## 7. Using it from an AI assistant

### 7.1 Invoke it with `/microgem` (recommended)

Type the slash command in the composer; the skill body is injected into that step:

```
/microgem
```

With no task attached, the assistant runs the two cheap checks (`info` and
`selftest`) and then asks one question:

> microGEM is ready: 17 panels, backend torch, selftest PASS.
> What would you like scored? Point me at a FASTA file or paste sequences, and say
> whether you want all 17 panels or a direction (e.g. GLY, FATE-L, AATE).

Attach the task directly when you have one:

```
/microgem which proteins in candidates.faa are glycoside hydrolases
/microgem anything here worth testing as a tryptophan-metabolising enzyme
/microgem microGEM called this SALE — should I believe it
```

The gesture also works mid-sentence ("run `/microgem` on this file"), but a second
slash breaks it: `/microgem/refs` is a path, not a command, and will not trigger.

### 7.2 Plain language works too

Without the slash command, the skill still loads when the request matches its
description (enzyme-function prediction, metagenome screening, how to read a
microGEM score). Both entry points behave identically.

The assistant follows the workflow in `SKILL.md`: `info` to confirm it can run →
`selftest` to confirm the numbers are trustworthy → score → report the four-part
sentence (panel + score + tier + margin) → recommend cross-checks.

### 7.3 What is in the skill

| File | Contents |
|---|---|
| `skill/SKILL.md` | invocation, workflow, six recurring mistakes, reporting template, hard guardrails |
| `skill/references/what-it-is.md` | performance table, quantified label corrections, validation status |
| `skill/references/panels.md` | all 17 panel definitions and the adjudication rules |
| `skill/references/interpreting.md` | six misreadings, choosing a threshold, reporting template, when to stop |

### 7.4 Hard guardrails (written into the skill)

- Never describe microGEM as "better than" UniProt/KEGG/CAZy — it sits *before*
  them (produce candidates, then look them up)
- Never claim a candidate is validated — microGEM has zero wet-lab validation
- Never silently drop weak panels from a direction the user asked about
- If the enzyme is outside the 17 panels, say so and point at eggNOG/KEGG/dbCAN
- Always cite the AUPRC when quoting a probability

---

## 8. From scores to experiments

### 8.1 Recommended flow

```
metagenome ORFs ─► microGEM ─► candidate list
                                 ├─► dbCAN/CAZy (GLY; independent ground truth)
                                 ├─► UniProt/BRENDA (characterised homologues)
                                 ├─► MetaCyc (which pathway could this feed?)
                                 ├─► taxonomy (microbial or host? required for RDX)
                                 └─► wet lab
```

### 8.2 Suggested cross-checks

| Panel | Independent check |
|---|---|
| `GLY` | **dbCAN3 HMM**. microGEM's GLY was validated against dbCAN3 V15: 67/67 positives hit a CAZy HMM, 66/67 (98.5%) are glycoside hydrolases; only 5.7% of 630 negatives hit at all. **Agreement is expected; disagreement is the informative case.** |
| `RDX` | taxonomic annotation (host homology is the main false-positive source) |
| `BA` | use BEAUT instead |
| others | BLAST/UniProt for the nearest characterised homologue; AlphaFold/ESMFold for pocket inspection |

### 8.3 How many candidates to take

Depends on throughput. Two reference points:

- 96-well validation: sort by the prevalence-adjusted precision, take the top 88 (leaving 8 for controls)
- discovery screen: relax the cut-off and rely on a second computational filter
  (homology, taxonomy, domain architecture)

---

## 9. Performance and cost

### 9.1 Compute

| Workload | GPU | CPU |
|---|---|---|
| one sequence (~400 aa) | 0.2 s | 1.5 s (torch) / 3 s (numpy) |
| 1,000 sequences | 3 min | 25 min / 50 min |
| 100,000 sequences | 5.5 h | 42 h / 83 h |

The scoring itself (17 linear probes) is negligible: a 1280×17 matrix multiply.
**The bottleneck is always the embedding.**

### 9.2 Footprint

| File | Size |
|---|---|
| 17 panel weights, total | 244 KB |
| ESM-2 checkpoint (`.pt`) | 2.6 GB |
| NumPy copy (570 `.npy` files) | 2.5 GB |
| reference set + metadata | 76 KB |

The tiny weight files mean they can ship inside the package, live in git, and go
into a paper's supplementary material.

### 9.3 Cache location

```bash
microgem info | grep cache
```

Default `~/.cache/microgem/`; override with `MICROGEM_HOME`.

---

## 10. Limitations (please read)

1. **Zero wet-lab validation.** Every number is computational. Its methodological
   ancestor BEAUT tested 102 of 614,616 candidates experimentally (47 active);
   microGEM has tested none.
2. **8.2% of Rhea only.** 17 panels over 1,734 reactions of 19,005 master
   reactions. Outside these 17 directions microGEM has nothing to say.
3. **No substrate-specificity prediction.** `FATE-L-modifying` says "acts on a
   long-chain acyl chain", not which fatty acid. CAZy's family-level
   classification is finer on that axis.
4. **No EC numbers.** A panel spans several EC classes by design.
5. **No macromolecule substrates.** Histone/DNA/RNA-acting enzymes are out of scope.
6. **Output is a hypothesis, not a conclusion.** It sits *before* the databases
   (produce candidates), not *after* (retrieve known enzymes).

### When not to use microGEM

| Situation | Use instead |
|---|---|
| the chemistry is outside the 17 panels | eggNOG-mapper, KEGG, dbCAN, MetaCyc |
| the protein is already characterised | UniProt / BRENDA (with experimental evidence) |
| you need "which substrate" | CAZy families (carbohydrates) / structure-based docking |
| you need a validated enzyme | microGEM cannot help — say so |

---

## 11. Troubleshooting

### `backend: none`

Missing model or missing PyTorch. `microgem info` tells you which:

```bash
pip install torch fair-esm && microgem download
```

Or convert once on a torch-capable machine, then stay on NumPy:

```bash
microgem download --convert
```

### `microgem selftest` fails

| Message | Cause |
|---|---|
| `DRIFT` (deviation > 1e-3) | the embedding changed: wrong weight file, truncated download, changed tokenisation |
| `positive control failed` | probe or weight file is corrupt |
| `cross-panel false-positive rate too high` | same; check whether a different model file is in place |

Check the checkpoint with `md5sum` first, then re-run `microgem convert`.

### Out of memory

The NumPy backend is memory-mapped (one `.npy` per tensor) and keeps only a few
hundred MB resident; the torch backend loads the full 2.6 GB. If memory is tight,
use `--backend numpy`.

### Out of GPU memory

```bash
microgem score big.faa --batch-size 4
```

### Too slow

1. Confirm the GPU is in use: `microgem info` should say `backend: torch`, and add `--device cuda` explicitly.
2. Try `--limit 100` first and extrapolate before committing.
3. `--panel GLY` reduces the *reporting* work but **not** the embedding cost —
   all 17 probes share one vector, so that step cannot be skipped.

### Every score is zero

Almost always the sequences were converted to `X` — e.g. nucleotide input, or a
format problem. Check:

```bash
head -2 proteins.faa
```

It should be amino acids (uppercase `ACDEFGHIKLMNPQRSTVWY`). microGEM maps
unrecognised characters to `X` rather than deleting them, because deleting
residues in the middle of a sequence silently changes the pooled vector.

---

## 12. Reproducibility

| What | How |
|---|---|
| published AUPRC | `pipeline/bin/07_baseline_linear_probe.py` |
| the weights shipped here | `scripts/build_weights.py` (replays the CV loop, then refits on 100%) |
| operating-characteristic table | `scripts/build_operating_points.py` (from the OOF predictions) |
| reference set | `scripts/build_reference.py` |
| this installation | `microgem selftest` |

`build_weights.py` has a hard gate: it replays the CV loop and refuses to export
if the AUPRC it computes differs from the published value by more than 0.0005.
The weights in this package and the numbers in the paper are therefore the same
model.

---

## 13. Citation

If you use microGEM, please cite:

> Jun Xu, microGEM: sequence-based scoring of 17 microbial enzyme chemistries with an explicitly calibrated, distribution-ready classifier, https://github.com/xGroup5/microgem (MIT licence).

BibTeX:

```bibtex
@software{xu2026microgem,
  author  = {Xu, Jun},
  title   = {microGEM: sequence-based scoring of 17 microbial enzyme chemistries
             with an explicitly calibrated, distribution-ready classifier},
  year    = {2026},
  url     = {https://github.com/xGroup5/microgem},
  license = {MIT}
}
```

microGEM builds on the BEAUT pipeline and on Rhea-derived labels; please cite those
too where relevant.

The methodological source:

> Ding, Y., Luo, X., Guo, J., et al. (2025). Identification of gut microbial bile acid
> metabolic enzymes via an AI-assisted pipeline. *Cell*, 188(21), 6012–6027.e20.
> https://doi.org/10.1016/j.cell.2025.07.017

and the label source:

> Morgat, A., Lombardot, T., Coudert, E., et al. (2020). Enzyme annotation in UniProtKB
> using Rhea. *Bioinformatics*, 36(6), 1896–1901. https://doi.org/10.1093/bioinformatics/btz817

and the protein language model:

> Lin, Z., Akin, H., Rao, R., et al. (2023). Evolutionary-scale prediction of atomic-level
> protein structure with a language model. *Science*, 379(6637), 1123–1130.
> https://doi.org/10.1126/science.ade2574

---

## 14. Layout

```
microgem-skill/
├── install.sh                    # one-shot install (command + skill)
├── pyproject.toml                # pip packaging
├── microgem/                     # the Python package
│   ├── cli.py                    # command line
│   ├── score.py                  # scoring core
│   ├── embed.py                  # ESM-2 embedding (torch / numpy backends)
│   ├── numpy_model.py            # torch-free forward pass
│   ├── panels.py                 # the 17-panel registry
│   ├── selftest.py               # installation verification
│   └── assets/                   # weights (244 KB) + metadata + reference set
├── skill/                        # the AI skill bundle (copy into a skill root)
│   ├── SKILL.md
│   └── references/
├── scripts/                      # build scripts (export from the research pipeline)
├── tests/                        # 32 tests
├── docs/
│   ├── USER-GUIDE.md             # Chinese
│   └── USER-GUIDE.en.md          # this file
└── README.md
```

## 15. License

MIT. The ESM-2 weights follow fair-esm's MIT licence and are not redistributed
with this repository.
