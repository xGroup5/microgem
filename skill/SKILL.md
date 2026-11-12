---
name: microgem
description: "Score microbial proteins against microGEM's 17 enzyme panels (FATE-L modifying/releasing, FATE-S, SATE, SALE, AATE hydrolysis/redox/lyase/ligase/CoA, GLY, TRP, PLY, ARO, VIT, RDX, BA) from sequence alone, using a frozen ESM-2 + linear-probe model. Use when the user asks whether a protein is a lipase, glycoside hydrolase, SAM enzyme, amino-acid transforming enzyme, tryptophan/indole, polyamine, aromatic-degradation, vitamin-biosynthesis, oxidative-stress or bile-acid enzyme; when screening a metagenome, MAG or gut-microbiome protein set for enzyme candidates; or when they mention microGEM, FATE, SATE, SALE, AATE, BEAUT-style enzyme mining, or gut-microbiome enzyme discovery. Also use for interpreting microGEM scores, choosing cut-offs, and reporting candidates."
whenToUse: "Any request to predict enzyme function from protein sequence in the microGEM panel set, to triage a candidate list before wet-lab validation, or to explain what a microGEM score does and does not mean. Not for: general protein annotation outside these 17 panels (use eggNOG/KEGG), structure prediction, or reactions on macromolecules."
# Slash entry point: typing `/microgem` in the composer loads this skill. Keep both
# flags explicit — a host default change must not silently disable the gesture.
user-invocable: true
disable-model-invocation: false
---

# microGEM — microbial enzyme scoring

microGEM answers one question: **given a protein sequence nobody has
characterised, is it likely to be one of these 17 enzyme chemistries?** It is a
screening tool that produces hypotheses, not an annotation database that reports
facts.

Read `references/what-it-is.md` before making claims about performance, and
`references/panels.md` when you need a panel definition or its caveats.

## The one thing that goes wrong

**The probability is not a posterior probability.** Each panel was trained with
balanced class weights on roughly 10% positives, so the output behaves like a
likelihood ratio. Concrete consequences, all measured out-of-fold:

| claim | reality |
|---|---|
| "0.9 means 90% chance it's a lipase" | At 0.5 the *precision* is 0.85 for FATE-L-modifying — and that is on data with 10% positives, which a metagenome is not |
| "the highest scoring panel is the answer" | Panels are not on a common scale; report the ranking and the margin, not a sum |
| "all 17 panels are equally usable" | AUPRC spans 0.503 (AATE-CoA) to 0.959 (BA) |

Always report the panel name, the score, the panel's tier, and the margin over
the runner-up. Never report a bare number.

## Invocation

Two entry points, same behaviour:

* **`/microgem`** (or `/microgem <task>`): the user typed the slash gesture. Its
  body is injected into the step; treat it as a direct instruction to run microGEM.
* **Automatic**: the catalog description matched, and you judged microGEM is the
  right tool.

When the gesture arrives **with no task attached** (bare `/microgem`), do not
start scoring something you picked yourself. Run the two cheap checks below, then
ask one short question:

```
microGEM is ready: 17 panels, backend <torch|numpy>, selftest PASS/FAIL.
What would you like scored? Point me at a FASTA file or paste sequences,
and say whether you want all 17 panels or a direction (e.g. GLY, FATE-L, AATE).
```

If the checks fail (no CLI, no model), say exactly what is missing and how to fix
it — do not ask the user a scoring question you cannot answer.

## Workflow

### 0. Resolve the command first

`microgem` may not be on `PATH` in every shell this runs in. Check once, and use
`$MG` for the rest of the session:

```bash
MG=$(command -v microgem \
     || command -v "$HOME/.local/bin/microgem" \
     || echo "")
[[ -n "$MG" ]] || { echo "microgem not installed"; exit 1; }
"$MG" --version
```

If neither resolves, the CLI is not installed (having the skill directory alone
is not enough) — tell the user to run `pip install /path/to/microgem-skill` and
stop there. Never guess a path, and never run the scoring code through a
different interpreter that may lack NumPy.

### 1. Check the install before promising anything

```bash
"$MG" info --format json
```

`backend: none` means no model is available — stop and tell the user to run
`"$MG" download` (2.6 GB). Do not attempt to fake a score.

```bash
"$MG" selftest             # ~15 s on GPU, ~2.5 min on CPU
```

Run this once per session if the user's conclusions depend on absolute scores.
`PASS` means the embeddings are the same ones the published numbers came from.

### 2. Score

```bash
"$MG" score proteins.faa --format json > microgem.json
"$MG" score proteins.faa --panel GLY            # one direction
"$MG" score proteins.faa --panel FATE-L         # a family (2 sub-panels)
"$MG" score proteins.faa --panel AATE           # a family (5 sub-panels)
```

`--format json` is what you want for downstream work: it carries per-panel
probabilities, the call decision at each panel's calibrated cut-off, the best
panel, and the margin. Cost is ~0.2 s/sequence on GPU, ~1.5 s/sequence on CPU —
tell the user before scoring 100k proteins.

Panel names are case-insensitive; `FATE`, `SAM` and `AATE` expand to families.

### 3. Read the result honestly

For each candidate, the reportable sentence has four parts:

> **{id}** — best panel **{panel}** at **{prob}** (tier **{tier}**, AUPRC
> {auprc}), **{margin}** ahead of **{runner-up}**.

Rules that follow from the numbers:

* **Tier `weak`** (AUPRC < 0.65: `AATE-CoA`, `SALE`) — never present as a
  finding. Mention only if the user asked for that specific direction, and say
  why it is weak.
* **Margin < 0.1** — the probe cannot separate two chemistries. Typical cases:
  FATE-L-modifying vs FATE-L-releasing (substrate-side ester vs free acid), and
  neighbouring AATE sub-panels. Report both.
* **Multiple calls** — a protein can legitimately clear several cut-offs
  (a SAM-dependent methyltransferase acting on a fatty acid). Report all of them
  in rank order; do not pick one silently.
* **`RDX`** — heavily shared with host enzymes. Ask whether the input is
  bacterial before interpreting; a human peroxiredoxin scores like a microbial one.
* **`BA`** — only 54 positives, and BEAUT (Ding et al., Cell 2025) covers bile
  acids with experimental validation. Recommend BEAUT for bile acids.

### 4. Turn scores into a prioritisation

When the user wants a shortlist rather than a score table:

```bash
"$MG" score proteins.faa --prevalence 0.001 --show-operating-points
```

`--prevalence` re-computes precision for a realistic prior. This matters: with
`--prevalence 0.001`, TRP's precision at its cut-off is 0.02, not 0.72. Use it
when the input is a whole metagenome rather than an enriched candidate set.

Rank by (tier, probability, margin) — not by probability alone, because a 0.62 on
`GLY` (AUPRC 0.86) is stronger evidence than a 0.80 on `SALE` (AUPRC 0.63).

### 5. Cross-check before the user spends money

The model gives no experimental evidence. State that explicitly. Then recommend,
in this order:

1. **dbCAN / CAZy HMM** for `GLY` — independent family-level truth; the panel was
   validated at 98.5% concordance, so a disagreement is informative.
2. **UniProt / BRENDA / Rhea** for anything already characterised.
3. **AlphaFold / ESMFold + pocket inspection** for a novel candidate.
4. **Taxonomic context** for `RDX` (host homology) and any gut-microbiome claim.

## Reporting template

```
## microGEM screen — {N} proteins, {panels}
Model: microGEM {version}, ESM-2 650M, backend {backend}; selftest {PASS/FAIL}

| id | best panel | score | tier | AUPRC | margin | also called |
|---|---|---|---|---|---|---|
| ... |

Top candidates for validation: ...
Not interpretable: AATE-CoA / SALE hits, and any call with margin < 0.1.
Limits: sequence-only prediction, no experimental validation, 8.2% of Rhea's
reaction space; panels do not distinguish substrate specificity within a family.
```

## Guardrails

* Never describe microGEM as "better than" UniProt/KEGG/CAZy — it sits *before*
  them in the pipeline (produce candidates, then look them up).
* Never claim a candidate is validated. microGEM has zero wet-lab validation;
  BEAUT, its methodological ancestor, validated 102 of 614,616 candidates.
* Never silently drop `weak` panels from a report the user asked for — mark them.
* If a user asks "is this enzyme X", and X is outside the 17 panels, say so and
  point at eggNOG/KEGG/dbCAN instead of forcing a panel.
* Cite the panel's AUPRC whenever you quote a probability. A probability without
  its panel's performance is uninterpretable.
