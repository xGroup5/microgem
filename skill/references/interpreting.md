# Interpreting scores, and the mistakes that are easy to make

## The four numbers to read

For every sequence microGEM reports:

| field | what it is | what to do with it |
|---|---|---|
| `scores[panel]` | sigmoid of a linear decision value | compare *within* a sequence, not across sequences |
| `called` | panels above their own calibrated cut-off | the candidate list |
| `best` | highest-scoring panel | the hypothesis to test |
| `margin` | best − runner-up | < 0.1 means the probe cannot separate them |

## Mistake 1: reading the probability as a probability

At a panel's cut-off, out-of-fold precision is 0.70–1.00 on a set with ~10%
positives. On a real metagenome at 0.1% prevalence, the same cut-off yields
precision of 0.02–0.09 for most panels. Both numbers are correct; only one is
about your data.

```bash
"$MG" score proteins.faa --prevalence 0.001 --show-operating-points
```

Use `--prevalence` whenever the input is a whole metagenome, a MAG collection, or
any set not already enriched for the chemistry you are looking for.

## Mistake 2: comparing panels by score

A 0.85 on `SALE` (AUPRC 0.63) is weaker evidence than a 0.65 on `GLY` (AUPRC
0.86). The panels were trained against different negative sets and different class
balances; only their *tiers* are meant to be compared, and even then only loosely.

Correct ranking when you must produce one:

```
1. panels with tier = reliable
2. within a tier, descending probability
3. tie-break on margin (a clear winner beats a crowded top)
```

## Mistake 3: treating a small margin as a call

A margin below 0.1 means two probes fire together. The three recurring cases:

| pattern | what it means |
|---|---|
| `FATE-L-modifying` + `FATE-L-releasing` both high | the protein sits on the ester/free-acid boundary; the panels split by substrate side, and a promiscuous esterase looks like both |
| two neighbouring `AATE-*` panels | the reaction is at the boundary of the chemistry split (e.g. a dehydrogenase that also runs a hydrolysis step) |
| `SATE` + `SALE` | the enzyme binds SAM but the product chemistry is ambiguous in the probe's representation |

Report both panels and say the probe cannot separate them. Do not pick one.

## Mistake 4: ignoring the weak panels

`AATE-CoA` (AUPRC 0.503) and `SALE` (0.628) are shipped because a complete panel
set is more useful than a silently truncated one, not because they work.

* `AATE-CoA`: 57.6% of its positives come from a single reaction (EC 2.3.1.1),
  and the succinyl-CoA subclass is called wrong in every fold. 30-seed AUPRC
  0.62 ± 0.068.
* `SALE`: chemically coherent, but the positive set spans several unrelated folds.
  A high score means "SAM-consuming, not transmethylating" and nothing more.

If a user asks specifically about these directions, say the panel is weak *before*
quoting a number.

## Mistake 5: RDX without taxonomy

`RDX` (glutathione, thioredoxin, peroxiredoxin, catalase) shares heavy homology
with host enzymes. A human peroxiredoxin scores like a microbial one. For any
gut-microbiome input, check that the candidate is actually bacterial — taxonomy
from the MAG/ORF caller, or a quick best-hit check — before interpreting an RDX
score.

## Mistake 6: quoting a probability without its AUPRC

> "This protein is a glycoside hydrolase with 0.91 confidence."

is not a reportable sentence. This is:

> "This protein scores 0.91 on `GLY` (AUPRC 0.863, tier reliable), with no
> competing panel above its cut-off."

The AUPRC is what makes the score interpretable; without it the probability is a
number with no scale.

## Choosing a threshold

Three options, in increasing order of how much you have thought about it:

1. **Default** — each panel's F1-optimal cut-off on the out-of-fold predictions,
   clamped to [0.20, 0.90]. This is what `microgem score` uses.
2. **`--threshold 0.5`** — the naive rule. It is *not* calibrated: at 0.5 recall
   ranges from 0.32 (AATE-CoA) to 0.92 (AATE-ligase) across panels.
3. **Explicit precision/recall targeting** — use `--show-operating-points` to read
   precision and recall at the cut-off, then adjust `--threshold` per panel until
   the trade-off matches what the experiment can afford.

A validation experiment with 96-well capacity wants precision; a discovery screen
that will be followed by a second computational filter can afford recall.

## Reporting template

```
## microGEM screen — {N} proteins, {panels}

| id | best panel | score | tier | AUPRC | margin | also called |
|---|---|---|---|---|---|---|
| contig_12_orf3 | GLY | 0.94 | reliable | 0.863 | 0.71 | — |

**Candidate for validation**: contig_12_orf3, predicted glycoside hydrolase.
Nearest characterised homologue: {from BLAST/UniProt}.
Suggested assay: {…}.

**Not interpretable**: 3 hits with margin < 0.1 (listed above), 1 AATE-CoA hit
(weak panel, AUPRC 0.503).

**Limits**: sequence-only prediction; no experimental validation for any panel;
covers 8.2% of Rhea's reaction space; does not resolve substrate specificity
within a panel.
```

## When to stop using microGEM

* The chemistry is outside the 17 panels → eggNOG-mapper, KEGG, dbCAN, MetaCyc.
* The protein is already characterised → UniProt/BRENDA answer better and with
  experimental evidence.
* The question is "which substrate" rather than "which chemistry" → CAZy families
  (for carbohydrates) or a docking/structure workflow.
* The user needs a validated enzyme, not a candidate → microGEM cannot help; say so.
