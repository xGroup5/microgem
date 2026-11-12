# What microGEM is, and what its numbers mean

Every number in this file is measured out-of-fold under cluster-aware
cross-validation — positives and negatives clustered together at bitscore 200,
whole clusters assigned to one fold — so a protein family never appears on both
sides of a split. That protocol is stricter than what most enzyme-mining papers
report, and it is why some of these numbers are lower than you may expect.

## The model

```
sequence ─► ESM-2 650M (frozen, layer 33, mean-pooled, 1280-d)
         ─► standardise
         ─► logistic regression, C=1.0, class_weight=balanced
         ─► sigmoid  ─► probability per panel
```

Seventeen independent probes share one embedding. Each was fitted once on every
positive and negative available for its panel, after the evaluation loop in the
research pipeline had already produced the published AUPRC with the identical
estimator — `scripts/build_weights.py` re-runs that loop and refuses to export if
the AUPRC it computes differs from the published value by more than 0.0005.

Weights are ~14 KB per panel. The whole model is 244 KB and needs only NumPy at
inference time.

## Performance, out-of-fold

| panel | n | pos | clusters | AUC | AUPRC | tier |
|---|---|---|---|---|---|---|
| BA | 384 | 54 | 277 | 0.987 | 0.959 | reliable |
| AATE-ligase | 112,231 | 11,136 | 11,385 | 0.987 | 0.925 | reliable |
| PLY | 3,483 | 323 | 1,511 | 0.980 | 0.919 | reliable |
| RDX | 14,869 | 1,664 | 4,190 | 0.954 | 0.909 | reliable |
| FATE-L-releasing | 1,611 | 191 | 942 | 0.969 | 0.899 | reliable |
| FATE-L-modifying | 2,623 | 298 | 1,132 | 0.983 | 0.894 | reliable |
| FATE-S | 5,109 | 544 | 1,881 | 0.947 | 0.891 | reliable |
| ARO | 11,393 | 1,263 | 3,566 | 0.950 | 0.890 | reliable |
| AATE-hydrolysis | 24,437 | 2,417 | 4,322 | 0.969 | 0.871 | reliable |
| GLY | 697 | 67 | 356 | 0.990 | 0.863 | reliable |
| SATE | 21,049 | 2,089 | 4,787 | 0.951 | 0.861 | reliable |
| VIT | 16,119 | 1,644 | 4,431 | 0.926 | 0.819 | moderate |
| TRP | 3,904 | 449 | 1,625 | 0.951 | 0.783 | moderate |
| AATE-redox | 5,326 | 596 | 1,985 | 0.878 | 0.751 | moderate |
| AATE-lyase | 35,723 | 3,633 | 6,354 | 0.928 | 0.715 | moderate |
| SALE | 13,356 | 1,351 | 3,291 | 0.870 | 0.628 | weak |
| AATE-CoA | 3,161 | 361 | 1,410 | 0.737 | 0.503 | weak |

Tier cut-offs: AUPRC ≥ 0.85 reliable, 0.65–0.85 moderate, < 0.65 weak.

**Why AUPRC and not AUC.** AUC stays high (0.87–0.99) even for panels that rank
their own positives badly, because it does not care where the positives land
relative to each other — only relative to negatives. AUPRC does. Every label
correction in this project showed the same pattern: the AUC barely moved while
the AUPRC moved by 0.15–0.29.

## Three label corrections worth knowing

The panels were not defined once and frozen. Each of these changes came from
looking at which positives were misclassified, and each was a label bug rather
than a model limitation:

| correction | before | after |
|---|---|---|
| Split FATE-L into modifying / releasing | AUPRC 0.906 / 0.607 | 0.982 / 0.898 |
| Split SATE into SATE / SALE | 0.941 / 0.757 | 0.952 / 0.877 |
| Remove aldehyde dehydrogenases from FATE-S | 0.944 / 0.752 | 0.956 / 0.901 |

The FATE-S case is the clearest: EC 1.2.1.10 (acetaldehyde dehydrogenase) had all
111 of its sequences called negative, because its reaction
(`acetaldehyde + NAD(+) + CoA = acetyl-CoA`) starts from an *aldehyde*, not a
fatty acid. The regex had matched `acetaldehyde`.

## What "probability" is not

The probes output a sigmoid of a linear decision value. Because of balanced class
weights and a ~1:10 positive:negative ratio in training, that number is not a
calibrated posterior. Two consequences:

**Precision depends on your input's prevalence.** At each panel's cut-off:

| panel | precision @ 10% prev | @ 1% | @ 0.1% |
|---|---|---|---|
| SATE | 0.923 | 0.451 | 0.075 |
| AATE-ligase | 0.877 | 0.407 | 0.064 |
| TRP | 0.702 | 0.169 | 0.020 |

A metagenome is closer to 0.1–1% for any single enzyme class. This is why
`microgem score --prevalence 0.001` exists: it reports the precision you would
actually see rather than the precision on the balanced evaluation set.

**Panels are not comparable to each other.** A 0.8 on SALE and a 0.8 on PLY are
not equally strong evidence, because the two probes were trained against
different negatives with different class balances. Rank by (tier, probability,
margin), never by probability across panels alone.

## Validation status

| panel | independent evidence |
|---|---|
| GLY | dbCAN3 V15 HMM screen: 67/67 positives hit a CAZy HMM, 66/67 (98.5%) are glycoside hydrolases; 36/630 negatives hit at all (5.7%), of which 7 are GH |
| all others | none — the AUPRC is internal cross-validation on Rhea-derived labels |
| all | **no wet-lab validation** |

The methodological ancestor, BEAUT (Ding et al., *Cell* 2025), validated 102 of
614,616 candidates experimentally (47 active). microGEM has zero.

## Coverage

17 panels over 1,734 Rhea reactions = **8.2% of Rhea's 19,005 master reactions**.
KEGG and eggNOG cover the whole genome but do not distinguish substrates within a
protein family. Outside these 17 directions microGEM has nothing to say — use
eggNOG-mapper, KEGG, dbCAN or MetaCyc instead.

## Where it sits in a pipeline

```
metagenome ORFs ─► microGEM ─► candidate list
                                  ├─► CAZy/dbCAN (GLY), UniProt/BRENDA (known enzymes)
                                  ├─► MetaCyc (which pathway could this feed?)
                                  ├─► taxonomy (is this microbial or host?)
                                  └─► wet lab
```

microGEM produces candidates; it does not confirm them. The databases above
confirm what is already known; they cannot speak about an uncharacterised ORF,
which is 60–80% of a gut metagenome's predicted proteins.

## Reproduction

| what | how |
|---|---|
| published AUPRC | `pipeline/bin/07_baseline_linear_probe.py` |
| shipped weights | `scripts/build_weights.py` (replays the CV loop, then refits on 100%) |
| operating points | `scripts/build_operating_points.py` (from the out-of-fold predictions) |
| this install | `microgem selftest` |
