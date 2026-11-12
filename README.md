# microGEM

**Microbial enzyme scoring from protein sequence — 17 panels, one command, works as an AI skill.**

[中文说明](docs/USER-GUIDE.md) ｜ [English guide](docs/USER-GUIDE.en.md)

```bash
./install.sh                                   # command + AI skill
microgem download                              # 2.6 GB, once
microgem score proteins.faa                    # the answer
```

```python
from microgem import score
result = score("proteins.faa")
print(result.sequences[0].best.panel, result.sequences[0].best.probability)
```

---

## What it does

Given a protein sequence nobody has characterised, microGEM says which of 17
enzyme chemistries it is likely to be — and, more usefully, how much to trust
that answer.

| family | panels | the question |
|---|---|---|
| **FATE-L** | modifying, releasing | long-chain fatty acid chemistry |
| **FATE-S** | FATE-S | short-chain (C2–C6) acyl chemistry |
| **SATE** / **SALE** | SATE, SALE | SAM transfers a methyl, or is consumed otherwise |
| **AATE** | hydrolysis, redox, lyase, ligase, CoA | amino-acid transformation, split by chemistry |
| **GLY** | GLY | glycoside hydrolysis (CAZy class GH) |
| **TRP** | TRP | tryptophan → AhR-ligand branch |
| **PLY** | PLY | polyamine metabolism |
| **ARO** | ARO | aromatic ring chemistry |
| **VIT** | VIT | de-novo vitamin/cofactor biosynthesis |
| **RDX** | RDX | oxidative-stress defence |
| **BA** | BA | bile acid transformation |

The model: frozen ESM-2 650M → 1280-d → standardise → logistic regression, one
probe per panel. 244 KB of weights, NumPy at inference, no GPU required.

The evaluation: cluster-aware cross-validation (positives + negatives clustered
together, whole clusters held out), which is stricter than most published
enzyme-mining work.

## Performance

| panel | AUPRC | tier | panel | AUPRC | tier |
|---|---|---|---|---|---|
| BA | 0.959 | reliable | AATE-hydrolysis | 0.871 | reliable |
| AATE-ligase | 0.925 | reliable | GLY | 0.863 | reliable |
| PLY | 0.919 | reliable | SATE | 0.861 | reliable |
| RDX | 0.909 | reliable | VIT | 0.819 | moderate |
| FATE-L-releasing | 0.899 | reliable | TRP | 0.783 | moderate |
| FATE-L-modifying | 0.894 | reliable | AATE-redox | 0.751 | moderate |
| FATE-S | 0.891 | reliable | AATE-lyase | 0.715 | moderate |
| ARO | 0.890 | reliable | SALE | 0.628 | **weak** |
| | | | AATE-CoA | 0.503 | **weak** |

## Installing as an AI skill

`./install.sh` copies `skill/` into the agent's skill root. Invoke it with the
slash command:

```
/microgem                          # loads the skill, asks what to score
/microgem score candidates.faa     # or attach the task directly
```

The skill teaches the assistant the workflow *and* the interpretation rules — that
the probability is not a posterior, that panels are not comparable, that `SALE`
and `AATE-CoA` are weak, that `RDX` needs taxonomy, and that a margin under 0.1
means "cannot separate these two".

| root | command |
|---|---|
| DSH | `./install.sh --agent dsh` |
| Claude Code | `./install.sh --agent claude` |
| generic | `./install.sh --agent ~/.agents/skills` |
| anything else | `cp -R skill /path/to/skills/microgem` |

## Requirements

* Python ≥ 3.9, NumPy
* 2.6 GB for the ESM-2 checkpoint (not bundled)
* Either `torch` + `fair-esm` (fast path), or one conversion step to go NumPy-only

```bash
pip install torch fair-esm && microgem download --convert
# torch can be uninstalled after this; scoring only needs numpy
```

## Verify

```bash
microgem selftest       # 47 reference proteins; PASS means the scores are the model's
pytest tests/ -q        # 32 tests
```

## Limitations, stated plainly

* **Zero wet-lab validation.** BEAUT, the methodological ancestor, tested 102 of
  614,616 candidates. microGEM has tested none.
* **8.2% of Rhea.** 1,734 of 19,005 master reactions. Outside these directions,
  use eggNOG/KEGG/dbCAN/MetaCyc.
* **No substrate specificity.** `FATE-L-modifying` says "long-chain acyl", not
  which fatty acid.
* **A probability without its AUPRC is uninterpretable.** Always report both.

It sits *before* the databases — it produces candidates; they confirm what is
already known.

## Layout

```
microgem/            the Python package (cli, score, embed, panels, selftest, assets)
skill/               the AI skill bundle (SKILL.md + references)
scripts/             build scripts: export weights, operating points, reference set
tests/               32 tests, including torch↔numpy backend parity
docs/                USER-GUIDE.md (中文) and USER-GUIDE.en.md
install.sh           command + skill installer
```

## Citation

If you use microGEM, please cite:

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

> Jun Xu, microGEM: sequence-based scoring of 17 microbial enzyme chemistries with an explicitly calibrated, distribution-ready classifier, https://github.com/xGroup5/microgem (MIT licence).

microGEM builds on the BEAUT pipeline and on Rhea-derived labels; please cite those
too where relevant (see `docs/USER-GUIDE.md`, section 13).

## License

MIT. ESM-2 weights are not redistributed.
