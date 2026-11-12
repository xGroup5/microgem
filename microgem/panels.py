"""Panel registry and per-panel reliability tiers.

The descriptions are the ones written in ``pipeline/lib/panels.py`` — the single
definition site for every label in the project — reproduced here so the runtime
does not need the research pipeline on disk.  If a panel definition changes
there, ``scripts/build_weights.py`` must be re-run and this file updated with it.

The tiers are not editorial preference.  They are read off the cluster-aware
out-of-fold AUPRC in ``assets/manifest.json`` and the 30-seed spread recorded in
the build guide:

    AUPRC >= 0.85   reliable      the panel separates its chemistry from the
                                  other 16 reliably; use it for screening
    0.65 - 0.85     moderate      directional signal; treat the score as a
                                  prior to be confirmed, not a verdict
    < 0.65          weak          known label-bound; only use with an external
                                  cross-check (see docs)

``AATE-CoA`` is the clearest case of a label problem rather than a model
problem: 0.62 ± 0.068 across 30 seeds.  The panel exists for completeness, and
the CLI says so whenever it is selected, because a 0.503 AUPRC printed next to
0.925 from ``AATE-ligase`` invites exactly the wrong comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

ASSET_PACKAGE = "microgem.assets"


@dataclass(frozen=True)
class PanelInfo:
    """Everything the runtime knows about one panel, including its caveats."""

    name: str
    family: str
    full_name: str
    description: str
    tier: str
    auprc: float
    auc: float
    n_pos: int
    n_neg: int
    caveat: str = ""

    @property
    def prevalence(self) -> float:
        return self.n_pos / (self.n_pos + self.n_neg)


# ── panel definitions ──────────────────────────────────────────────────────
#
# `family` matches pipeline/lib/panels.py; several panels share a family
# (FATE-L-modifying and FATE-L-releasing are both FATE-L), and the CLI groups by
# family when it reports, because that is how the build guide discusses them.

_DEFINITIONS: tuple[dict, ...] = (
    {
        "name": "FATE-L-modifying",
        "family": "FATE-L",
        "full_name": "Fatty Acyl Transformation Enzyme — Long chain (chain-modifying)",
        "description": (
            "A free long-chain fatty acid or its CoA/ACP thioester is the substrate: "
            "desaturase, hydratase, cyclopropane synthase, CoA ligase, thioesterase."
        ),
    },
    {
        "name": "FATE-L-releasing",
        "family": "FATE-L",
        "full_name": "Fatty Acyl Transformation Enzyme — Long chain (ester-releasing)",
        "description": (
            "A lipid ester (sterol ester, glycerolipid, phospholipid, wax ester, "
            "acylcarnitine) is the substrate and a long-chain fatty acid is a product: "
            "lipase / esterase chemistry, hydrolytic only."
        ),
    },
    {
        "name": "FATE-S",
        "family": "FATE-S",
        "full_name": "Fatty Acyl Transformation Enzyme — Short chain",
        "description": (
            "A short-chain (C2-C6) fatty acid or acyl-CoA is the substrate: "
            "short-chain dehydrogenase, thioesterase, CoA transferase. Aldehyde "
            "dehydrogenases are excluded on purpose — the substrate there is an aldehyde."
        ),
    },
    {
        "name": "SATE",
        "family": "SATE",
        "full_name": "SAM-Associated Transmethylation Enzyme",
        "description": (
            "S-adenosyl-L-methionine donates a methyl group and the reaction produces "
            "S-adenosyl-L-homocysteine. Radical SAM enzymes are excluded: they bind SAM "
            "but do not transfer a methyl, which is a different structural family."
        ),
    },
    {
        "name": "SALE",
        "family": "SALE",
        "full_name": "SAM-Associated Lyase Enzyme",
        "description": (
            "SAM is consumed without transferring a methyl: radical SAM "
            "(5'-deoxyadenosine product) and aminopropyl/allyl transfer."
        ),
        "caveat": (
            "Weakest precise panel after AATE-CoA. Label is chemically coherent "
            "(radical SAM is a real family) but the positive set spans several "
            "unrelated folds, so a high score means 'SAM-consuming, not "
            "transmethylating' rather than a specific reaction."
        ),
    },
    {
        "name": "AATE-hydrolysis",
        "family": "AATE",
        "full_name": "Amino Acid Transformation Enzyme — hydrolysis",
        "description": (
            "Water is consumed breaking a bond of a free amino acid: deaminase, "
            "amidase, deamidase, hydantoinase."
        ),
    },
    {
        "name": "AATE-redox",
        "family": "AATE",
        "full_name": "Amino Acid Transformation Enzyme — oxidation/reduction",
        "description": (
            "NAD(P)/FAD/FMN/O2-dependent conversion of a free amino acid: "
            "dehydrogenase, oxidase, monooxygenase/hydroxylase."
        ),
        "caveat": (
            "Mixes four mechanisms (53 dehydrogenases, 47 oxidases, 36 hydroxylases, "
            "26 deaminating dehydrogenases). The score says 'redox on an amino acid', "
            "not which of the four."
        ),
    },
    {
        "name": "AATE-lyase",
        "family": "AATE",
        "full_name": "Amino Acid Transformation Enzyme — lyase/isomerase",
        "description": (
            "Carbon-carbon or carbon-nitrogen bond cleavage and rearrangement without "
            "water or a redox cofactor: decarboxylase, ammonia-lyase, racemase, "
            "aminotransferase."
        ),
    },
    {
        "name": "AATE-ligase",
        "family": "AATE",
        "full_name": "Amino Acid Transformation Enzyme — ligase/kinase",
        "description": (
            "ATP-dependent bond formation on an amino acid: aminoacyl-tRNA synthetase, "
            "peptide ligase, adenylation. The largest label pool in microGEM."
        ),
    },
    {
        "name": "AATE-CoA",
        "family": "AATE",
        "full_name": "Amino Acid Transformation Enzyme — CoA transfer",
        "description": (
            "Acyl-CoA-dependent acylation of an amino acid: N-acyltransferase "
            "(N-acetyl, N-succinyl), CoA transferase."
        ),
        "caveat": (
            "Not usable for screening. 57.6% of positives come from one reaction "
            "(EC 2.3.1.1) and the succinyl-CoA subclass is called wrong in every "
            "fold; 30-seed AUPRC is 0.62 +/- 0.068. Kept for completeness."
        ),
    },
    {
        "name": "TRP",
        "family": "TRP",
        "full_name": "Tryptophan and Indole Metabolism",
        "description": (
            "Tryptophan is the substrate of a reaction producing an indole, tryptamine, "
            "kynurenine or indole-3-pyruvate derivative — the aryl hydrocarbon receptor "
            "ligand branch that shapes IL-22 and the Treg/Th17 balance."
        ),
    },
    {
        "name": "PLY",
        "family": "PLY",
        "full_name": "Polyamine Metabolism",
        "description": (
            "Putrescine, spermidine, spermine, agmatine or cadaverine is the substrate, "
            "including aminopropyl transfer from decarboxylated SAM. Ornithine is "
            "admitted only through ornithine decarboxylase, by its product."
        ),
    },
    {
        "name": "ARO",
        "family": "ARO",
        "full_name": "Aromatic Compound Degradation",
        "description": (
            "A benzoate, catechol, protocatechuate, phenylacetate or similar aromatic "
            "ring is the substrate — the ring-cleavage chemistry downstream of aromatic "
            "amino acid catabolism, and a precursor route to uraemic toxins."
        ),
    },
    {
        "name": "VIT",
        "family": "VIT",
        "full_name": "Vitamin and Cofactor Biosynthesis",
        "description": (
            "De-novo biosynthesis of biotin, thiamine, riboflavin, cobalamin, folate, "
            "menaquinone, pantothenate, pyridoxal or lipoate. THF used as a one-carbon "
            "carrier is excluded — that is one-carbon metabolism, not vitamin synthesis."
        ),
        "caveat": (
            "These pathways are highly conserved and well studied, so a high score is "
            "usually a known enzyme. Best used as a method control, not a discovery panel."
        ),
    },
    {
        "name": "RDX",
        "family": "RDX",
        "full_name": "Oxidative Stress Defence",
        "description": (
            "Glutathione, thioredoxin, peroxiredoxin, catalase or a hydroperoxide is "
            "the substrate."
        ),
        "caveat": (
            "Heavy homology with host enzymes, so bacterial specificity is the main "
            "false-positive risk: a human peroxiredoxin scores like a microbial one. "
            "Filter candidates by taxonomy before interpreting."
        ),
    },
    {
        "name": "GLY",
        "family": "GLY",
        "full_name": "Glycoside Hydrolase",
        "description": (
            "A glycoside is hydrolysed (water consumed) to release a sugar; "
            "glycosyltransferases using nucleotide-sugar donors are excluded, so the "
            "panel corresponds to CAZy class GH rather than GH+GT."
        ),
        "caveat": (
            "Validated against dbCAN3 V15 independently: 67/67 positives hit a CAZy HMM "
            "and 66/67 are glycoside hydrolases (98.5%), while only 5.7% of negatives "
            "hit at all. The one exception is CE16, a carbohydrate esterase."
        ),
    },
    {
        "name": "BA",
        "family": "BA",
        "full_name": "Bile Acid Transformation",
        "description": (
            "A bile acid (cholate, chenodeoxycholate, deoxycholate, lithocholate and "
            "their conjugates) is the substrate."
        ),
        "caveat": (
            "Only 54 positives (277 clusters), so the confidence interval on AUPRC is "
            "wide. BEAUT (Ding et al., Cell 2025) covers this direction with far more "
            "data and experimental validation; prefer it for bile acids. Also note the "
            "positives here are eukaryotic and bacterial enzymes mixed."
        ),
    },
)


# ── tiers ───────────────────────────────────────────────────────────────────

_TIER_CUTOFFS = ((0.85, "reliable"), (0.65, "moderate"), (0.0, "weak"))


def _tier(auprc: float) -> str:
    for cutoff, label in _TIER_CUTOFFS:
        if auprc >= cutoff:
            return label
    return "weak"


@lru_cache(maxsize=1)
def _manifest() -> dict[str, dict]:
    with resources.files(ASSET_PACKAGE).joinpath("manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build() -> dict[str, PanelInfo]:
    manifest = _manifest()
    missing = [row["name"] for row in _DEFINITIONS if row["name"] not in manifest]
    if missing:
        raise RuntimeError(
            f"assets/manifest.json has no entry for: {', '.join(missing)}. "
            "Re-run scripts/build_weights.py."
        )
    extra = set(manifest) - {row["name"] for row in _DEFINITIONS}
    if extra:
        raise RuntimeError(
            f"assets/manifest.json describes panels this build does not know: "
            f"{', '.join(sorted(extra))}. Update microgem/score.py or rebuild."
        )
    panels: dict[str, PanelInfo] = {}
    for row in _DEFINITIONS:
        metrics = manifest[row["name"]]
        panels[row["name"]] = PanelInfo(
            name=row["name"],
            family=row["family"],
            full_name=row["full_name"],
            description=row["description"],
            tier=_tier(metrics["auprc"]),
            auprc=metrics["auprc"],
            auc=metrics["auc"],
            n_pos=metrics["n_pos"],
            n_neg=metrics["n_neg"],
            caveat=row.get("caveat", ""),
        )
    return panels


PANELS: dict[str, PanelInfo] = _build()

# Claim priority, as in pipeline/lib/panels.py. The order decides which panel
# "owns" a sequence when several are above threshold; it is not alphabetical and
# not by performance.
PANEL_ORDER: tuple[str, ...] = tuple(row["name"] for row in _DEFINITIONS)
