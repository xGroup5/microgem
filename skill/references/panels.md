# The 17 panels

A panel is defined by **what the enzyme acts on**, not by EC number and not by
what the reaction produces. Two consequences matter in practice: several panels
span multiple EC classes, and two enzyme families that share an EC can sit in
different panels because their chemistry differs.

All panels are pairwise disjoint — 45 cross-checks, zero reaction overlap — which
is what makes the scores comparable enough to rank within a sequence.

## Families

| family | panels | the question it answers |
|---|---|---|
| **FATE-L** | `FATE-L-modifying`, `FATE-L-releasing` | long-chain fatty acid chemistry |
| **FATE-S** | `FATE-S` | short-chain (C2–C6) acyl chemistry |
| **SATE** | `SATE` | does SAM transfer a *methyl*? |
| **SALE** | `SALE` | does the enzyme consume SAM *without* methyl transfer? |
| **AATE** | `AATE-hydrolysis`, `-redox`, `-lyase`, `-ligase`, `-CoA` | amino-acid transformation, split by reaction chemistry |
| **GLY** | `GLY` | glycoside hydrolysis (CAZy class GH, not GT) |
| **TRP** | `TRP` | tryptophan → AhR-ligand branch |
| **PLY** | `PLY` | polyamine synthesis and interconversion |
| **ARO** | `ARO` | aromatic ring cleavage and modification |
| **VIT** | `VIT` | de-novo vitamin/cofactor biosynthesis |
| **RDX** | `RDX` | oxidative-stress defence |
| **BA** | `BA` | bile acid transformation |

## Definitions

### FATE-L-modifying
A free long-chain fatty acid, or its CoA/ACP thioester, is the **substrate**:
desaturase, hydratase, cyclopropane synthase, CoA ligase, thioesterase,
acyltransferase acting on the chain.

### FATE-L-releasing
A lipid ester (sterol ester, glycerolipid, phospholipid, wax ester, acylcarnitine)
is the substrate and a long-chain fatty acid is a **product**: lipase/esterase.
Strictly hydrolytic — an acyltransferase run in the synthetic direction belongs to
neither FATE-L panel.

The split exists because a single label taught the model both chemistries at once
and scored 0.607 AUPRC. Split, they score 0.894 and 0.899.

### FATE-S
A short-chain (C2–C6) fatty acid or acyl-CoA is the substrate: short-chain
dehydrogenase, thioesterase, CoA transferase.

**Aldehyde dehydrogenases are excluded.** `acetaldehyde + NAD(+) + CoA =
acetyl-CoA` starts from an aldehyde, not a fatty acid; including EC 1.2.1.10 made
the panel call all 111 of its members negative and cost ~0.15 AUPRC.

### SATE
SAM donates a methyl group and the reaction produces S-adenosyl-L-homocysteine.
Includes metK (`L-methionine + ATP → SAM`), because that is the source of the
whole pathway.

### SALE
SAM is consumed **without** methyl transfer: radical SAM (5'-deoxyadenosine
product) and aminopropyl/allyl transfer. Split out of SATE because radical SAM
enzymes bind SAM but are a structurally different family — merging them left 40.8%
of positives scoring below 0.5.

### AATE sub-panels
Split by reaction chemistry, not EC: 131 of 348 AATE reactions carry no EC number,
so an EC-based split would silently drop 40% of the label. Priority order is redox
cofactors > ATP > CoA > H2O > everything else, because an O2/NAD-dependent reaction
is an oxidation no matter what else it also consumes.

| sub-panel | reaction signature |
|---|---|
| `AATE-hydrolysis` | water consumed breaking a bond: deaminase, amidase, hydantoinase |
| `AATE-redox` | NAD(P)/FAD/FMN/O2-dependent: dehydrogenase, oxidase, monooxygenase |
| `AATE-lyase` | C–C or C–N cleavage/rearrangement without water or redox: decarboxylase, ammonia-lyase, racemase, aminotransferase |
| `AATE-ligase` | ATP-dependent bond formation: aminoacyl-tRNA synthetase, peptide ligase |
| `AATE-CoA` | acyl-CoA-dependent N-acylation: N-acetyltransferase, CoA transferase |

### GLY
A glycoside is hydrolysed (water consumed) to release a sugar. Nucleotide-sugar
donors (UDP-/GDP-/ADP-sugar) are excluded, so this maps to CAZy class **GH**, not
GH+GT — which is what makes the dbCAN cross-validation meaningful.

Both ends are checked: a glycosidic substrate on the left **and** a free sugar on
the right. Without the right-hand check, EC 3.5.1.25 (a deacetylase) and 3.5.1.26
(an asparaginase) slip in because their substrates contain N-acetylglucosamine.

### TRP
Tryptophan is the substrate of a reaction producing an indole, tryptamine,
kynurenine or indole-3-pyruvate derivative. These are aryl hydrocarbon receptor
ligands that shape IL-22 production, barrier integrity and the Treg/Th17 balance;
faecal tryptophan metabolites are depleted in IBD and correlate inversely with
disease activity.

### PLY
Putrescine, spermidine, spermine, agmatine or cadaverine is the substrate,
including aminopropyl transfer from decarboxylated SAM.

**Ornithine is excluded by name** and admitted only through ornithine
decarboxylase, by its product. The bare name pulled in ornithine
carbamoyltransferase, ornithine acetyltransferase and acetylornithine
aminotransferase — urea-cycle and arginine-catabolism enzymes that made up 70% of
PLY's misclassified positives.

### ARO
A benzoate, catechol, protocatechuate, phenylacetate, salicylate or similar
aromatic ring is the substrate. Ring-cleavage chemistry downstream of aromatic
amino acid catabolism, and a precursor route to uraemic toxins that reach the
liver via the portal vein.

### VIT
De-novo biosynthesis of biotin, thiamine, riboflavin, cobalamin, folate,
menaquinone, pantothenate, pyridoxal or lipoate.

**Tetrahydrofolate as a one-carbon carrier is excluded.** Only the de-novo folate
route is admitted; when THF appears as a co-substrate the enzyme is doing
one-carbon metabolism, not vitamin synthesis. Including both pulled in glycine
hydroxymethyltransferase, phosphoribosylglycinamide formyltransferase and
formyl-THF synthetase — 911 sequences that scored near zero.

### RDX
Glutathione, thioredoxin, peroxiredoxin, catalase or a hydroperoxide is the
substrate. Homology with host enzymes is heavy, so bacterial specificity is the
main false-positive risk.

### BA
A bile acid (cholate, chenodeoxycholate, deoxycholate, lithocholate and their
conjugates) is the substrate. Only 54 positives; BEAUT covers this direction with
far more data and experimental validation.

## Claim priority

When several panels match the same reaction, the earlier one in this list wins and
the reaction is removed from the others:

```
BA > GLY > SALE > SATE > PLY > TRP > ARO > VIT > RDX
   > FATE-L-modifying > FATE-L-releasing > FATE-S
   > AATE-CoA > AATE-ligase > AATE-lyase > AATE-redox > AATE-hydrolysis
```

Without this, adding a panel forces an exclusion edit in every panel it overlaps
with — a rule set that grows quadratically and breaks silently whenever one panel
is forgotten.

## What the panels do not do

* **They do not predict substrate specificity.** `FATE-L-modifying` says "this
  acts on a long-chain acyl chain", not which fatty acid. CAZy's family-level
  classification is finer on this axis.
* **They do not assign EC numbers.** A panel spans several EC classes by design.
* **They do not cover 91.8% of Rhea.** 1,734 of 19,005 master reactions.
* **They do not handle macromolecule substrates.** Histone/DNA/RNA-acting enzymes
  are out of scope, as in BEAUT's Workflow II.
