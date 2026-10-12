# Polyester-degrader benchmark v1.0

A homology-bounded benchmark for detecting **polyester-degrading enzymes** from protein
sequence, with the negative set graded by **fold and catalytic machinery** rather than by
enzyme class.

**847 proteins** — 811 modelled (388 positive, 423 negative), 36 unused.

If you are an AI agent working in this repository, read **`AGENTS.md`** first; it carries the
rules. `EVALUATION.md` defines how results must be reported.

---

## What a positive is

> **An enzyme experimentally reported to degrade a plastic whose backbone is joined by
> carboxylic ester bonds.**

A polyester is a polymer whose chain is held together by the ester linkage `-C(=O)-O-`.
Degrading it means hydrolysing that bond so the chain breaks. Every record comes from **PAZy**
or **PlasticDB**, which document measured activity on a named material. 340 of the 388 have
verified activity; 48 are reported but not independently confirmed.

### The 21 polyester substrates

Counts are enzymes recorded as acting on that substrate. They sum past 388 because one enzyme
often degrades several.

| substrate | full name | backbone bond | enzymes |
|---|---|---|---|
| **Aromatic polyesters** | | | |
| PET | polyethylene terephthalate | ester, aromatic diacid | **320** |
| PEF | polyethylene furanoate | ester, furan diacid | 7 |
| **Aromatic-aliphatic copolyesters** | | | |
| PBAT | poly(butylene adipate-*co*-terephthalate) | ester, mixed | 31 |
| PBSeT | poly(butylene sebacate-*co*-terephthalate) | ester, mixed | 2 |
| Ecovio-FT | commercial PBAT / PLA blend | ester, mixed | 2 |
| **Aliphatic polyesters** | | | |
| PLA | polylactic acid | ester, alpha-hydroxy acid | 52 |
| PCL | polycaprolactone | ester, lactone-derived | 43 |
| PBS | polybutylene succinate | ester, aliphatic diacid | 22 |
| PES | polyethylene succinate (**not** polyethersulfone) | ester, aliphatic diacid | 16 |
| PBSA | poly(butylene succinate-*co*-adipate) | ester, aliphatic diacid | 15 |
| PEA | polyethylene adipate | ester, aliphatic diacid | 3 |
| PPL | poly(beta-propiolactone) | ester, lactone-derived | 1 |
| **Polyhydroxyalkanoates — bacterial polyesters** | | | |
| PHA | polyhydroxyalkanoate | ester, beta-hydroxy acid | 23 |
| PHB | poly(3-hydroxybutyrate) | ester, beta-hydroxy acid | 11 |
| PHBV | poly(3-hydroxybutyrate-*co*-3-hydroxyvalerate) | ester, beta-hydroxy acid | 5 |
| P3HP | poly(3-hydroxypropionate) | ester, beta-hydroxy acid | 3 |
| P4HB | poly(4-hydroxybutyrate) | ester, omega-hydroxy acid | 3 |
| PHO | poly(3-hydroxyoctanoate) | ester, beta-hydroxy acid | 2 |
| PHBH | poly(3-hydroxybutyrate-*co*-3-hydroxyhexanoate) | ester, beta-hydroxy acid | 1 |
| PHBVH | PHB-*co*-valerate-*co*-hexanoate terpolymer | ester, beta-hydroxy acid | 1 |
| PHPV | poly(hydroxypropionate-*co*-valerate) | ester, beta-hydroxy acid | 1 |

Every bond in that column is an ester. Whatever the polymer is called, the enzyme's job is the
same reaction — which is what makes these 388 enzymes a coherent class.

### What is not here, and why

| plastic | backbone bond | why it is out |
|---|---|---|
| polyamide (nylon) | amide `-C(=O)-NH-` | hydrolysable, but by an **amidase** — a different catalytic machine |
| polyurethane | urethane `-NH-C(=O)-O-` | hydrolysable, different machine |
| polycarbonate | carbonate `-O-C(=O)-O-` | an ester, but not a carboxylic one; few enzymes act on it |
| polyether (PEG) | ether `-C-O-C-` | very stable; needs oxidative chemistry |
| polyolefins, vinyls (PE, PP, PS, PVC) | C-C only | **no hydrolysable bond exists** — oxidative attack only |
| rubber | C-C / C=C | same — nothing for a hydrolase to cut |

**Why the ester is the bond that works.** An ester's carbonyl carbon is electrophilic and its
leaving group is good, so a serine nucleophile attacks it at a modest barrier. In an **amide**
the nitrogen lone pair delocalises into the carbonyl, giving the C-N bond partial double-bond
character and raising the uncatalysed barrier by orders of magnitude. In a **C-C backbone**
there is no polarised bond and no heteroatom, so hydrolysis is not chemically available at
all; degradation requires radical chemistry with cofactors and oxygen.

There is also an evolutionary reason. **Cutin, the polymer of plant cuticle, is a natural
polyester**, so cutinases already existed when PET was invented in 1941, and PET's ester is
close enough that they partly work on it. Nothing in nature ever resembled polyethylene.

> **These plastics are not excluded for being unbreakable.** The source databases contain
> **65 polyamide-degrading and 15 polyurethane-degrading enzymes**, and 92 of the 118 excluded
> entries have experimentally verified activity. Nylonases and polyurethanases are real. They
> are excluded because a different bond needs a different catalytic machine, so pooling them
> with PETases would mean training one classifier on unrelated enzyme families and calling the
> mixture a plastic degrader. Only the polyolefins and rubber are excluded on chemistry alone.

---

## What a negative is

> **A protein never reported to degrade any plastic**, drawn from UniProt and graded by how
> closely it resembles a polyester hydrolase.

Every negative label is an assumption of inactivity, not a measurement — none of the 423 was
assayed on a plastic. The value is in the grading, which was measured by **fold** (InterPro
`IPR029058`) and by **curated catalytic residues**, not by enzyme class.

| `negative_group` | catalytic machinery | n | train | val | test |
|---|---|---|---|---|---|
| **`ab_hydrolase_hard_negative`** | Ser-His-Asp triad, **identical to the positives** | **51** | 40 | **1** | 10 |
| `ec311_wrong_fold` | serine nucleophile, other geometry (GDSL-SGNH, patatin, beta-lactamase) | 36 | 23 | 0 | 13 |
| `trna_processing` | histidine only — **no serine nucleophile** | 99 | 56 | 40 | 3 |
| `naive_control` | unrelated — kinases, gyrases, oxygenases | 237 | 166 | 35 | 36 |
| total | | 423 | 285 | 76 | 62 |

**Only the first row is a hard negative.** Those 51 carry the same fold and the same catalytic
triad as the positives and have simply never been reported to touch a plastic. Separating them
is the actual task.

The 99 `trna_processing` entries are D-aminoacyl-tRNA deacylase (Pfam PF02580) and
peptidyl-tRNA hydrolase (PF01195). They are EC 3.1.1 by *reaction chemistry* but share neither
the fold nor the machinery: of the 43 with a curated active site, every one is a single
histidine proton acceptor, and **not one has an annotated serine nucleophile**. They are kept
and labelled so you can exclude them deliberately.

---

## Splits

| split | positives | negatives | total | prevalence | hard negatives |
|---|---|---|---|---|---|
| train | 303 | 285 | 588 | 0.515 | 40 |
| val | 43 | 76 | 119 | 0.361 | **1** |
| test | 42 | 62 | 104 | 0.404 | **10** |

No cross-split pair exceeds 40% sequence identity: train/val 0.3967, train/test 0.3800,
val/test 0.3983.

`split == "unused"` holds 36 PHA depolymerases. They carry no claim and were never used for
training, tuning or evaluation.

---

## The two numbers that bound every claim

**51 hard negatives**, one of them in validation. And **66% of the positives are a single
homology component** — `BPS0190`, 256 of 388, with 74 components in all.

Together they are why the headline test AU-ROC of 0.9681 against all negatives becomes
**0.8167** against the 10 fold-matched negatives in test, with a 95% interval reaching 0.676.

Neither is fixable by modelling. Read `EVALUATION.md` before reporting anything.

---

## Provenance

**Positives** from PAZy and PlasticDB, each row traceable to a source id. 118 candidates were
excluded during assembly: 109 for a non-ester mechanism, 8 for a non-esterase fold (six
PLA-degrading proteases, two chitinases), and one manually — named a PETase by homology but
only ever assayed on PCL.

**Negatives** from UniProt, with a per-phylum quota matched to the positives (the PET class is
72.6% Actinomycetota) and a filter removing anything resembling a known plastizyme.

**Annotation** from the UniProt REST API — `ec`, `xref_pfam`, `xref_interpro`, `ft_act_site` —
fetched 2026-10-11.

### References

| citation | DOI |
|---|---|
| Danso D, Chow J, Streit WR (2019). Plastics: environmental and biotechnological perspectives on microbial degradation. *Appl Environ Microbiol* 85, e01095-19. — the hydrolysable vs C-C backbone split | `10.1128/AEM.01095-19` |
| Wei R, Tiso T, Bertling J, O'Connor K, Blank LM, Bornscheuer UT (2020). Possibilities and limitations of biotechnological plastic degradation and recycling. *Nature Catalysis* 3, 867-871. — why the task is scoped to polyesters | `10.1038/s41929-020-00521-w` |
| **Buchholz PCF et al. (2022). Plastics degradation by hydrolytic enzymes: the plastics-active enzymes database — PAZy. *Proteins* 90, 1443-1456.** — **cite this if you use the positives** | `10.1002/prot.26325` |
| Wei R, Zimmermann W (2017). Microbial enzymes for the recycling of recalcitrant petroleum-based plastics: how far are we? *Microbial Biotechnology* 10, 1308-1322. — PE/PS/PUR/PET compared | `10.1111/1751-7915.12710` |
