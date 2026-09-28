"""Merge PAZy and PlasticDB into one non-redundant, fully-annotated plastizyme table.

Both sources record only *positives* (enzymes shown to degrade some plastic), so this
builds the positive side of the dataset plus the fold-matched contrast classes. It does
not touch UniProt; bulk negatives are pulled separately.

Dedup runs in two passes because the same protein appears in the two databases in
different forms: exact sequence match, then substring containment (one source stores the
precursor with its signal peptide, the other the mature form). Exact-match dedup alone
leaves ~100 near-duplicate pairs that would later leak across a homology split.

Usage:
    uv run python -m biopet_sae.build_dataset [--raw data/raw] [--out data/processed]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Substrate taxonomy.
#
# Grouped by the chemistry of the bond being attacked, because that determines
# whether two enzymes are even mechanistically comparable. PET is separated from
# the other polyesters because it is the only *crystalline aromatic* one in the
# set: its terephthalate units stiffen the backbone, so attacking it needs a wide
# shallow substrate cleft and thermostability that aliphatic-ester hydrolysis
# does not.
# ---------------------------------------------------------------------------

SUBSTRATE_GROUPS: dict[str, str] = {
    # aromatic polyester - the target class
    "PET": "PET_AROMATIC_POLYESTER",
    "PEF": "PET_AROMATIC_POLYESTER",  # polyethylene furanoate, the furan analogue
    # aliphatic-aromatic co-polyesters: contain terephthalate but flexible aliphatic
    # segments, so they are the intermediate case between PET and pure aliphatics
    "PBAT": "AROMATIC_COPOLYESTER",
    "PBSeT": "AROMATIC_COPOLYESTER",
    "Ecovio-FT": "AROMATIC_COPOLYESTER",
    # purely aliphatic polyesters - readily hydrolysed, the fold-matched contrast
    "PCL": "ALIPHATIC_POLYESTER",
    "PLA": "ALIPHATIC_POLYESTER",
    "PBS": "ALIPHATIC_POLYESTER",
    "PBSA": "ALIPHATIC_POLYESTER",
    "PES": "ALIPHATIC_POLYESTER",
    "PEA": "ALIPHATIC_POLYESTER",
    "PPL": "ALIPHATIC_POLYESTER",
    # bacterial polyhydroxyalkanoate storage polymers - aliphatic esters, but
    # degraded by a distinct depolymerase lineage rather than cutinases/lipases
    "PHA": "PHA_FAMILY",
    "PHB": "PHA_FAMILY",
    "PHBV": "PHA_FAMILY",
    "PHO": "PHA_FAMILY",
    "P4HB": "PHA_FAMILY",
    "P3HP": "PHA_FAMILY",
    # carbonate ester rather than carboxylic ester
    "PC": "POLYCARBONATE",
    # different bond entirely - excluded from any ester-hydrolysis comparison
    "PA": "POLYAMIDE",
    "Nylon": "POLYAMIDE",
    "PU": "POLYURETHANE",
    "PUR": "POLYURETHANE",
    "NR": "RUBBER",
    "PE": "POLYOLEFIN_VINYL",
    "LDPE": "POLYOLEFIN_VINYL",
    "HDPE": "POLYOLEFIN_VINYL",
    "PP": "POLYOLEFIN_VINYL",
    "PS": "POLYOLEFIN_VINYL",
    "PVC": "POLYOLEFIN_VINYL",
    "PVA": "POLYOLEFIN_VINYL",
    "O-PVA": "POLYOLEFIN_VINYL",
    "PEG": "POLYETHER",
}

# which bond the enzyme has to break, per group
MECHANISM: dict[str, str] = {
    "PET_AROMATIC_POLYESTER": "carboxylic_ester",
    "AROMATIC_COPOLYESTER": "carboxylic_ester",
    "ALIPHATIC_POLYESTER": "carboxylic_ester",
    "PHA_FAMILY": "carboxylic_ester",
    "POLYCARBONATE": "carbonate_ester",
    "POLYAMIDE": "amide",
    "POLYURETHANE": "urethane",
    "RUBBER": "c_c_oxidative",
    "POLYOLEFIN_VINYL": "c_c_oxidative",
    "POLYETHER": "ether",
}

PET_SUBSTRATES = {"PET"}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ---------------------------------------------------------------------------
# Class assignment, implementing decision D1 in PROGRESS.md.
#
# Encoded here rather than re-derived downstream so every script agrees on the
# labels and the reasoning travels with the data.
# ---------------------------------------------------------------------------

CLASS_PET = "1_pet"
CLASS_OTHER_POLYESTER = "2_other_polyester"
CLASS_HELDOUT_PHA = "heldout_pha"
CLASS_EXCLUDED = "excluded"

# Class 2 is the fold-matched contrast: enzymes assayed on a non-PET polyester that
# are cutinase/lipase/esterase-like. PHA is deliberately not here -- it is a distinct
# depolymerase lineage, held out as a cross-activation probe instead.
CLASS2_GROUPS = {"ALIPHATIC_POLYESTER", "AROMATIC_COPOLYESTER"}

# Curated exclusions, keyed by source id so they survive any change in row ordering.
MANUAL_EXCLUSIONS: dict[str, str] = {
    # Gene g16887.t1 from Clonostachys rosea. PlasticDB names the enzyme "PETase",
    # but that is a homology-based gene annotation -- the paper's assay was on PCL
    # (clear zone + weight loss), with no PET assay at all. Placing it in class 2
    # would assert PET-inactivity and placing it in class 1 would assert PET
    # activity; neither is supported, so it is excluded from both.
    "PlasticDB:00230": "named PETase by homology, only ever assayed on PCL",
}


def assign_class(p: "Protein") -> tuple[str, str]:
    """Return (class_label, reason) per decision D1."""
    for sid in p.source_ids:
        if sid in MANUAL_EXCLUSIONS:
            return CLASS_EXCLUDED, MANUAL_EXCLUSIONS[sid]
    group = p.primary_group
    if p.is_pet:
        return CLASS_PET, "PET or PEF substrate recorded"
    if group == "PHA_FAMILY":
        return CLASS_HELDOUT_PHA, "distinct depolymerase lineage, held out as probe"
    if group in CLASS2_GROUPS:
        if not p.is_esterase_fold_candidate:
            return CLASS_EXCLUDED, f"not an esterase fold: {join(p.enzyme_labels)}"
        return CLASS_OTHER_POLYESTER, f"non-PET polyester ({group})"
    return CLASS_EXCLUDED, f"non-ester mechanism ({group})"

# PlasticDB `Enzyme` values that are not alpha/beta-hydrolase esterases and so are
# not fold-matched to the PETases even when the substrate is a polyester
NON_ESTERASE_ENZYMES = {
    "protease",
    "serine protease",
    "chitinase",
    "peg dehydrogenase",
    "laccase",
    "peroxidase",
    "manganese peroxidase",
    "lignin peroxidase",
    "alkane hydroxylase",
    "monooxygenase",
    "dehydrogenase",
    "oxidase",
    "amidase",
    "nitrilase",
    "urease",
}


def norm_seq(raw: str | None) -> str:
    """Uppercase and strip everything that is not a letter (PAZy has stray spaces)."""
    return re.sub(r"[^A-Z]", "", (raw or "").upper())


def split_multi(value: str | None, sep: str) -> list[str]:
    return [v.strip() for v in (value or "").split(sep) if v.strip()]


@dataclass
class Protein:
    """One non-redundant protein, accumulating evidence from every source row."""

    sequence: str
    names: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    source_ids: set[str] = field(default_factory=set)
    substrates: set[str] = field(default_factory=set)
    organisms: set[str] = field(default_factory=set)
    phyla: set[str] = field(default_factory=set)
    tax_ids: set[str] = field(default_factory=set)
    uniprot: set[str] = field(default_factory=set)
    uniref50: set[str] = field(default_factory=set)
    uniref90: set[str] = field(default_factory=set)
    uniref100: set[str] = field(default_factory=set)
    pdb: set[str] = field(default_factory=set)
    genbank: set[str] = field(default_factory=set)
    refseq: set[str] = field(default_factory=set)
    dois: set[str] = field(default_factory=set)
    evidence: set[str] = field(default_factory=set)
    enzyme_labels: set[str] = field(default_factory=set)
    verified_activity: bool = False
    extrapolated_only: bool = True  # PlasticDB: degradation inferred from enzyme alone
    alias_sequences: list[str] = field(default_factory=list)

    def absorb(self, other: "Protein") -> None:
        """Fold a duplicate record into this one, keeping the union of all evidence."""
        for attr in (
            "names", "sources", "source_ids", "substrates", "organisms", "phyla",
            "tax_ids", "uniprot", "uniref50", "uniref90", "uniref100", "pdb",
            "genbank", "refseq", "dois", "evidence", "enzyme_labels",
        ):
            getattr(self, attr).update(getattr(other, attr))
        self.verified_activity = self.verified_activity or other.verified_activity
        self.extrapolated_only = self.extrapolated_only and other.extrapolated_only
        if other.sequence != self.sequence:
            self.alias_sequences.append(other.sequence)

    # -- derived fields -----------------------------------------------------

    @property
    def substrate_groups(self) -> set[str]:
        return {SUBSTRATE_GROUPS.get(s, "UNKNOWN") for s in self.substrates}

    @property
    def mechanisms(self) -> set[str]:
        return {MECHANISM.get(g, "unknown") for g in self.substrate_groups}

    @property
    def is_pet(self) -> bool:
        return bool(self.substrates & PET_SUBSTRATES)

    @property
    def primary_group(self) -> str:
        """One label per protein, most-specific-substrate-first.

        A promiscuous enzyme that degrades both PET and PCL is a PET degrader for
        labelling purposes; the full substrate list is kept so this is recoverable.
        """
        order = [
            "PET_AROMATIC_POLYESTER", "AROMATIC_COPOLYESTER", "ALIPHATIC_POLYESTER",
            "PHA_FAMILY", "POLYCARBONATE", "POLYAMIDE", "POLYURETHANE", "RUBBER",
            "POLYOLEFIN_VINYL", "POLYETHER",
        ]
        groups = self.substrate_groups
        for g in order:
            if g in groups:
                return g
        return "UNKNOWN"

    @property
    def is_esterase_fold_candidate(self) -> bool:
        """Whether the annotated enzyme type is plausibly an alpha/beta hydrolase."""
        return not any(
            e.strip().lower() in NON_ESTERASE_ENZYMES for e in self.enzyme_labels
        )

    @property
    def qc_flags(self) -> list[str]:
        flags = []
        n = len(self.sequence)
        if n < 50:
            flags.append("too_short")
        if n > 1000:
            flags.append("too_long")
        if set(self.sequence) - STANDARD_AA:
            flags.append("nonstandard_aa")
        if not self.sequence:
            flags.append("empty")
        return flags


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_pazy(raw: Path) -> tuple[list[Protein], dict[str, int]]:
    meta_path = raw / "pazy_proteins_metadata.csv"
    seq_path = raw / "pazy_proteins_sequences.csv"

    sequences: dict[str, str] = {}
    with seq_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            sequences[row["protein_id"]] = norm_seq(row["amino_acid_sequence"])

    proteins: list[Protein] = []
    stats = {"rows": 0, "no_sequence": 0}
    with meta_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            stats["rows"] += 1
            pid = row["protein_id"]
            seq = sequences.get(pid, "")
            if not seq:
                stats["no_sequence"] += 1
                continue
            # the export has two near-identical GenBank columns, one with a leading space
            genbank = split_multi(row.get("EMBL-GenBank-DDBJ_CDS"), ";")
            genbank += split_multi(row.get(" EMBL-GenBank-DDBJ_CDS"), ";")
            proteins.append(
                Protein(
                    sequence=seq,
                    names={row["protein_name"].strip()} - {""},
                    sources={"PAZy"},
                    source_ids={f"PAZy:{pid}"},
                    substrates=set(split_multi(row.get("substrates"), "|")),
                    organisms=set(split_multi(row.get("organism_name"), "|")),
                    phyla=set(split_multi(row.get("phylum"), "|")),
                    uniprot=set(split_multi(row.get("UniProtKB"), ";")),
                    uniref50=set(split_multi(row.get("UniRef50"), ";")),
                    uniref90=set(split_multi(row.get("UniRef90"), ";")),
                    uniref100=set(split_multi(row.get("UniRef100"), ";")),
                    pdb=set(split_multi(row.get("PDB"), ";")),
                    genbank=set(genbank),
                    refseq=set(split_multi(row.get("RefSeq_Protein"), ";")),
                    dois=set(split_multi(row.get("literature_dois"), "|")),
                    verified_activity=row.get("verified_activity", "").strip() == "1",
                    extrapolated_only=False,  # PAZy entries are assayed enzymes
                )
            )
    return proteins, stats


def load_plasticdb(raw: Path) -> tuple[list[Protein], dict[str, int]]:
    """PlasticDB is one row per (organism, plastic, publication), so rows are grouped
    by sequence here. Most rows describe whole-organism degradation with no protein
    sequence at all and are skipped."""
    path = raw / "degraders_list.tsv"
    grouped: dict[str, Protein] = {}
    stats = {"rows": 0, "no_sequence": 0}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            stats["rows"] += 1
            seq = norm_seq(row.get("Sequence"))
            if not seq:
                stats["no_sequence"] += 1
                continue
            p = grouped.setdefault(seq, Protein(sequence=seq, sources={"PlasticDB"}))
            eid = (row.get("Enzyme ID") or "").strip()
            if eid:
                p.source_ids.add(f"PlasticDB:{eid}")
            enzyme = (row.get("Enzyme") or "").strip()
            if enzyme and enzyme.lower() != "no":
                p.names.add(enzyme)
                p.enzyme_labels.add(enzyme)
            plastic = (row.get("Plastic") or "").strip()
            if plastic:
                p.substrates.add(plastic)
            for key, target in (
                ("Microorganism", p.organisms),
                ("Tax ID", p.tax_ids),
                ("GenbankID", p.genbank),
                ("DOI", p.dois),
            ):
                v = (row.get(key) or "").strip()
                if v:
                    target.add(v)
            p.evidence.update(split_multi(row.get("Evidence"), ";"))
            # "Yes" means degradation was inferred from the enzyme rather than measured
            if (row.get("Degradation extrapolated from enzyme") or "").strip() == "No":
                p.extrapolated_only = False
    return list(grouped.values()), stats


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def merge_proteins(
    groups: list[list[Protein]], containment_slack: int = 60
) -> tuple[list[Protein], dict[str, int]]:
    """Two-pass dedup: exact sequence, then substring containment.

    `groups` is given in priority order; the first source to contribute a sequence
    owns the canonical record. Containment is only accepted when the two lengths are
    within `containment_slack` residues, which is what a signal peptide or short
    expression tag looks like -- a genuinely different protein that happens to embed
    a shorter one would differ by far more.
    """
    stats = {"exact_dupes": 0, "containment_dupes": 0}

    by_seq: dict[str, Protein] = {}
    for group in groups:
        for p in group:
            if p.sequence in by_seq:
                by_seq[p.sequence].absorb(p)
                stats["exact_dupes"] += 1
            else:
                by_seq[p.sequence] = p

    # longest first so the precursor form becomes canonical and shorter mature forms
    # fold into it
    ordered = sorted(by_seq.values(), key=lambda p: -len(p.sequence))
    kept: list[Protein] = []
    for p in ordered:
        host = next(
            (
                k
                for k in kept
                if abs(len(k.sequence) - len(p.sequence)) <= containment_slack
                and p.sequence in k.sequence
            ),
            None,
        )
        if host is not None:
            host.absorb(p)
            stats["containment_dupes"] += 1
        else:
            kept.append(p)
    return kept, stats


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

COLUMNS = [
    "seq_id", "class_label", "class_reason", "primary_group", "is_pet", "mechanism", "substrates", "substrate_groups",
    "n_substrates", "promiscuous", "protein_names", "enzyme_labels",
    "esterase_fold_candidate", "sources", "source_ids", "verified_activity",
    "extrapolated_only", "evidence", "organism", "phylum", "tax_id", "uniprot",
    "uniref50", "uniref90", "uniref100", "pdb", "n_pdb", "genbank", "refseq",
    "n_literature_dois", "dois", "seq_len", "qc_flags", "n_alias_sequences", "sequence",
]


def join(values) -> str:
    return "|".join(sorted(v for v in values if v))


def to_row(idx: int, p: Protein) -> dict[str, object]:
    return {
        "seq_id": f"BPS{idx:04d}",
        "class_label": assign_class(p)[0],
        "class_reason": assign_class(p)[1],
        "primary_group": p.primary_group,
        "is_pet": int(p.is_pet),
        "mechanism": join(p.mechanisms),
        "substrates": join(p.substrates),
        "substrate_groups": join(p.substrate_groups),
        "n_substrates": len(p.substrates),
        "promiscuous": int(len(p.substrates) > 1),
        "protein_names": join(p.names),
        "enzyme_labels": join(p.enzyme_labels),
        "esterase_fold_candidate": int(p.is_esterase_fold_candidate),
        "sources": join(p.sources),
        "source_ids": join(p.source_ids),
        "verified_activity": int(p.verified_activity),
        "extrapolated_only": int(p.extrapolated_only),
        "evidence": join(p.evidence),
        "organism": join(p.organisms),
        "phylum": join(p.phyla),
        "tax_id": join(p.tax_ids),
        "uniprot": join(p.uniprot),
        "uniref50": join(p.uniref50),
        "uniref90": join(p.uniref90),
        "uniref100": join(p.uniref100),
        "pdb": join(p.pdb),
        "n_pdb": len(p.pdb),
        "genbank": join(p.genbank),
        "refseq": join(p.refseq),
        "n_literature_dois": len(p.dois),
        "dois": join(p.dois),
        "seq_len": len(p.sequence),
        "qc_flags": join(p.qc_flags),
        "n_alias_sequences": len(p.alias_sequences),
        "sequence": p.sequence,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--containment-slack", type=int, default=60)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    pazy, pazy_stats = load_pazy(args.raw)
    pdb_recs, pdb_stats = load_plasticdb(args.raw)

    # PAZy first: every entry is curator-verified and carries UniRef/PDB/UniProt
    # cross-references that PlasticDB lacks, so it should own the canonical record.
    merged, dedup_stats = merge_proteins([pazy, pdb_recs], args.containment_slack)
    merged.sort(key=lambda p: (p.primary_group, -len(p.sequence)))

    rows = [to_row(i + 1, p) for i, p in enumerate(merged)]

    tsv_path = args.out / "plastizymes_merged.tsv"
    with tsv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    fasta_path = args.out / "plastizymes_merged.fasta"
    with fasta_path.open("w") as fh:
        for r in rows:
            fh.write(f">{r['seq_id']} {r['primary_group']} {r['substrates']}\n")
            seq = str(r["sequence"])
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")

    report = {
        "pazy": pazy_stats | {"usable_entries": len(pazy)},
        "plasticdb": pdb_stats | {"usable_unique_sequences": len(pdb_recs)},
        "dedup": dedup_stats,
        "merged_total": len(merged),
        "by_primary_group": {
            g: sum(1 for p in merged if p.primary_group == g)
            for g in sorted({p.primary_group for p in merged})
        },
        "by_class_label": {
            c: sum(1 for p in merged if assign_class(p)[0] == c)
            for c in sorted({assign_class(p)[0] for p in merged})
        },
        "pet_positives": sum(1 for p in merged if p.is_pet),
        "qc_flagged": sum(1 for p in merged if p.qc_flags),
    }
    (args.out / "merge_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nwrote {tsv_path} ({len(rows)} rows)")
    print(f"wrote {fasta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
