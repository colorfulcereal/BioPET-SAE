# CLAUDE.md — BioPET-SAE

## Core thesis (don't drift from this)
The project's contribution is: can **PET-degrading enzyme (plastizyme) activity be
detected from pLM embeddings**, and can the resulting decision be **mechanistically
explained** via sparse autoencoder (SAE) features of the underlying protein language
model? Two halves, in order:
1. **Detection** — embeddings (ESM-2) + classifiers (linear / MLP probes) that
   separate PETases / plastic-degrading hydrolases from close non-degrading
   relatives (esterases, cutinases, lipases, other EC 3.1 hydrolases).
2. **Explainability** — which SAE features the classifier's decision actually rests
   on, i.e. what sequence/structural signal the model is using to call something a
   plastizyme.

The hard part is the **negative set**: a classifier that only distinguishes PETases
from random proteins has learned "is this a hydrolase," not "is this a plastizyme."
Frame infrastructure work (dataset assembly, embedding pipeline, SAE reconstruction
fidelity, probe plumbing) as *in service of* those two claims, never as a finding in
itself.

Sibling project with shared conventions and tooling:
`colorfulcereal/PLMCircuitInterp` (SAE circuits → EC 2.7 kinase function). Check its
`PROGRESS_LOG.md` (that project's own filename) before re-deriving pipeline details —
much of the ESM-2 / InterPLM plumbing is already solved there.

## Required after any meaningful unit of work
- **Update `PROGRESS.md`** at the project root (create it if it doesn't exist
  yet): what was done, why, current status, and what's next. This is the source of
  truth for project state — check it before re-deriving something from first
  principles.
- **Update persistent memory** (`~/.claude/projects/.../memory/`) for state changes
  that matter across sessions: new pipeline component status, environment/library
  gotchas discovered, and any explicit course-corrections from the user.

## Git workflow
- **Never run `git commit` or `git push` without the user's explicit go-ahead for
  that specific commit/push.** When changes are ready to commit, draft the commit
  message and show it to the user — they will run the commit and push themselves.

## Verification standard
Never trust a pipeline's or tool's own reported statistics at face value.
Independently re-derive important claims — re-query the live source (e.g. UniProt,
PAZy) for a random sample, or recompute a metric via an independent method — before
treating output as ground truth. When re-deriving, match the *actual* algorithm the
original tool uses (e.g. global vs. local sequence alignment) rather than assuming a
plausible-sounding default; a mismatched method can produce a false alarm that looks
like a real bug.

Dataset-specific: known plastizymes are a **small, homology-dense set** (PETase,
MHETase, LCC, cutinases, and engineered variants). Any headline accuracy number is
meaningless without a homology-aware split — cluster/partition by sequence identity
(SpanSeq or equivalent) so train and test don't share near-duplicates, and report the
class balance alongside the metric.

## Planning
For any implementation task with real design decisions (new script involving
external tools/APIs, ambiguous methodology, multiple valid approaches), research
first (read source, test APIs directly) and present a concrete plan before writing
code — don't skip straight to implementation on nontrivial asks.

## Scripting conventions
- Use `uv` for all dependency management (`uv add`, `uv run`) — not raw pip/conda.
- To depend on a package only available via git, use
  `uv add "<pkg> @ git+<url>"` rather than manually cloning into the project tree.
- Layout: `src/<package>/` with `uv_build`, data under `data/`, all run outputs
  (metrics, weights, scalers, JSON dumps) under `results/<experiment_name>/`.
- Scripts consuming file paths that aren't fixed yet (e.g. outputs from an external
  tool run elsewhere) take those paths as CLI arguments with sensible defaults, not
  hardcoded constants.
- Reuse existing functions across scripts when logic overlaps (e.g. UniProt
  fetch/clean helpers, FASTA parsing, label loading) rather than duplicating.
- When comparing two model architectures head-to-head (e.g. linear vs. MLP probe),
  train them with an *identical* procedure (same loss/optimizer/epochs/seed) so any
  gap reflects architecture, not training differences.
- Persist fitted preprocessing transforms (e.g. a `StandardScaler`) alongside model
  weights — later evaluation must apply the exact same transform, never a refit one.
- Record the pooling strategy (mean vs. max over residues) in the output directory
  name; it materially changes probe performance and must stay comparable.
- Prototype expensive/slow steps on a small subset first before running full-scale.

## Known environment gotchas
- Local machine is Apple Silicon (macOS arm64). Some bioinformatics tools (e.g.
  CCPhylo, a SpanSeq dependency) have no `osx-arm64` build — prefer running such
  steps on Lightning AI (Linux, has GPU) over local Docker/Rosetta workarounds.
- The GPU only matters for ESM-2 embedding extraction; SpanSeq's tools
  (KMA/Mash/CCPhylo/GGsearch36/CD-HIT) are CPU-bound.
- UniProt's `ec:X.Y.*` query is a loose string-prefix match, not an exact subclass
  match (e.g. `ec:3.1.1.*` also matches 3.1.10–3.1.14). Use `ec:X.Y.-` for an exact
  subclass match instead.
- `pip install -e .` can fail on packages whose `setup.py` imports project code at
  the top level (e.g. SpanSeq) due to pip's default build isolation — use
  `--no-build-isolation` when that happens.
- ESM-2 embeddings must be extracted via HuggingFace `transformers`
  (`EsmModel`/`AutoTokenizer`), not `fair-esm` — InterPLM's own embedder uses
  HuggingFace, and the pretrained SAEs are only guaranteed compatible with
  activations produced the same way.
- `interplm`'s installed package is missing its `train` subpackage (upstream
  packaging bug) — `load_sae_from_hf`/`load_sae` cannot work. Load SAEs via
  `interplm.sae.dictionary.ReLUSAE.from_pretrained(path)` directly after downloading
  the weight file with `huggingface_hub.hf_hub_download`.
