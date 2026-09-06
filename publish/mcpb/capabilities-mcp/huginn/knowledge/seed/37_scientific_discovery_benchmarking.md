# Scientific Discovery Benchmarking — Hard-Won Lessons

Source: RCBench cross-domain evaluation (Astronomy, Math, Physics, Material, Chemistry).
Each lesson cost at least one failed run. Don't relearn them the hard way.

## 1. Phased Protocol — the anti-over-engineering discipline

Agent repeatedly fails by going deep on one model (VAE, transformer) and running
out of tool budget before writing any report. The fix is a hard phase schedule:

- Phase 1 (calls 1-15): EDA, read instructions. NO modeling.
- Phase 2 (calls 16-30): ONE simple model (GPR / Ridge / RF / OLS). 2-3 figures.
- Phase 3 (calls 31-40): WRITE report.md NOW. Incomplete results are fine.
- Phase 4 (calls 41+): Iterate — add models, update report.
- Phase 5: Verify report references all figures.

Deep learning (VAE/transformer/GNN) is ADVISORY-DELAYED until report.md exists —
UNLESS the task explicitly requires reproducing a DL architecture (e.g. "reproduce
this VAE", "implement CGCNN"). For DL-required tasks, proceed with DL immediately
but write report.md EARLY (after first DL attempt, even if broken) and OVERWRITE
as you iterate. The phase discipline is "report first" for open-ended discovery;
for paper-reproduction tasks it is "DL early, report alongside".
A short report with correct simple analysis beats a long report with broken complex ML.

## 2. Path discipline

Agent confuses `/data/x.csv` (Unix absolute, doesn't exist on Windows) with
`data/x.csv` (relative to workspace). The absolute path silently fails and agent
gives up instead of trying alternatives.

Rule: ALWAYS use relative paths. If a path fails, glob for the filename first.
NEVER stop on a single failed tool call.

## 3. Noise as feature — scientific epistemology

Boundary conditions, edge cases, and noise are NOT bugs — they are intrinsic
features of how nature runs. Three sources to distinguish:

1. **Observation/measurement error** → suppress via Kalman/Bayes filter.
2. **Parametric uncertainty** → propagate via GP posterior or polynomial chaos.
3. **Intrinsic stochasticity of the physical process** → MODEL IT, do not average
   it out. It carries mechanism information:
   - Thermal fluctuations → temperature
   - Shot noise → quantization
   - 1/f noise → self-organized criticality

When residuals show structure (autocorrelation, heteroscedasticity, heavy tails),
this is the model telling you what physics it's missing. Do not just report R².
Diagnose: which parameter dominates residual variance? Which mechanism is uncertain?

A clean R² with unexamined residuals is curve-fitting, not science.

### Itô/Stratonovich insight

If noise comes from system parameters themselves, the random diffusion term
often INHERITS the structure of the deterministic dynamics. The noise covariance
is shaped by the drift field. This is why multiplicative noise (state-dependent
diffusion) appears in so many physical systems — it's not arbitrary, it reflects
the geometry of the underlying dynamics.

## 4. Scoring — visual evaluation without a vision model

Text-only LLM judges (deepseek-chat) cannot see images. Three-tier fallback:

1. **Vision judge** (qwen2.5-vl / gpt-4o): send image_paths directly. Best accuracy.
2. **CV pre-analysis** (numpy/skimage, ~50ms/image): image type guess + statistics
   + edge density. Inject as text into judge prompt. No vision model needed.
3. **Text-only**: judge based on report text description of figures. Worst accuracy.

The `build_cv_context()` function in `huginn.vision.router` does tier 2 automatically.
It gives the text judge "eyes" — enough to distinguish a scatter plot from a bar
chart, a smooth curve from a noisy one, a microscopy image from a schematic.

## 5. Cross-domain patterns (from 5 domains × RCBench)

| Domain | What works | What fails |
|--------|-----------|------------|
| Astronomy | GP regression, light curve fitting | Over-fitting periodograms |
| Math | Symbolic derivation, convergence plots | Numerical precision issues |
| Physics | GPR on composition-property, phase diagrams | Missing torch_geometric for GNN |
| Material | RDKit descriptors, classical ML | VAE training eats all tool budget |
| Chemistry | sklearn on MoleculeNet, ECFP | KA-GNN needs torch_geometric |

Universal pattern: classical methods (GPR, Ridge, RF, OLS) get 60-80% of the
score. Deep learning adds marginal improvement but costs 3-5x tool calls.
Default to classical, only escalate to DL if report.md is already written AND
the checklist explicitly requires it.

## 6. Tool budget allocation (100 calls total)

- Data exploration: 10-15 calls
- Simple modeling + figures: 15-20 calls
- Report writing: 5-10 calls (DO THIS EARLY)
- Iteration + advanced models: 30-40 calls
- Report polishing: 10-15 calls

If you hit call 70 with no report.md, STOP all analysis and write report.md.
The deliverable is report.md, not a perfect model.

## 7. Common failure modes

1. **VAE/transformer rabbit hole**: agent trains a VAE for 40+ calls, runs out
   of budget, no report. Fix: phased protocol + DL ceiling.
2. **Path confusion**: `/data/` vs `data/`. Fix: relative path discipline.
3. **Circuit breaker lockout**: code_tool fails 5x → locked 60s → agent gives
   up. Fix: `HUGINN_HEALTH_MONITOR=0` in autonomous settings.
4. **RestrictedPython blocking science**: os/pathlib/pickle/eval all banned.
   Fix: monkey-patch `validate_code` for isolated workspaces.
5. **Single error → agent stops**: one failed tool call ends the run.
   Fix: "On error: fix and continue. NEVER stop on a single failed tool call."
