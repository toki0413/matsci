# Nature-Style Peer Review & Author Response — Distilled Taxonomy

Distilled from `mumdark/nature-review-studio` v1.4.1 (1287 Nature Peer Review
Files, 2025–2026). Use this whenever the agent is asked to (a) generate
Nature-style reviewer reports, (b) draft a point-by-point author response,
or (c) track revision tasks. The taxonomy is a closed vocabulary learned from
real PRF corpora — picking labels from outside the vocabulary breaks the
output contract.

Source: https://github.com/mumdark/nature-review-studio (MIT-style,
distillation artefacts released under repository LICENSE).

---

## 1. The 12 concern axes (what reviewers ask)

Default severity is learned from frequency in the 1287 PRF corpus: ≥5%
of cases with this axis → `major`; 1–5% → `minor-major`; <1% → `minor`.
The skill may upgrade/downgrade based on manuscript type.

| Axis | Default severity | 1287 PRF hits | What to look for |
|---|---|---|---|
| `experimental-design` | major | 6776 | Controls, replicates, conditions, dose-response, batch effects |
| `figures-and-tables` | minor | 6379 | Axis labels, statistical overlays, color-blind safety, scale bars |
| `claim-moderation` | minor-major | 3824 | Overclaim beyond evidence, hedging needed |
| `novelty-significance` | major | 3185 | Mechanism novelty, advance over prior art, cross-discipline interest |
| `reproducibility` | major | 2751 | Code/data availability, environment pins, random seeds, ablation |
| `writing-clarity` | minor | 2513 | Redundancy, jargon, abstract-vs-body consistency |
| `data-resource-quality` | minor-major | 1153 | Completeness, documentation, future usability |
| `clinical-validity` | major | 1149 | Cohort selection, external validation, sensitivity/specificity |
| `mechanism-evidence` | major | 875 | Causal claims, rescue experiments, orthogonal evidence |
| `mechanistic-vs-correlative` | major | 516 | Whether correlative claims are reified into causal ones |
| `statistical-rigor` | major | 465 | Power, multiple testing, estimator assumptions, pre-registration |
| `ethical-governance` | major | 83 | IRB, consent, animal welfare, dual-use, data sharing |

### Example axis assignment

> Reviewer reads: "X gene knockout completely regressed tumors."

| Real reviewer question | Axis | Default |
|---|---|---|
| "X was already reported in Nature 2022" | `novelty-significance` | major |
| "How do you rule out downstream Y? Add rescue" | `mechanism-evidence` | major |
| "Only 1 cell line; need ≥3" | `experimental-design` | major |
| "n=6 isn't power; do power analysis" | `statistical-rigor` | major |
| "Upload analysis code" | `reproducibility` | major |
| "Validate in patient samples" | `clinical-validity` | major |
| "Missing IRB number" | `ethical-governance` | major |
| "DB annotation incomplete" | `data-resource-quality` | minor-major |
| "Figure 3 y-axis should be 0–100%" | `figures-and-tables` | minor |
| "Intro too verbose" | `writing-clarity` | minor |
| "Conclusion overclaims 'complete regression'" | `claim-moderation` | minor-major |
| "Correlation ≠ causation, hedge" | `mechanistic-vs-correlative` | major |

---

## 2. Manuscript fingerprint → reviewer set (6 archetypes)

| Manuscript fingerprint | # Reviewers | Roles |
|---|---|---|
| Pure wet-lab mechanism | 3 | mechanism / experimental-design / figures |
| Cohort + ML clinical | 5 | clinical-validity / ml-methods / statistics / ethics / figures |
| Observational + theory | 3 | mechanism / statistical-rigor / writing |
| Resource (large data + tooling) | 4 | data-resource-quality / reproducibility / experimental-design / figures |
| Review / Perspective | 2 | novelty-significance / writing-clarity |
| Mixed multi-method | 4–5 | As needed, cover each method family once |

### Method family frequency (1287 PRF)

| Method family | Cases hit |
|---|---|
| review-theory | 1177 |
| data-resource | 778 |
| ML | 588 |
| wet-lab | 492 |
| omics | 457 |
| imaging | 424 |
| clinical | 413 |
| simulation | 146 |
| unspecified | 20 |

Rule of thumb: a paper spanning ≥4 method families gets 4–5 reviewers
(one per family); a Review article gets 2 (writing + novelty).

---

## 3. The 21 response strategies (author action dictionary)

Each reviewer comment maps to **exactly one** strategy. Closed vocabulary;
new strategies only via manual review of the corpus.

### A. Accept / supplement (concede + add)

| Strategy | When |
|---|---|
| `acknowledge_and_correct` | Reviewer caught a real error |
| `clarify_existing_content` | Reviewer misread; point to existing text |
| `add_textual_explanation` | Belongs in manuscript, not in letter |
| `add_reference` | A missing key citation |
| `add_method_detail` | Methods section too sparse |
| `add_statistical_analysis` | Re-run a stat on existing data |
| `add_robustness_analysis` | Sensitivity / bootstrap / alternative model |
| `add_control` | A control experiment addresses the concern |
| `add_experiment` | New wet-lab / clinical experiment is feasible |
| `add_validation_dataset` | External cohort / dataset |
| `provide_data_or_code` | Upload anonymized data or code |

### B. Limit / adjust (lower intensity, don't add)

| Strategy | When |
|---|---|
| `moderate_claim` | Add hedging ("suggest" → "indicate") |
| `change_terminology` | Replace over-strong wording |
| `restructure_figure` | Re-arrange panels / labels |
| `move_content_to_supplement` | Demote content for readability |
| `withdraw_claim` | Claim unsupportable; remove it |

### C. Refuse / delegate (disagree or escalate)

| Strategy | When |
|---|---|
| `explain_infeasibility` | Genuinely impossible in this revision |
| `respectfully_disagree` | Reviewer is wrong, with evidence |
| `request_editor_adjudication` | Reviewer conflict / scope issue |
| `defer_to_future_work` | Real but out of scope |

---

## 4. Action-status taxonomy (8 states)

The revision task table tracks each concern through these states.

| Status | Meaning | Trigger |
|---|---|---|
| `DONE` | Verified in revised MS | User supplies revised MS, text matches |
| `DRAFTED` | Reply written but not in MS | User says "done" without MS |
| `TODO_TEXT` | Needs text edit | Strategy is `add_textual_explanation` etc |
| `TODO_ANALYSIS` | Needs computational re-run | Strategy is `add_statistical_analysis` etc |
| `TODO_EXPERIMENT` | Needs wet-lab work | Strategy is `add_experiment` etc |
| `TODO_AUTHOR_CONFIRM` | Recommended but uncertain | Agent is not sure |
| `NOT_FEASIBLE` | Cannot be done | Strategy is `explain_infeasibility` |
| `PROPOSED_DISAGREEMENT` | Suggest respectful pushback | Strategy is `respectfully_disagree` |

Hard rule: never label `DONE` unless the user supplied a revised manuscript
or explicitly listed completed work. "We have already done X" without MS →
`DRAFTED` + quiet inline note.

---

## 5. 10 failure modes (internal silent check)

Detected silently; may downgrade `DRAFTED` → `TODO_AUTHOR_CONFIRM`. Never
surfaced as a critic-flag block in the docx.

1. **Empty agree** — "we agree with the reviewer" with no action
2. **Decorative thanks** — thank-you with no concession
3. **Status inflation** — claims `DONE` without supporting material
4. **Repeat-the-manuscript** — quotes MS without addressing the concern
5. **Oversold experiment** — claims new experiment but no revised-MS section
6. **Hedge-without-action** — "interesting point" with no follow-up
7. **Over-defensiveness** — disputes concern using only rhetoric
8. **Out-of-scope expansion** — adds a major new study unrelated to concern
9. **Lost sub-question** — multi-part concern, only one part answered
10. **Status-symbol citation** — adds a citation that doesn't support the claim

---

## 6. Output contract (two-file deliverable)

Every `review` or `respond` invocation produces **exactly two files**:
1. `<case>_<YYYYMMDD>.docx` — editable Word
2. `<case>_<YYYYMMDD>.md` — clean Markdown mirror, same stem

Both share substantive content. No additional reports, no sidecar JSON, no
compliance notes in the file.

### Revision-level banner (top of editor letter, mandatory)

Pick exactly one:

| Banner | Trigger |
|---|---|
| `Major revision` | ≥1 Major concern open that text alone cannot close |
| `Accept with minor revisions` | All Major concerns closable by text or minor analyses |
| `Reject in present form` | Core conclusion unsupported; resubmission = different study |
| `Cannot be assessed` | MS or required files missing / unreadable |

### Per-reviewer section shape (mandatory)

```
### Reviewer X — <Role label>
**Overall.**  <60–120 word paragraph: significance + headline concerns>

**Major concerns**
1. <short noun-phrase heading> — <one body paragraph, evidence pointer inline>
2. ...

**Minor concerns**
1. <short noun-phrase heading> — <one body paragraph>
2. ...

**Confidence:** high | medium | low
```

Role labels (no names, no bios):
`Mechanism Reviewer` / `Statistical Reviewer` / `Clinical Validity Reviewer`
/ `Reproducibility Reviewer` / `Figures & Tables Reviewer` /
`Writing & Clarity Reviewer` / `ML Methods Reviewer` / `Ethics Reviewer`.

### Cross-review consensus (single block)

Keep only ONE cross-review block. Format:
```
Cc.N — <short description>
   raised by: Reviewer X, Y
   axis: <axis>
   severity: major
   <1–3 sentence rationale>
```

Deliberately NOT emitted (internal-only):
- Divergence block (single-reviewer concerns)
- Severity escalation/demotion rules
- Decision-prediction hints (`likely-major-revision` band, Nature-fit verdict)
- Claim-evidence map dump

### Revision task table (mandatory, last block)

| ID | Reviewer (审稿人) | Concern (问题) | Strategy (策略) | Status (状态) | Input needed (所需输入) | Output (预期输出) | Blocks response? (是否阻塞) |

`Blocks response?` is `Yes` for tasks the author must complete before
credible resubmission, otherwise `No`.

---

## 7. Anti-patterns (never appear in the docx)

- "We have fully addressed all concerns" — banned (status inflation)
- Decision predictions ("the editor will likely accept…") — banned
- Decorative thanks opening ("First, we would like to thank…") — banned;
  first sentence must summarize the revision band, not thank
- Preamble / compliance note / coverage note — banned
- Adversarial self-check / critic-flag block — internal only
- Severity escalation / divergence block — internal only

---

## 8. How huginn should use this taxonomy

### Trigger pattern (auto-suggest this seed when)

- User mentions "review", "审稿", "Nature-style review", "multi-perspective review"
- User mentions "respond", "rebuttal", "回复审稿人", "point-by-point"
- User supplies a manuscript draft + asks for "what would reviewers say"
- User supplies reviewer comments + asks for help drafting a response

### Huginn workflow

1. Classify the manuscript fingerprint (count method families from abstract
   + methods; map to one of the 6 archetypes in §2).
2. Select reviewer set per §2 mapping.
3. For each reviewer, scan the manuscript against the axes in §1 that match
   that role; emit concerns with axis + severity + claim/evidence pointers.
4. For each concern, pick one strategy from §3 (closed vocabulary); never
   invent a 22nd strategy.
5. Build the revision task table per §6 with the 8 status values from §4;
   never label `DONE` without a revised MS.
6. Run the 10 failure-mode checks from §5 silently; downgrade statuses as
   needed; never emit the checks themselves.
7. Render the two-file deliverable per §6 contract; no extras.

### Adaptation notes for materials science manuscripts

The corpus is bio/clinical-heavy (note `clinical-validity` and
`ethical-governance` axes). For pure materials papers:

- `clinical-validity` rarely fires; demote to `experimental-design` +
  `mechanism-evidence` instead.
- Add materials-specific sub-axes mentally (e.g. `reproducibility` should
  check POSCAR/CIF, INCAR tags, k-mesh, pseudopotential version — not just
  code/data availability).
- `mechanistic-vs-correlative` maps cleanly to "property predicted by
  descriptor vs measured directly" — a common ML-in-materials pitfall.
- `statistical-rigor` should check train/test split leakage, descriptor
  redundancy, error bars on property predictions.
- `data-resource-quality` should check FAIR compliance of CIF/POSCAR
  uploads, lattice-parameter precision, composition provenance.

### Cross-references

- **Seed 41** (OneScience) — research workflow framing this fits into
- **Seed 38** (benchmark evaluation lessons) — `reproducibility` axis +
  `DRAFTED`-without-MS failure mode parallel benchmark reproducibility
  gates already in huginn

## Sources

- Repository: https://github.com/mumdark/nature-review-studio (v1.4.1)
- README: https://github.com/mumdark/nature-review-studio/blob/main/README.md
- `review-axes.md`: https://github.com/mumdark/nature-review-studio/blob/main/skill/references/review-axes.md
- `response-axes.md`: https://github.com/mumdark/nature-review-studio/blob/main/skill/references/response-axes.md
- Corpus: 1287 Nature Peer Review Files (2025–2026); see `knowledge/index_axes.json`, `knowledge/index_methods.json`, `knowledge/index_severity.json`
