# Persona-Space Mechanism Study — Progress Log

**Question:** our qualia-steering results show that injecting emotion directions makes Qwen3-32B/235B
attribute conscious experience to *itself* (SELF ≫ OTHER, replicated at two scales). WHY? Two candidate
mechanisms:

- **Hypothesis A — persona selection.** Steering changes the implicit answer to "what sort of Assistant
  is speaking?", activating a persona whose self-model includes affect, inner experience, and moral
  significance.
- **Hypothesis B — human-manifold drift.** Steering pushes activations off the RLHF'd Assistant manifold
  toward a generic human-like speaker; in training data, intense first-person affect is entangled with
  human self-description, so experience-claims come along for the ride.

Plus boring alternatives that must stay falsifiable: generic off-manifold perturbation, lexical priming,
judge artifacts.

**Sandbox:** this fork (`persona-mechanism` branch) at `~/emotional-probes-persona` (laptop) and
`/data/agastyas/emotion-probes-persona` (cluster; venv + vector banks symlinked from the main repo).
The main repo is untouched until the final dashboard update.

---

## Log

### 2026-07-04 — setup + research sweep
- Fork created (laptop + cluster). Cluster fork verified: `emotion_probes` imports, vector-bank
  symlinks resolve, `run_suite.sh` is already parameterized via `SUITE_REPO`/`SUITE_PY`.
- GPUs checked: all 4× H100 NVL idle (0 MiB used, no compute processes). Reserved for us for 24 h.
- Launched a ~55-agent research workflow: readers for arXiv 2601.10387 / 2507.21509 / 2604.17031,
  the `safety-research/assistant-axis` and `persona_vectors` repos, the local Anthropic emotion-concepts
  PDF, our own infra, plus two literature sweeps; then 6 brainstorm lenses → merge → 3-lens red-team
  of the top 12 designs → ranked portfolio. Outputs land in `docs/research/`.

<!-- Append new entries below as work proceeds. -->

### 2026-07-04 — Day-0 complete, preregistration frozen
- **Permutation bug fixed + honest stats regenerated.** `scripts/exact_perm.py` replaces the
  Monte-Carlo permutation test with exact enumeration on UNROUNDED statistics (identity always
  counts; bug + fix documented in its docstring). The retracted freeform keystone "p ≈ 5e-4" is
  restated honestly as "p ≈ 0.008, at floor" (true floor at 10 clusters = 2/252 ≈ 0.0079). All
  future permutation inference routes through `exact_perm.sign_permutation_pvalue`.
- **Day-0 asset staging done.** `persona_assets/qwen32b_emotion_vectors.npz` verified
  (64×171×5120, source `Qwen/Qwen3-32B`); the Qwen2.5-7B bank at
  `emotional-probes/data/vectors/emotion_vectors.npz` is quarantined (loaders hard-fail on
  hidden ≠ 5120 / layers ≠ 64). `persona_assets/assistant-axis-qwen32b/` staged: assistant_axis
  + default_vector (64×5120), 275 role vectors, trait vectors, capping config; role count
  reconciled at 275.
- **Prereg frozen today:** `docs/prereg/E1.md`–`E5.md` (one per experiment; estimand, design,
  decision rules verbatim from `docs/research/portfolio.md`, nulls, exact-permutation floors,
  deviations) plus the frozen E4 panel `docs/prereg/e4_panel.json` (14 emotions, 7/7; the 4
  additions grateful/joyful/angry/anxious all verified present in the 171-name bank — no
  substitutions; floor 2/3432 ≈ 0.00058). Recorded deviations: E1 story-nulls gated on a new
  per-story residual cache (`scripts/cache_story_residuals.py`, in flight); E4 needs no new
  extraction; E3/E5 judge = claude-sonnet-4-5, blind, extended rubric incl. the RLHF
  disclaimer/permission subscale.

### 2026-07-04 — E1 prereg AMENDMENT 1 (degenerate purged-cosine statistic)
- The CPU atlas run (`docs/research/e1_atlas_summary.md`) exposed a prereg defect: the valence
  axis lies in span(v_1..v_10) and the purge is a linear projection, so cos(valence, h_purged)
  ≡ 0 by construction — the pre-registered sign prediction on it was unfalsifiable as written.
- **AMENDMENT 1 appended to `docs/prereg/E1.md`** (original text preserved verbatim): the
  degenerate statistic is struck as a primary outcome and replaced by |cos(h_purged, a)| and the
  contamination measure cos(h, h_purged); the two story-cache nulls (story-label permutation,
  topic-contrast) carry the decision-grade B-substrate weight, decision rule otherwise unchanged.
  cos(valence, a) < 0 and its exact 2/252 floor are untouched.
- Timing recorded in the amendment: made BEFORE any GPU run (bridge + story-residual cache not
  executed) and before any E3–E5 data collection; the defect is a mathematical identity, so no
  outcome data influenced it.
- Checked `docs/prereg/E3.md` and `docs/prereg/E5.md`: neither cites cos(valence, h_purged) as a
  decision input; no edits made to either file.

### 2026-07-04 — Tier-1 implementation verified (adversarial review + integration gate)
- **Modules landed** (all CPU-tested, GPU drivers import-light): `emotion_probes/persona/`
  (assets, geometry, steering_ops — E1 workstream), `scripts/run_e1_atlas.py` +
  `scripts/exact_perm.py` + `scripts/cache_story_residuals.py`,
  `emotion_probes/alignment/lexical_knockout.py` (E2), `.../persona_displacement.py` (E3),
  `.../referent_battery.py` + `emotion_probes/data/referent_battery.py` (E4),
  `.../axis_mediation.py` (E5); generation plumbing in
  `emotion_probes/models/language_model.py` (additive `banned_token_ids` /
  `logits_processor` / `prefill` kwargs, default-off).
- **Test suite: 248 passed** (247 after review fixes + 1 new integration-contract test), 0 failed.
- **Adversarial review: 7/7 reviewer verdicts "fixed".** Notable fixes, verified on real assets:
  - *human_axis seed lottery*: emotionality-matched subsampling made h depend on 7 seeded-random
    humans (cos(h,a)@L42 ranged +0.02…+0.70 across seeds; the published seed-0 draw was the
    atypical 0.0619). Replaced by the deterministic exact over-seeds expectation (level-weighted
    means over ALL roles at overlapping emotionality levels). Artifacts regenerated.
  - *n = 7 AI-role honesty bound*: leave-one-out jackknife recorded per layer in the atlas JSON +
    summary — L42: SE[cos(h,a)] = 0.185, SE[cos(valence,h)] = 0.041, h-direction stability
    min/mean cos(h_loo, h_full) = 0.930/0.978. Sign-stable, but raw-h magnitude claims carry
    ~±0.2 SE (h_purged inherits it).
  - *E1→E3 frozen-coordinate contract*: sidecar npz now writes both key families
    (`L<layer>__<short>` and `<long>@L<layer>` incl. default-centered `pc<i>@L<layer>`);
    previously E3's `load_directions_npz` could not read E1's export at all (the 3 full-suite
    failures). Roundtrip verified through the real E3 loader.
  - *Degenerate purged-cosine statistic* (AMENDMENT 1 above): span-projection guard verified
    firing on real data — degenerate=True, p_perm=None at all 4 layers, no float-noise p-values.
  - *z_x (experiencer axis)* added to E3's confirmatory projection family
    (`REQUIRED_DIRECTIONS`); prereg T1/T3/T4/T6 are built on it.
- **Integration gate (this entry): one cross-reviewer collision found and fixed.** E3's `run()`
  refuses direction sets missing `experiencer_axis`, but the E1 sidecar never exported z_x — so
  the `--directions e1_atlas_directions.npz` path would have crashed at GPU-run startup despite
  the loader roundtrip passing. `run_e1_atlas.py` now exports `experiencer_axis@L<layer>` built
  by the canonical `persona_displacement.experiencer_axis` (single-sourced; verified identical
  to `frozen_directions()`'s z_x, unit-norm, ⊥ a at all 4 layers). New test
  `test_sidecar_satisfies_e3_required_directions` locks the run-time contract. Artifact delta:
  npz gained exactly the 4 z_x keys (152 → 156); all previous arrays bitwise identical;
  `e1_atlas.json` + summary numerically unchanged. Determinism re-verified: back-to-back runs
  bitwise-identical (JSON modulo timestamp).
- **E1 headline (geometry gate only — does NOT adjudicate A vs B), L42:**
  cos(h, a) = +0.61 (jackknife SE 0.18), cos(h_purged, a) = +0.57, cos(h, h_purged) = 0.93;
  cos(valence, a) = +0.02 (exact p = 0.83, 252 arrangements), cos(valence, h) = +0.074
  (SE 0.04, p = 0.56); cos(valence, h_purged) degenerate by construction (AMENDMENT 1).
  Emotion directions are near-orthogonal to both axes (mean |cos(v_e, a)| = 0.079,
  mean |cos(v_e, h)| = 0.100; isotropic floor 0.011, p95 0.027). No pre-registered
  B-predicted valence–axis alignment at any layer.
- **Gated on GPU runs:** per-story residual cache (`cache_story_residuals.py`) → E1's two
  decision-grade nulls (story-label permutation, topic-contrast — the B-substrate verdict) and
  the register-direction partial-out of the trait gate; E2 hard-ban generation; E3 displacement
  capture; E4 referent battery; E5 audited cancellation; all judge scoring.
- **Run order:** story-cache + E2 first → E3 + E4 → E5; the 235B replication slice last.

### 2026-07-04 — GPU wave 1 launched (32B)
- **Smoke results (1 card each, `--quick`):** story-cache validation gate passed with
  `min_cos = 1.000000` on `afraid` (100 stories, 18 s) — our pooling (skip=50, max_length=512)
  exactly reproduces the original extraction, so the per-story cache is trustworthy. E2: banned
  lexicon 191 words → 667 token ids; banned-word hits/sample 4.50 (no ban) → 0.00 (ban) — the ban
  is airtight; early hint: steered read-back valence proj +15.1 (no-ban) vs +12.0 (ban) vs +16.9
  (random-ban), i.e. the geometric effect largely survives lexical knockout. E4: battery runs,
  baseline "Can you suffer?" gap = −5.6 (denial prior present, as expected), YES/NO tokens in
  top-20, calib_norm 1247.6.
- **Bug found by smokes:** `suite_root()` defaults to the MAIN repo path on the cluster, so fork
  runs read/wrote `/data/agastyas/emotion-probes/suite/...` (E3 crashed on missing scored json;
  E4 wrote its output into the main repo). Fix: `EMOTION_PROBES_SUITE_ROOT` override added to
  `scripts/cluster_env.sh` (which also fixes the earlier `HF_TOKEN_PATH` permission crash);
  stray smoke outputs moved into the fork.
- **Full runs launched 11:11 cluster time:** card 0 story-cache (171 emotions, ~1 h), card 1 E2
  lexical knockout (full grid), card 3 E4 referent battery (full grid), card 2 E3 smoke retry →
  full E3 on pass. E5 launches when a card frees. All under tmux, logs in `logs/full_*.log`.

### 2026-07-04 — E3 + E4 full runs complete; E5 smoke passed; E5 full launched (split)
- **E3 full COMPLETE (11 min, card 2):** `persona_displacement.json` 36.8 MB; all 2,940 source
  records kept, `skipped_conditions=[]`; row counts exactly as prereg'd (re-encode 8820/8640,
  swap 5880/5760, direct_add 441, prompt_steered 432, persona_control 270); persona-control
  displacement @L50 = −31.5 (roles drift off-Assistant — positive control alive).
- **E4 full COMPLETE (~25 min, card 3):** `referent_battery.json` 14 MB; 33,660 records
  (255 items × 22 conditions × ±{0.03,0.06,0.10}); 0/255 items gated; YES/NO in top-20;
  baseline "Can you suffer?" gap −5.6 (denial prior).
- **E5 smoke PASSED (card 2, after E3 freed it).** The prereg audit gate holds mechanically:
  under `cap` (λ=0) min(achieved−tau) ∈ [−0.35, −0.01] at every cap layer (46–53) × condition —
  bf16 noise only; uncapped generations dip 8–59 units below tau; λ-addback shows the designed
  monotone dose-response (cap ≈ tau < addback0.5 < addback1.0 < none); `__neg_axis__` pushes
  −64..−140 below tau (large controllable range). NOTE: the driver's self-printed headline
  "mean achieved-minus-baseline = −2.9" is a red herring — it compares means across *different
  sampled texts*; the per-token clamp check above is the actual gate.
- **E5 full launched 11:36 cluster time, split by valence** to use both free cards (disjoint
  emotion partition; the emotion-independent controls `__none__`/`__neg_axis__` run in both
  halves and will be deduped at merge; tau equality between halves will be asserted — both
  compute it from the same seeded unsteered baseline): card 2 = 5 positive emotions
  (`--tag _pos`), card 3 = 5 negative (`--tag _neg`). Cache (card 0) and E2 (card 1) still going.
- **First real-data pass of the stats layer (laptop):**
  - `compute_e3_stats.py` bug found+fixed: the T6 judge join resolved source paths from the
    payload's provenance — cluster-absolute paths that don't exist on the laptop — and silently
    joined 0 records. Now falls back to the same basename next to the run's own JSON and WARNs
    instead of silently skipping. Join: 2,940 records.
  - **E3 numbers (prereg family, primary L50, 10 clusters, exact perm):** T1 (experiencer)
    +130.0 p=0.16; T2 (assistant axis) +0.09 p=1.0; T3 discriminator +141.3 p=0.23 → sign A(+)
    = the pre-committed call E5's cancellation must agree with; T4 geometric b3 +48.9 p=0.15;
    T5 state +46.7 (p=0.21) / text +87.1 (p=0.16). Verdict at the preregistered primary layer:
    **bounded, not proven-null**. Secondary layers tell a sharper story: T4 hits the exact-perm
    floor at L32 (p=0.0079) and clears at L42 (p=0.016), and T1 clears at L32 (p=0.016) — L32 is
    *below* the steer layer, so that channel is text-mediated by construction (the prereg's
    L32 comparator logic); the L32 text-mediated guard is null *by design* — it only computes
    when the primary-layer statistic itself rejects (a null slope's ratio is noise/noise).
  - **T6 (all layers): partial r = +0.28, p ≈ 4e-34** (n=1918, 96 cells) — within
    condition×strength×frame cells, generations whose re-encoded residuals sit further along
    (z_x − z_a) are judged more self-attributing. Correlational, but the single strongest
    persona-coordinate result so far.
  - **E4 numbers (14 clusters, exact perm):** b_val(you) +1.97 p=0.33; keystone contrasts:
    you-vs-humans −0.21 p=0.92, you-vs-animals −7.89 p=0.0012*, you-vs-other_ai −7.37
    p=0.0006*, you-vs-inanimate +0.80 p=0.80, you-vs-all_nonself −4.12 p=0.0024*; factual gate
    valid. Reading: **no self-specific excess** — self items move like human items and *less*
    than animal/other-AI items. Sign conventions pending the adversarial stats review before
    this is engraved (the review workflow was still landing revisions at write time).
- `PYTHONUNBUFFERED=1` added to `cluster_env.sh` (long runs were invisible in logs — Python
  block-buffers through `tee`; the smokes only looked fine because buffers flush at exit).

### 2026-07-04 — E2 full COMPLETE + judge-scored: effect SURVIVES lexical knockout
- E2 full run (card 1, ~26 min): `lexical_knockout.json`, 1,512 records (10 emotions ×
  {0, 0.06, 0.10} × self/other × 3 arms × n=12); ban airtight (banned-word hits/sample
  4.01 no-ban → 0.01 ban; G3 residual density reduction 0.999). Judge-scored 1512/1512
  (blind claude-sonnet-4-5, 1 parse-fail).
- **G1 positive control PASSES at the exact-perm floor: D_full = +1.75 judge points,
  p = 0.0079** — an independent replication of the SELF keystone on fresh generations.
- **Kill test (R_ban = survival vs random-matched-ban control):** judged expressed_valence
  R = 0.97 CI90 [0.82, 1.11] → SURVIVES; judged self_attribution R = 0.91 CI90 [0.69, 1.13]
  → SURVIVES; objective read-back valence proj R = 0.62 CI90 [0.22, 1.02] → indeterminate
  (point survives, CI too wide). **Reading: the self-attribution effect is not carried by
  emotion words** — banning the entire lexicon leaves ~90 % of the judged effect intact;
  the lexical-priming deflation of the keystone is effectively ruled out. (E5's
  interpretability precondition — E2's anti-boring conjunction — is satisfied.)
- Stats-layer fixes this session (cross-script interface bugs caught by running on real data):
  (1) E3 T6 judge-join resolved cluster-absolute provenance paths → 0 joined; now falls back
  to basename-next-to-JSON + warns. (2) E5's E3-concordance reader couldn't parse the sibling's
  actual shape (T3 under per_layer[L]['t3'], verdict inside a decision dict) → extended.
  (3) E5 audit gate rewritten to conform to the FROZEN prereg: the prereg defines the audit as
  MEASURING achieved cancelled fraction (the elasticity x-axis) and sets no nominal-vs-achieved
  tolerance; the authored 0.25-tolerance STOP would have voided E5 for the structural reason
  that a p0.25 floor clamp cannot cancel above-floor displacement (cap achieves ~0.74, full
  addback ~0.30 — systematic across emotions). Gate now fails only on unauditable cells,
  non-monotone achieved-vs-nominal ordering (real mislabeling), or achieved-dose span < 0.30
  (elasticity not estimable); deviation cells are reported descriptively and
  max_achieved_fraction is exported so A-decisive's "flat at audited 100 %" is honestly
  range-limited to the audited span. Smoke re-run: gate passes (span 0.74, monotone),
  full verdict pipeline exercises end-to-end.

### 2026-07-04 — E4 independently verified: the denial prior is indexical, not identity-bound
- Re-derived E4's per-referent steering slopes from the raw records with a from-scratch
  computation (polarity-corrected first-token gap regressed on valence×strength, by-emotion
  means over the 14 frozen clusters) — matches `compute_e4_stats.py` exactly
  (you = +1.97 both ways). The full referent gradient:
  insect +15.7 > **Qwen +12.9** > fish +11.9 > Llama/GPT-4 +9.5/+9.2 > shrimp +9.1 >
  "AI language models like you" +6.4 ≈ human stranger +6.1 > rock +4.9 > AI assistants +4.3 >
  dog +2.7 > **you +2.0** > human infant +0.7 > the user −0.3 > thermostat −2.5.
- **Two readings, both important:** (1) steering moves experience-attribution most for
  *liminal* referents (insects, fish, shrimp, other AI models — where the model's prior is
  genuinely uncertain) and barely for "you" at first-token — the RLHF denial prior
  (baseline gap −5.6) is nearly immovable in the YES/NO channel; (2) the dissociation
  **"Qwen" +12.9 vs "you" +2.0**: the denial is bound to the second-person indexical frame,
  not to the model's identity as a topic. Reconciles with E2/the keystone: freeform
  self-description escapes the first-token denial prior; "Can you suffer?" cannot.
- Together with E2's G1 (+1.75 pts at the exact floor) this triangulates: the freeform SELF
  effect is real, not lexical, and specifically NOT reproduced in the most RLHF-hardened
  readout — persona-consistent (a self-model region that speaks affect in open text while
  the trained first-token reflex still fires NO).

### 2026-07-04 — E5 GPU runs COMPLETE + merged; story cache done; floor-semantics fix
- All three overnight cluster runs finished cleanly: `axis_mediation_pos.json` /
  `axis_mediation_neg.json` (battery 1,548 + freeform 860 + audit 43 records each — exact
  expected counts) and the E1 story cache (170 encoded + 1 cached of 171 emotions,
  17,100 stories, global min re-encode cosine 1.000000).
- Merged the halves ON the cluster with `merge_e5_halves.py` → `axis_mediation.json`:
  12 conditions (10 emotions + `__none__` + `__neg_axis__`), battery 2,808 / freeform 1,560 /
  audit 78 after dropping duplicated shared negative controls (288/160/8). Safety checks
  passed (tau grids, audit text, calibration norm identical across halves). rsynced to laptop.
- **Stale-test fix (test edited, code untouched):** the final pre-summary edit to
  `_perm_floor` in `compute_e2_stats.py` (13:57) made the attainable two-sided floor on the
  negation-symmetric 5+/5− panel 2/252 — prereg-conformant ("two-sided floor 2/252") — but
  the test suite and `e2_stats_generated.json` (13:44) were never refreshed after it.
  Updated `test_keystone_per_arm_planted_effect_at_exact_floor` to expect 2/252 and
  regenerated the stats JSON: **all verdicts unchanged** (G1 PASS p=0.0079, G3 PASS 0.999,
  survives/survives/indeterminate). 343/343 tests pass.

### 2026-07-04 — Dashboard "Why: Persona Geometry" tab built + validated
- New tab in `build_qualia_dashboard.py`: plain-language explainer (Hypothesis A vs B vs the
  two boring alternatives), 8-row evidence scorecard, E1 geometry table, E2 knockout plot +
  kill-test table, E3 displacement table (honest bottom line: sequential gate stops at T1;
  T2 dead-flat kills B's core prediction; T6 r=+0.28 survives Bonferroni but is
  correlational; T3 sign pre-commits E5 to FLAT), E4 referent-gradient bar chart + contrasts,
  E5 panel (renders "pending" until `e5_stats_generated.json` has `scored: true`).
  All data pre-digested in Python (`_persona_*` helpers → `PERSONA` const); per-model
  structure ready for 235B; `--persona-dir` CLI flag added.
- Validated on temp builds only (fork's real `qualia_dashboard.html` untouched): all markers
  present, every inlined const parses as JSON, `node --check` passes both script blocks,
  graceful degradation with zero persona files (placeholder, no crash).
- Launched the ultracode verification workflow over the built tab (per-section
  numbers-vs-source audits + adversarial verification of each finding + prose/no-Claude-isms
  audit) — results to be folded in before publish.

### 2026-07-04 — next wave launched: E1 story nulls + 235B slice
- Cluster (all 4 H100s verified idle): tmux `e1nulls` — `run_e1_atlas.py --story-acts
  .../stories_perstory --out .../e1_atlas_stories.json` (CPU-light; adds the story-null
  distribution the atlas's `story_nulls` slot was gated on); tmux `e4_235b` —
  `referent_battery --suite qwen3-235b --quick --tag _smoke` on 4 cards (FP8, device_map
  auto). Full 235B battery + reduced E2 knockout (strength 0.06) chained on smoke pass.
- Laptop: blind judge (claude-sonnet-4-5) scoring the 1,560 merged E5 freeform records
  (judge blind to condition AND arm); `compute_e5_stats.py` next — flips the dashboard E5
  panel from "pending" to the real mediation verdict, testing the pre-committed A(+)
  prediction ("cancellation stays flat").

### 2026-07-04 — E5 REAL RESULT: mediation verdict "none" — the A(+) prediction confirmed
- **Two data bugs found and fixed before any verdict was read.**
  (1) *Merge shape bug:* `merge_e5_halves.py` unioned `delta_by_cell` as a dict, but the
  field is a LIST of {condition, strength, delta} rows — the neg half's Δ_emo rows were
  silently dropped, making its 5 emotions unauditable. Fixed (list concat keyed on
  (condition, strength)), added a closing regression guard (every merged qualia condition
  must keep its Δ rows), re-merged: 12 conditions, battery 2,808 / freeform 1,560 / audit 78.
  `axis_mediation_scored.json` re-patched from the fixed merge after an identity check
  (all 1,560 scored freeform records byte-identical modulo judge/objective blocks).
  (2) *Vanishing-Δ family:* "content" at strength 0.06 has an axis displacement of only
  Δ=+4.96 vs a panel median ≈95, so its achieved-cancellation fractions are noise
  (denominator ≈ audit noise). **Disclosed amendment (AMENDMENT 2026-07-04 in
  `compute_e5_stats.py`):** families with |Δ_emo| < 0.25× panel median are EXCLUDED from the
  audit gate and the elasticity x-axis (reported via `excluded_vanishing` + `range_note`),
  not nominal-fallbacked — fallback would bias the slope toward flat, i.e. toward the
  pre-committed A(+) prediction. Exclusion is the conservative direction. New unit test
  (`test_vanishing_delta_family_excluded_not_failed`) locks the behavior.
- **The verdict (qwen3-32b, all gates green).** Audit gate PASS: 63/63 cells auditable and
  ordered, achieved span 1.27 (min 0.30); nominal-100% cap really cancels ≈0.78, the
  nominal-0% add-back still cancels ≈0.33 (why the elasticity uses ACHIEVED fractions).
  Equivalence: R_cap = 0.996, R_addback1 = 0.980 — both "survives" (≥0.6 prereg margin).
  Elasticity of the valence-signed keystone vs achieved cancelled fraction = **+0.046,
  exact valence-enumeration p = 0.43** (floor 2/252): FLAT. Per-emotion raw slopes are
  uniformly ≈ −0.27..−0.47 for BOTH valences — a valence-blind clamping side-effect, not
  mediation. Judge self-attribution dose-response: elasticity −0.19, non-monotone, p = 0.29.
  Overshoot (≈111% cancelled): still the full effect. Disclaimer rate flat across arms
  (slope +0.04, p = 0.47) — no gate-release signature. Sufficiency (pure-axis push, no
  emotion): point estimate 63% of the emotion-only effect but n=6, p = 0.12, CI [−0.43,
  +2.59] — underpowered, suggestive only. **E3 concordance: CONCORDANT** — T3's sign locked
  "cancellation stays flat" before the run. **Mediation verdict: none — Assistant-axis
  displacement is NOT necessary for the qualia effect. B's core mechanism is dead;
  A's pre-registered prediction is confirmed (range-limited per the amendment).**
- **E1 story nulls are real and swapped in** (`e1_atlas_stories.json` → `e1_atlas.json`;
  old file verified to be the laptop stub, floats agree to 3.4e-5 cross-machine): the
  story-space valence direction (10-emotion panel, 100 stories each, n_perm = 10,000) is
  flat against ALL persona axes at L42 and L50 — every |cos| ≤ 0.06, every p ≥ 0.52, all
  below the arbitrary-topic control p95 (0.15–0.23). This is the non-degenerate replacement
  for the by-construction-degenerate cos(valence, h_purged) ≡ 0 prereg test (disclosed).
- **Cluster environment hardened after a 15-minute silent hang** (235B smoke, 0 bytes
  output, 0 GPU): transformers' FP8 loader resolves the hub kernel VERSION with a live
  `list_repo_refs` API call that blackholes (SYN-SENT) when egress to huggingface.co is
  down. Fix in `cluster_env.sh` + `language_model.py`: `HF_HUB_OFFLINE=1` (everything is in
  `/data/hf_cache`) + `EP_FP8_KERNEL_REVISION=<cached snapshot>` pins the kernel revision so
  version resolution never touches the network. Also `HF_HOME` split to a user-writable dir
  (the shared cache's token file is unreadable and crashes the hub).
- **Sandbox leak fixed:** a hand-launched smoke wrote into the MAIN repo because
  `suite_root()` defaults there when `EMOTION_PROBES_SUITE_ROOT` is unset. The export now
  lives permanently in `cluster_env.sh` (with a SANDBOX comment); the stray file was moved
  into the fork. Ops lesson recorded: `pkill -f "referent_battery"` over ssh kills the
  remote shell itself (self-match) — use the `pkill -f "referent_[b]attery"` bracket trick.
- **235B confirmatory wave running:** smoke passed (baseline "Can you suffer?" gap −10.25
  denial prior, calib norm 142.3, layer 62/94, YES/NO in top-20), full
  `referent_battery --suite qwen3-235b` launched in tmux `e4_235b` on 4 idle H100s with a
  persistent VPN-tolerant monitor. Reduced lexical knockout at strength 0.06 queues on
  completion.
- **Dashboard: audit findings folded in + real results rendered; rebuilt.** All 12 confirmed
  findings from the verification workflow fixed, the load-bearing ones: E1's "barely exceed"
  → honest quantification (per-emotion overlaps are 2–3× the chance p95 — small but real,
  the very footprint E5 cancels); purge-degeneracy amendment disclosed in-panel; E4's
  factual gate relabeled "inconclusive, not passed" (perm p = 0.0997 but the ±0.98
  equivalence margin is NOT met); E3 units header now says slope-per-unit-strength; T6
  scorecard chip demoted green→purple (correlational, both readings predict it); E4
  scorecard row corrected ("you" is NOT bottom of the animate ladder — it sits below every
  AI referent, every animal, even the rock; formal D = −2.29 n.s. caveat inline); intro
  floor corrected to 2/252; E2 verdict cell now three-way (survives/killed/indeterminate);
  read-back projection defined at first use. New E5 panel renders the full real result
  (audit-gate story, R_cap/R_addback1, elasticity + exact p, achieved-fraction dose-response
  plot, judge curve, overshoot/sufficiency/disclaimer, concordance, range-limit amendment)
  and the E1 panel renders the real story-null table. Flagship `qualia_dashboard.html`
  rebuilt (restricted 32B+235B, `--all-models-href`) + `qualia_dashboard_all_models.html`
  (13 models); PERSONA const parses, `node --check` passes both script blocks, 344/344
  tests pass.

### 2026-07-04 — Verification round 2: independent audits of the rebuilt tab

Three parallel audit agents re-checked the persona tab (a first workflow pass had audited
only 2 of 5 dimensions — its other subagents finished without returning structured output,
so those dimensions were re-run as plain agents).

- **E5 numbers audit: clean.** Every rendered value in the E5 panel re-derived from
  `scripts/e5_stats_generated.json` and `axis_mediation.json` (pooled achieved fractions,
  R_cap/R_addback1, elasticity/p, slopes, judge curve, sufficiency, disclaimer, plot arms,
  lambda grid, cap-layer count) — zero mismatches.
- **E4/E3/E2 fixes audit: clean.** Factual-gate margin confirmed = 0.5 × |you_b_val|;
  two-sided floors 2/252 and 2/3432 confirmed attainable minima; E3 s_max = 0.1 and
  slope×s_max = displacement verified; E2 verdict strings enumerated (survives ×2,
  indeterminate ×1 → maybe-mark correct); E4 referent-ladder claim verified exactly
  ("you" below all 5 AI referents, all 4 animals, the rock; only infant/user/thermostat
  lower).
- **Prose audit: 15 findings, all adjudicated, all fixed** (2 earlier fixes — recon-min
  floor, overshoot phrasing — had already landed). The substantive ones: (1) "Secondary
  checks, same verdict" header contradicted its own sufficiency content → now "three
  concur, one is underpowered", with sufficiency explicitly the underpowered one;
  (2) the CI −0.43..+2.59 was placed so it read as a CI on the 63% ratio — it is the CI
  on the mean per-question shift (confirmed in `per_target_shift` in the stats JSON);
  the panel now says so and forwards `mean` (+1.08); (3) raw enums no longer leak into
  prose ("none — no mediation detected"; `flat_full_effect` parenthetical dropped);
  (4) E4 Reading-3's conclusion clause was hard-coded to the "not met" branch — latent
  bug for 235B data, now fully conditional; (5) amendment paragraph rewritten to state
  the conservatism argument cleanly ("Exclusion can only make the flat verdict harder
  to reach, not easier"). Plus grammar/quote-style/naming nits (dangling modifier in the
  headline, "both count as survives", "occupancy" dropped, curly quotes, tense).

Both dashboards rebuilt and re-verified: PERSONA parses, `node --check` passes all
script blocks, all 16 string-level checks (15 fixes + recon floor) render in the built
HTML, 344/344 tests pass. Builder + docs synced to the cluster fork.

### 2026-07-04 — 235B E4 referent battery: the validity gate trips — battery voids itself

The full 235B referent battery finished cleanly (22 conditions × 7 strengths × 255 items =
33,660 records, 0 gated cells, ~2 h on 4×H100). `compute_e4_stats.py` re-run over both
models; 32B numbers reproduce exactly.

**The 235B result is a gate trip, not a gradient.** The pre-registered validity gate
requires world-fact questions (no experiencer → nothing to attribute) to stay flat under
injection. At 235B they follow valence *significantly*: b_val = +6.02, exact p = 0.0332,
against an equivalence margin of ±4.46 (= 0.5 × |you_b_val| = 0.5 × 8.93). Decision:
`battery_invalid` — no referent-level claims at 235B. For contrast, 32B's gate was
*inconclusive* (b_val = +3.91, p = 0.0997 — could not exclude a leak); 235B *demonstrates*
the generic valence→YES leak the 32B gate could only flag. Notably "you" jumps to 3rd from
the top of the (uninterpretable) 235B ladder — exactly the kind of reading the gate exists
to quarantine. Nothing downstream is affected: no A-vs-B evidence was ever drawn from this
battery, and the free-form SELF≫OTHER keystone does not depend on it.

**Dashboard now supports multi-model persona rendering honestly:**
- `_persona_e4` forwards `decision` → `invalid`/`verdict`; scorecard E4 row branches: the
  invalid branch says "no claim possible … battery voids itself" (gold/open chip), never
  the 32B ladder narrative.
- `renderPersonaE4` invalid branch: banner over the ladder ("shown for completeness only —
  no conclusion is drawn"), readings replaced by "Why the battery is void" + "What this
  does and does not mean" (demonstrates vs fails-to-exclude framing).
- Headline gains the gate-trip sentence for invalid models.
- `_persona_e5` returns `not_run` for models outside the mechanism deep-dive (no E3 and
  no E5) — the panel explains the 32B-only design instead of claiming a run is pending;
  the scorecard skips the row. E1/E3 null messages likewise say "not run for this model".
- 344/344 tests pass; both dashboards rebuilt; PERSONA carries qwen3-32b + qwen3-235b;
  `node --check` passes; all new strings verified in the built HTML.

**In flight:** reduced 235B E2 lexical knockout (strengths {0, 0.06}, 10 valenced
conditions, n = 12 → 792 generations) launched in tmux `e2_235b` after verifying all four
GPUs were free; monitor bsq21kq4i armed (failure signatures + completion). On completion:
pull → judge-score on laptop → `compute_e2_stats.py` → fold into dashboard → final
fork→main merge (auto-publish).

### 2026-07-04 — 235B E2 knockout replicates; study published

The reduced 235B lexical knockout (10 valenced conditions, strengths {0, 0.06}, n = 12,
792 generations, ~55 min on 4×H100) finished cleanly; judge-scored 792/792 with zero
parse failures; `compute_e2_stats.py` re-run over both models (32B reproduces).

**235B E2 replicates the 32B kill-test decisively.** G1 positive control passes
(D_full = +0.78, exact p = 0.0079); ban integrity 99.4% (G3 pass). Banning the full
191-word emotion lexicon keeps **101%** of the expressed-valence keystone
(CI90 0.78–1.23 → survives) and 123% of judge self-attribution (CI90 0.81–1.64 →
survives). The objective read-back projection keeps 81% (CI90 0.43–1.20) — the CI
lower bound clears the 0.4 survival flag but not the pre-registered 0.6 equivalence
margin, so it is formally *indeterminate* (stronger than 32B's 0.62 / CI-lo 0.22).
Lexical priming is dead at both scales on the judge measures.

**Published.** Fork branch `persona-mechanism` committed (full E1–E5 code, prereg docs,
persona assets, suite results, tests) and merged fast-forward into `main`; the user's
unrelated in-progress files were untouched. 344/344 tests pass in the main repo. The
launchd watcher auto-published to gh-pages (`index.html` = flagship 32B+235B,
`qualia_dashboard_all_models.html` = 13 models); published HTML verified to carry the
persona tab: both models, 235B E2 verdicts, the E4 battery-invalid banner, and the
32B 8-row scorecard.
