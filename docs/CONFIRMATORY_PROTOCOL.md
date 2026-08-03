# IUI 2027 P0 Confirmatory Protocol

Version: 1.0  
Freeze date: 2026-08-03  
Status: **LOCK BEFORE TARGET-LABEL EVALUATION**

## 1. Purpose

This protocol fixes the confirmatory evaluation of the already-developed Quality-Contribution Neutralization with Participant Calibration (QCN-PC).

The development evidence was obtained on the English subset of WebQAmGaze. The confirmatory analysis must therefore be conducted on an unused target subset without changing the model, feature schema, quality-feature partition, source references, decision threshold, calibration rule, coverage points, primary comparison, or statistical procedure after inspecting target performance.

The protocol is intended to reduce researcher-side adaptation to WebQAmGaze outcomes. It does not turn another language within WebQAmGaze into a fully independent fourth dataset. The resulting evidence must be described as an **independent language-level confirmation within the WebQAmGaze collection**.

## 2. Frozen source model

Primary source model:

- Dataset: OneStop regular first-reading trials
- Representation: Full event-free AOI, 41 features
- Classifier: LinearSVC
- Frozen `C`: `0.01`
- Frozen decision threshold: `0.010769`  
  - Rounded manuscript display: `0.01077`
- Preprocessing: training-median imputation, missingness indicators if present in the serialized pipeline, and source-fitted standardization
- Source model, preprocessing, feature order, and source references must be loaded from the existing frozen release artifacts. They must not be refit on the confirmatory target.

Secondary model:

- Compact 27 may be reported only as a prespecified secondary analysis.
- Frozen `C`: `0.1`
- Frozen threshold: `0.00281`
- Compact results cannot replace the Full 41 primary result based on target performance.

## 3. Frozen feature partition

Full 41 is partitioned as follows:

- Measurement-associated group: the following 12 membership/quality features exactly:
  1. `valid_rate`
  2. `active_rate`
  3. `word_box_hit_rate`
  4. `sentence_envelope_hit_rate`
  5. `qmax_mean`
  6. `qmax_median`
  7. `qmax_q10`
  8. `qmax_ge_0_5_rate`
  9. `nearest_distance_mean_word_heights`
  10. `nearest_distance_median_word_heights`
  11. `nearest_distance_q90_word_heights`
  12. `nearest_distance_q95_word_heights`
- Content-related group: all remaining 29 Full features, in the serialized Full 41 feature order.

No target result may be used to move a feature between groups.

## 4. Frozen QCN-PC transformation

For target trial feature vector `x`, let the frozen linear score be

`score(x) = intercept + content_contribution(x) + quality_contribution(x)`.

### 4.1 QCN

Let `r_quality` be the already-frozen OneStop source reference: the median across source participants of their participant-level median quality contribution, computed from source out-of-fold scores.

`score_qcn(x) = score(x) - quality_contribution(x) + r_quality`

The confirmatory analysis must read the stored source reference. It must not recalculate `r_quality` from the target.

### 4.2 Participant calibration

For participant `p` and target trial `i`, the primary calibration is leave-one-trial-out median centering:

`score_pc[p,i] = score[p,i] - median_{j != i}(score[p,j]) + r_participant`

`r_participant` is the frozen OneStop source reference: the median across source participants of their participant-level median out-of-fold score.

The evaluated trial must be excluded from its own participant-center estimate. Target labels must not be used.

### 4.3 QCN-PC

Apply participant calibration after QCN:

`score_qcn_pc[p,i] = score_qcn[p,i] - median_{j != i}(score_qcn[p,j]) + r_qcn_participant`

`r_qcn_participant` is the already-frozen quality-neutralized OneStop participant score reference.

The order is fixed: QCN first, participant calibration second.

## 5. Confirmatory target selection

Candidate target units are non-English WebQAmGaze language subsets not used in method development.

The primary language is selected using the following deterministic, label-outcome-independent rule:

1. Complete repository audit for every non-English language.
2. Exclude a language if:
   - fewer than 40 eligible unique participant IDs remain;
   - both NR and IS are not available for participant-level calibration/evaluation;
   - the existing event-free AOI pipeline cannot create Full 41 features using the same schema;
   - required trial, participant, label-orientation, or AOI audits fail;
   - more than 20% of otherwise eligible trials fail feature extraction.
3. Among remaining languages, choose the language with the largest number of eligible unique participant IDs.
4. Break an exact tie by the largest number of eligible trials, then by alphabetical ISO language code.

Eligibility may use labels only to establish that both predefined classes exist; no performance metric, feature-label association, or model score may be inspected during target selection.

The expected primary candidate from the prior repository inventory is Turkish, but the deterministic eligibility rule above is authoritative.

A second qualifying language may be reported as a secondary sensitivity analysis. It must not be substituted as the primary target after performance inspection.

## 6. Inclusion and exclusion rules

Use the same trial construction, AOI resolution, 30 Hz resampling, feature extraction, and label orientation audit as the frozen English WebQAmGaze pipeline.

Required rules:

- Positive label: NR = 1
- Negative label: IS = 0
- Aggregation unit: unique `participant_id`
- Duplicate source records for one participant are clustered under the same participant ID
- Each included participant must have at least:
  - two usable trials total for leave-one-trial-out centering, and
  - at least one usable NR and one usable IS trial for participant-internal diagnostic metrics
- Trials failing the pre-existing extraction/audit rules are excluded with a machine-readable reason
- No exclusion may depend on model correctness, score magnitude, confidence, or class-specific performance

## 7. Prespecified methods

Primary four-method comparison:

1. `frozen_full41`
2. `participant_median_loo`
3. `qcn_only`
4. `qcn_pc`

Prespecified development-set controls, to be completed on English WebQAmGaze before opening confirmatory performance:

5. `participant_mean_loo`
6. `participant_median_mad`
7. `participant_rank_normalization`
8. `content_only_29_retrained_on_onestop`
9. `random_12_feature_neutralization` with 100 seeded feature groups

The primary confirmatory claim is not changed by which development-set control is best.

## 8. Outcomes

### 8.1 Primary estimand

Difference in pooled ROC-AUC:

`delta_auc = ROC_AUC(qcn_pc) - ROC_AUC(participant_median_loo)`

### 8.2 Primary uncertainty analysis

- 2,000 paired participant-cluster bootstrap replicates
- Resampling unit: unique participant ID
- The same participant bootstrap sample is used for both methods
- Report percentile 95% confidence interval
- Random seed: `20260803`

### 8.3 Secondary outcomes

- Macro F1
- Balanced accuracy
- PR-AUC with NR as positive class
- Accuracy
- NR prediction rate
- Mean within-participant ROC-AUC
- AURC
- Excess AURC
- Confidence-correctness Spearman correlation
- Difference between QCN-PC and participant calibration for each metric

### 8.4 Absolute discrimination

For QCN-PC ROC-AUC:

- 2,000 participant-cluster bootstrap replicates for the absolute 95% CI
- 2,000 within-participant label permutations
- One-sided test of ROC-AUC greater than 0.5
- Random seed: `20260803`

Relative improvement and absolute above-chance discrimination must be interpreted separately.

## 9. Frozen selective-prediction analysis

Confidence:

`confidence_i = abs(score_qcn_pc_i - 0.010769)`

Prespecified coverage points:

- 100%
- 50%
- 20%

Report global and participant-balanced selection.

No coverage point may be chosen as an optimum on the confirmatory target. The 50% and 20% points are representative operating conditions only.

Primary selective summary:

- Macro F1 at 50% participant-balanced coverage
- Number of retained participants
- Per-participant retained-trial distribution

Secondary summaries:

- Global macro F1 and ROC-AUC at 50% and 20%
- AURC and excess AURC
- Comparison with participant calibration using the same confidence definition and coverage

## 10. Interpretation rule

### Strong confirmation

All of the following:

- `delta_auc > 0`
- primary 95% CI excludes 0
- macro F1 difference is nonnegative
- QCN-PC absolute ROC-AUC is above 0.5 by the prespecified permutation test

### Partial confirmation

At least:

- `delta_auc > 0`, but its CI includes 0; and
- direction of macro F1, quality-contribution variability, or score-location repair is consistent with the English development result

Describe as limited or directional confirmation.

### Non-confirmation

- `delta_auc <= 0`, or
- major implementation/audit failure prevents the frozen method from being applied

Do not add a new target-specific correction. QCN-PC must be downgraded to an exploratory case study, while the event-free representation and robustness results remain separate contributions.

## 11. Blinding and file separation

Before confirmatory scoring:

- Produce an unlabeled feature table and score table.
- Store labels in a separate file.
- Generate SHA-256 hashes for:
  - protocol
  - config
  - Git commit
  - frozen model directory
  - frozen source-reference artifacts
  - unlabeled target manifest
  - unlabeled feature table
  - unlabeled prediction table
- Only after the prediction table is frozen may labels be joined for metric computation.

The final release must retain both the pre-label and labeled manifests.

## 12. Deviations

Every deviation must be appended to `CONFIRMATORY_DEVIATIONS.md` before the labels are opened when possible. Each entry must contain:

- timestamp
- reason
- affected rule
- whether target labels or performance had already been inspected
- exact code/config change
- expected direction of influence

Undocumented deviations invalidate the confirmatory designation.

## 13. Required outputs

Create under:

`outputs/iui_p0_confirmatory_<language>_v1/`

Required files:

- `audit/candidate_language_audit.csv`
- `audit/inclusion_exclusion.csv`
- `audit/label_orientation_audit.json`
- `audit/feature_schema_audit.json`
- `manifests/unlabeled_target_manifest.csv`
- `predictions/unlabeled_frozen_scores.csv`
- `predictions/labeled_confirmatory_predictions.csv`
- `tables/primary_confirmatory_results.csv`
- `tables/bootstrap_differences.csv`
- `tables/selective_prediction.csv`
- `figures/external_stage_comparison.png`
- `figures/risk_coverage.png`
- `text/confirmatory_summary.md`
- `release_manifest.json`
- `SHA256SUMS`

## 14. Manuscript wording

Until strong confirmation is obtained:

- QCN-PC on English WebQAmGaze is described as exploratory.
- The unused-language result is described as a frozen language-level confirmation.
- It is not called a fully independent fourth-dataset replication.
- Target labels are stated not to be used in feature extraction, transformation, source-reference estimation, participant centering, threshold choice, confidence calculation, or trial selection.
