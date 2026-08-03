# P0 Execution Checklist

## P0-1 — Freeze the research protocol

- [ ] Place `docs/CONFIRMATORY_PROTOCOL.md` in the project.
- [ ] Place `configs/iui_p0_confirmatory_config.json` in the project.
- [ ] Commit or stash all unrelated changes.
- [ ] Run the snapshot script.
- [ ] Confirm that `outputs/iui_p0_protocol_lock_v1/SHA256SUMS` exists.
- [ ] Copy the snapshot directory to a second storage location.
- [ ] Do not change the frozen model, threshold, feature partition, source references, target-selection rule, or primary estimand after this point.

## P0-2 — Development-set control comparisons

Complete on English WebQAmGaze:

- [ ] Frozen Full 41
- [ ] Participant median LOO
- [ ] Participant mean LOO
- [ ] Participant median/MAD normalization
- [ ] Participant rank normalization
- [ ] Content-only 29 model retrained only on OneStop
- [ ] QCN only
- [ ] QCN-PC
- [ ] Random 12-feature-group neutralization, 100 seeds
- [ ] Participant-cluster bootstrap for direct comparisons
- [ ] Save all target-side operations and references in a machine-readable table

Decision rule:

- Do not replace the primary QCN-PC definition based on these results.
- Controls are used to test whether the existing method is explained by a simpler normalization or arbitrary feature-group neutralization.

## P0-3 — Unused-language audit and confirmation

Before model-performance inspection:

- [ ] Audit every unused WebQAmGaze language.
- [ ] Apply the deterministic candidate-selection rule.
- [ ] Freeze the selected target manifest.
- [ ] Generate Full 41 features without joining labels to predictions.
- [ ] Audit exact feature order and label orientation.
- [ ] Generate unlabeled predictions for all four primary methods.
- [ ] Hash the prediction table.
- [ ] Join labels only after the prediction table is frozen.
- [ ] Run 2,000 participant-cluster bootstrap replicates.
- [ ] Run 2,000 within-participant label permutations.
- [ ] Generate global and participant-balanced selective-prediction results.
- [ ] Classify the result as strong, partial, or non-confirmation using the protocol.

## P0-4 — Submission hygiene

- [ ] Remove Appendix B `AUTHOR ACTIONS BEFORE SUBMISSION` from the submission paper.
- [ ] Replace the provisional ethics instruction with an institution-specific statement.
- [ ] Verify all 35 references and update preprints to final versions where available.
- [ ] Remove author names, affiliations, acknowledgments, self-identifying repository paths, document properties, image metadata, and supplemental metadata.
- [ ] Verify that supplemental files are anonymous.
- [ ] Add alt text to every figure.
- [ ] Check that 50% and 20% coverage are described as representative, not optimized.
- [ ] Document code/data availability without deanonymizing the authors.

## P0 stop rule

After the confirmatory target is opened, do not:

- tune thresholds;
- change the quality-feature group;
- select another language because it performs better;
- alter participant inclusion based on errors or confidence;
- add a new target-specific adaptation method.

A negative result is retained and reported.
