# P0 Issues

## P0-01 Confirmatory protocol freeze
Priority: Critical  
Status: Ready

Acceptance criteria:

- protocol and JSON config committed;
- frozen-model and source-reference artifacts hashed;
- Git commit and dirty state recorded;
- snapshot copied to independent storage.

## P0-02 Standard calibration controls
Priority: Critical  
Status: Not started

Acceptance criteria:

- participant mean LOO;
- median/MAD normalization;
- rank normalization;
- OneStop-retrained content-only 29 model;
- 100 seeded random 12-feature groups;
- paired participant-cluster bootstrap;
- one consolidated CSV and figure.

## P0-03 Unused-language repository audit
Priority: Critical  
Status: Not started

Acceptance criteria:

- all non-English languages audited without inspecting performance;
- deterministic language selection executed;
- candidate audit CSV frozen;
- label orientation and AOI/feature schema pass.

## P0-04 Frozen confirmatory prediction
Priority: Critical  
Status: Blocked by P0-03

Acceptance criteria:

- unlabeled target manifest frozen;
- unlabeled Full 41 features frozen;
- four primary-method scores frozen;
- all files hashed before labels are joined.

## P0-05 Confirmatory statistics
Priority: Critical  
Status: Blocked by P0-04

Acceptance criteria:

- 2,000 paired participant-cluster bootstrap replicates;
- 2,000 within-participant permutations;
- primary ROC-AUC contrast;
- secondary metrics;
- global and participant-balanced selective prediction;
- strong/partial/non-confirmation classification.

## P0-06 Ethics determination
Priority: Critical  
Status: Requires institution response

Acceptance criteria:

- institution-specific determination recorded;
- reviewer-note wording drafted;
- provisional command-style ethics text removed from paper.

## P0-07 Submission hygiene
Priority: Critical  
Status: Not started

Acceptance criteria:

- internal Appendix B removed;
- references verified;
- document and image metadata inspected;
- supplemental material anonymized;
- figure alt text added;
- coverage wording corrected.
