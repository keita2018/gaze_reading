from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

SELECTED = [
  {
    "category": "Table 3: ZuCo nested dual holdout",
    "relative_path": ".\\outputs\\soft_aoi_official_common16_nested_dual\\common16_nested_dual_manifest.json",
    "file_name": "common16_nested_dual_manifest.json",
    "extension": ".json",
    "bytes": 1247,
    "sha256": "c30c702b74268013838d7cffc456b3769fb8fc4465864bb83aa2f0cdae07ade7",
    "rationale": "Pooled, participant-level, pairwise, hyperparameter, and implementation evidence"
  },
  {
    "category": "Table 3: ZuCo nested dual holdout",
    "relative_path": ".\\outputs\\soft_aoi_official_common16_nested_dual\\common16_nested_dual_pooled.csv",
    "file_name": "common16_nested_dual_pooled.csv",
    "extension": ".csv",
    "bytes": 1046,
    "sha256": "8b5beede0ba80d22c1e2f13507c1105c8da306202a5b913c023adc8f568ff076",
    "rationale": "Pooled, participant-level, pairwise, hyperparameter, and implementation evidence"
  },
  {
    "category": "Table 3: ZuCo nested dual holdout",
    "relative_path": ".\\outputs\\soft_aoi_official_common16_nested_dual\\common16_nested_dual_summary.csv",
    "file_name": "common16_nested_dual_summary.csv",
    "extension": ".csv",
    "bytes": 1856,
    "sha256": "67fcd963f53a3e1a2988725998f3da4112d95389821d4439984a5b0fcd9b4cf5",
    "rationale": "Pooled, participant-level, pairwise, hyperparameter, and implementation evidence"
  },
  {
    "category": "Table 3: ZuCo nested dual holdout",
    "relative_path": ".\\outputs\\soft_aoi_official_common16_nested_dual\\common16_nested_dual_subject_metrics.csv",
    "file_name": "common16_nested_dual_subject_metrics.csv",
    "extension": ".csv",
    "bytes": 13804,
    "sha256": "a2e8984630d25d53a0b9d7f42dac4bb16b6e7679c77e3f4c62cb05fbc70a71c4",
    "rationale": "Pooled, participant-level, pairwise, hyperparameter, and implementation evidence"
  },
  {
    "category": "Table 3: ZuCo nested dual holdout",
    "relative_path": ".\\outputs\\soft_aoi_official_common16_nested_dual\\common16_nested_dual_pairwise_tests.csv",
    "file_name": "common16_nested_dual_pairwise_tests.csv",
    "extension": ".csv",
    "bytes": 6431,
    "sha256": "8f90d9745f39a4cbb0be30ed5724d5f6f13eefd252093aa2da5a8979f0c56f7d",
    "rationale": "Pooled, participant-level, pairwise, hyperparameter, and implementation evidence"
  },
  {
    "category": "Table 3: ZuCo nested dual holdout",
    "relative_path": ".\\outputs\\soft_aoi_official_common16_nested_dual\\common16_nested_dual_selected_c.csv",
    "file_name": "common16_nested_dual_selected_c.csv",
    "extension": ".csv",
    "bytes": 11669,
    "sha256": "38a095c56763107e03bedf94dbcef6f69795716f255105efd662fcb2006dcd2b",
    "rationale": "Pooled, participant-level, pairwise, hyperparameter, and implementation evidence"
  },
  {
    "category": "Table 3: ZuCo nested dual holdout",
    "relative_path": ".\\src\\models\\run_zuco_soft_aoi_official_common16_nested_dual.py",
    "file_name": "run_zuco_soft_aoi_official_common16_nested_dual.py",
    "extension": ".py",
    "bytes": 51362,
    "sha256": "eb5449122e48ee8fe11a2842d96af9acb96bc5d6aaa8af91a351aaf2da38c57e",
    "rationale": "Pooled, participant-level, pairwise, hyperparameter, and implementation evidence"
  },
  {
    "category": "Figure 4 / Table 4: Soft–Hard clean",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_clean_30hz\\soft_vs_hard_manifest.json",
    "file_name": "soft_vs_hard_manifest.json",
    "extension": ".json",
    "bytes": 1523,
    "sha256": "e763e4554272eb2668d3b8d63415aa6c29b00b8c5c8643d614fc6f7ddabb01e9",
    "rationale": "Clean common25 metrics and Soft–Hard control"
  },
  {
    "category": "Figure 4 / Table 4: Soft–Hard clean",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_clean_30hz\\soft_vs_hard_summary.csv",
    "file_name": "soft_vs_hard_summary.csv",
    "extension": ".csv",
    "bytes": 1825,
    "sha256": "b4db0fa849fee0b42d76cba11f050f903b2663e4f1dfe220564ef95dc5b12522",
    "rationale": "Clean common25 metrics and Soft–Hard control"
  },
  {
    "category": "Figure 4 / Table 4: Soft–Hard clean",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_clean_30hz\\soft_vs_hard_subject_metrics.csv",
    "file_name": "soft_vs_hard_subject_metrics.csv",
    "extension": ".csv",
    "bytes": 6572,
    "sha256": "467e8bbe25aa3a3b5d80288b0f7025ecea01f96fe8f8c552c3fb0d8f1bfc1dba",
    "rationale": "Clean common25 metrics and Soft–Hard control"
  },
  {
    "category": "Figure 4 / Table 4: Soft–Hard clean",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_clean_30hz\\soft_vs_hard_pairwise_tests.csv",
    "file_name": "soft_vs_hard_pairwise_tests.csv",
    "extension": ".csv",
    "bytes": 1516,
    "sha256": "eb891b8fc2add2f36eaa622c8cea5d98bfd54575c2377627c645f6b297018dc2",
    "rationale": "Clean common25 metrics and Soft–Hard control"
  },
  {
    "category": "Figure 4 / Table 4: Soft–Hard clean",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_clean_30hz\\soft_vs_hard_feature_sets.json",
    "file_name": "soft_vs_hard_feature_sets.json",
    "extension": ".json",
    "bytes": 2230,
    "sha256": "9e9578829a1f08348a12757aba395c7715bce5c9b920bbe52bb23e07388c1883",
    "rationale": "Clean common25 metrics and Soft–Hard control"
  },
  {
    "category": "Figure 4 / Table 4: Soft–Hard clean",
    "relative_path": ".\\src\\models\\run_zuco_soft_vs_hard_aoi_clean_30hz.py",
    "file_name": "run_zuco_soft_vs_hard_aoi_clean_30hz.py",
    "extension": ".py",
    "bytes": 17509,
    "sha256": "c36a10e4c4b0a5d0e075635e282d9e830ca00ef9d96eb5c507d6f9876f079761",
    "rationale": "Clean common25 metrics and Soft–Hard control"
  },
  {
    "category": "Figure 4 / Table 4: 10 Hz transfer",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_train30_test10\\soft_vs_hard_train30_test10_manifest.json",
    "file_name": "soft_vs_hard_train30_test10_manifest.json",
    "extension": ".json",
    "bytes": 2582,
    "sha256": "52d45b70b13716ab24eb48cb2d0c59b9845a7738d8e5bce9d5162e4c18c44680",
    "rationale": "Train-30/test-10 rate-mismatch evidence"
  },
  {
    "category": "Figure 4 / Table 4: 10 Hz transfer",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_train30_test10\\soft_vs_hard_train30_test10_summary.csv",
    "file_name": "soft_vs_hard_train30_test10_summary.csv",
    "extension": ".csv",
    "bytes": 644,
    "sha256": "27ea7813a8e389f19ee429ef672a20d3e03c8d5df3c1fa082bd668fe4a369c4e",
    "rationale": "Train-30/test-10 rate-mismatch evidence"
  },
  {
    "category": "Figure 4 / Table 4: 10 Hz transfer",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_train30_test10\\soft_vs_hard_train30_test10_subject_metrics.csv",
    "file_name": "soft_vs_hard_train30_test10_subject_metrics.csv",
    "extension": ".csv",
    "bytes": 7721,
    "sha256": "bb1b0a278983c43b5d25ed7dce25c3518f0fc1333eaa263af4051240e46e0792",
    "rationale": "Train-30/test-10 rate-mismatch evidence"
  },
  {
    "category": "Figure 4 / Table 4: 10 Hz transfer",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_train30_test10\\soft_vs_hard_train30_test10_tests.csv",
    "file_name": "soft_vs_hard_train30_test10_tests.csv",
    "extension": ".csv",
    "bytes": 1177,
    "sha256": "62ab163d50b8495aa677ab573a7b3e692a4ef4f2979d1550c6a564db393c8fe0",
    "rationale": "Train-30/test-10 rate-mismatch evidence"
  },
  {
    "category": "Figure 4 / Table 4: 10 Hz transfer",
    "relative_path": ".\\src\\models\\run_zuco_soft_vs_hard_aoi_train30_test10.py",
    "file_name": "run_zuco_soft_vs_hard_aoi_train30_test10.py",
    "extension": ".py",
    "bytes": 19911,
    "sha256": "0ef63134ab72322cc5d6a64ddf6bcdb0c0d9c8466eb180511d03025b7c5ec9db",
    "rationale": "Train-30/test-10 rate-mismatch evidence"
  },
  {
    "category": "Figure 4 / Table 4: coordinate noise",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_coordinate_noise_30hz\\soft_vs_hard_noise_manifest.json",
    "file_name": "soft_vs_hard_noise_manifest.json",
    "extension": ".json",
    "bytes": 2924,
    "sha256": "df3abeaa45b5f06400b246d61c70f660738020ccc19df2eb92f18c450fb441cb",
    "rationale": "50-pixel noise metrics, repeats, tests, and implementation"
  },
  {
    "category": "Figure 4 / Table 4: coordinate noise",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_coordinate_noise_30hz\\soft_vs_hard_noise_summary.csv",
    "file_name": "soft_vs_hard_noise_summary.csv",
    "extension": ".csv",
    "bytes": 557,
    "sha256": "60ee3e232ed5d8652e04ae5c499347a507837d5af27ec71ccde67aa4d3a467c8",
    "rationale": "50-pixel noise metrics, repeats, tests, and implementation"
  },
  {
    "category": "Figure 4 / Table 4: coordinate noise",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_coordinate_noise_30hz\\soft_vs_hard_noise_subject_metrics.csv",
    "file_name": "soft_vs_hard_noise_subject_metrics.csv",
    "extension": ".csv",
    "bytes": 5899,
    "sha256": "cb2e6d933eb0538e8d7b7ad0265905e8c04a4d0bcd0f5b844d5edfec9ab2b805",
    "rationale": "50-pixel noise metrics, repeats, tests, and implementation"
  },
  {
    "category": "Figure 4 / Table 4: coordinate noise",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_coordinate_noise_30hz\\soft_vs_hard_noise_subject_repeat_metrics.csv",
    "file_name": "soft_vs_hard_noise_subject_repeat_metrics.csv",
    "extension": ".csv",
    "bytes": 32492,
    "sha256": "e4718dad93f8e0daca763563e7b36b99b11bed09905c611ca7fa046433e3d9c1",
    "rationale": "50-pixel noise metrics, repeats, tests, and implementation"
  },
  {
    "category": "Figure 4 / Table 4: coordinate noise",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_coordinate_noise_30hz\\soft_vs_hard_noise_tests.csv",
    "file_name": "soft_vs_hard_noise_tests.csv",
    "extension": ".csv",
    "bytes": 4965,
    "sha256": "ca23797e0eb7532b824c29dccc815d158c3bc069262d37252f1d716a6436c489",
    "rationale": "50-pixel noise metrics, repeats, tests, and implementation"
  },
  {
    "category": "Figure 4 / Table 4: coordinate noise",
    "relative_path": ".\\src\\models\\run_zuco_soft_vs_hard_aoi_coordinate_noise_30hz.py",
    "file_name": "run_zuco_soft_vs_hard_aoi_coordinate_noise_30hz.py",
    "extension": ".py",
    "bytes": 39929,
    "sha256": "9bbc2a3eec48d9f70b7331fad3cc08fe0b782b8b3b1d1ce9e505a7fad17c984b",
    "rationale": "50-pixel noise metrics, repeats, tests, and implementation"
  },
  {
    "category": "Figure 4 / Table 4: missingness",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_missingness_30hz\\soft_vs_hard_missingness_manifest.json",
    "file_name": "soft_vs_hard_missingness_manifest.json",
    "extension": ".json",
    "bytes": 2633,
    "sha256": "1b6472ef5941b1162fd0c0b913c255ad6fa23afa9d052f0b1bfe29c060f87680",
    "rationale": "Random/burst 40% missingness metrics and tests"
  },
  {
    "category": "Figure 4 / Table 4: missingness",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_missingness_30hz\\soft_vs_hard_missingness_summary.csv",
    "file_name": "soft_vs_hard_missingness_summary.csv",
    "extension": ".csv",
    "bytes": 937,
    "sha256": "613c8fd8b0d248e4b0196d9749adafdc72d0b02ff3bb5be750d26157c3bc98f7",
    "rationale": "Random/burst 40% missingness metrics and tests"
  },
  {
    "category": "Figure 4 / Table 4: missingness",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_missingness_30hz\\soft_vs_hard_missingness_subject_metrics.csv",
    "file_name": "soft_vs_hard_missingness_subject_metrics.csv",
    "extension": ".csv",
    "bytes": 12139,
    "sha256": "e7f0eb4a11c90ed12c5a658895e8b93e1577562e7ee9d59c55651b30d050f37b",
    "rationale": "Random/burst 40% missingness metrics and tests"
  },
  {
    "category": "Figure 4 / Table 4: missingness",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_missingness_30hz\\soft_vs_hard_missingness_subject_repeat_metrics.csv",
    "file_name": "soft_vs_hard_missingness_subject_repeat_metrics.csv",
    "extension": ".csv",
    "bytes": 66288,
    "sha256": "9a556615e93659e87d195426eabcf972c14bbc86edfd9a64226f463e8b5101f0",
    "rationale": "Random/burst 40% missingness metrics and tests"
  },
  {
    "category": "Figure 4 / Table 4: missingness",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_missingness_30hz\\soft_vs_hard_missingness_tests.csv",
    "file_name": "soft_vs_hard_missingness_tests.csv",
    "extension": ".csv",
    "bytes": 1919,
    "sha256": "c6fbafac19ffe1af821a5fd90314f55f6adc85db439fdd944221a42cb69203ae",
    "rationale": "Random/burst 40% missingness metrics and tests"
  },
  {
    "category": "Figure 4 / Table 4: missingness",
    "relative_path": ".\\src\\models\\run_zuco_soft_vs_hard_aoi_missingness_30hz.py",
    "file_name": "run_zuco_soft_vs_hard_aoi_missingness_30hz.py",
    "extension": ".py",
    "bytes": 34521,
    "sha256": "b1d202ba52babd20691cddab76d575c69e9d6c45c971901d6eacf8abb8cbef53",
    "rationale": "Random/burst 40% missingness metrics and tests"
  },
  {
    "category": "Figure 4 / Table 4: compound degradation",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_combined_high_30hz\\soft_vs_hard_combined_high_manifest.json",
    "file_name": "soft_vs_hard_combined_high_manifest.json",
    "extension": ".json",
    "bytes": 2848,
    "sha256": "c0d76c7cb030d134e22ab9f4b457897fb54e41846d4f41866f2b386a5f47d53b",
    "rationale": "Noise+random and noise+burst metrics and tests"
  },
  {
    "category": "Figure 4 / Table 4: compound degradation",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_combined_high_30hz\\soft_vs_hard_combined_high_summary.csv",
    "file_name": "soft_vs_hard_combined_high_summary.csv",
    "extension": ".csv",
    "bytes": 913,
    "sha256": "83b966c78d17ed92abdc4102191438a9eb5e586b2a4614abb9603e1a58f86a5e",
    "rationale": "Noise+random and noise+burst metrics and tests"
  },
  {
    "category": "Figure 4 / Table 4: compound degradation",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_combined_high_30hz\\soft_vs_hard_combined_high_subject_metrics.csv",
    "file_name": "soft_vs_hard_combined_high_subject_metrics.csv",
    "extension": ".csv",
    "bytes": 12113,
    "sha256": "41ee2c329e4f1a227ac6d59c2197ea91e666ccc78f5ebcd2e01e4759d09ec925",
    "rationale": "Noise+random and noise+burst metrics and tests"
  },
  {
    "category": "Figure 4 / Table 4: compound degradation",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_combined_high_30hz\\soft_vs_hard_combined_high_subject_repeat_metrics.csv",
    "file_name": "soft_vs_hard_combined_high_subject_repeat_metrics.csv",
    "extension": ".csv",
    "bytes": 65686,
    "sha256": "261c958073ed1ed46ed5a2175f556fe5899702fe65f0a2cd235598bcd0b86a6c",
    "rationale": "Noise+random and noise+burst metrics and tests"
  },
  {
    "category": "Figure 4 / Table 4: compound degradation",
    "relative_path": ".\\outputs\\soft_vs_hard_aoi_combined_high_30hz\\soft_vs_hard_combined_high_tests.csv",
    "file_name": "soft_vs_hard_combined_high_tests.csv",
    "extension": ".csv",
    "bytes": 1970,
    "sha256": "982cca9b585dceb6d66f7311ecae57e7dd54541d613a7a02d3e6850bab9eef1f",
    "rationale": "Noise+random and noise+burst metrics and tests"
  },
  {
    "category": "Figure 4 / Table 4: compound degradation",
    "relative_path": ".\\src\\models\\run_zuco_soft_vs_hard_aoi_combined_high_30hz.py",
    "file_name": "run_zuco_soft_vs_hard_aoi_combined_high_30hz.py",
    "extension": ".py",
    "bytes": 36640,
    "sha256": "36ccd63a9c6ace398ffbdff90cafdde529ebd5bb731673fb111e6c7da8ec2b1c",
    "rationale": "Noise+random and noise+burst metrics and tests"
  },
  {
    "category": "Figure 4 / Table 4: fixed I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_fixed_benchmark_exact\\ivt_fixed_manifest.json",
    "file_name": "ivt_fixed_manifest.json",
    "extension": ".json",
    "bytes": 1215,
    "sha256": "e5d768ff0ae316e0ceae71af1156c636cbd5e153497fcc1cdcf3bc6ca24ae7f9",
    "rationale": "Fixed detector metrics and zero-event audit"
  },
  {
    "category": "Figure 4 / Table 4: fixed I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_fixed_benchmark_exact\\ivt_fixed_summary.csv",
    "file_name": "ivt_fixed_summary.csv",
    "extension": ".csv",
    "bytes": 1247,
    "sha256": "a847f3f7a805f165cb803991c012f74dcf36ef8aef6c46c18ed794fc9ffc0009",
    "rationale": "Fixed detector metrics and zero-event audit"
  },
  {
    "category": "Figure 4 / Table 4: fixed I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_fixed_benchmark_exact\\ivt_fixed_subject_metrics.csv",
    "file_name": "ivt_fixed_subject_metrics.csv",
    "extension": ".csv",
    "bytes": 18612,
    "sha256": "e6f22a6bdc24d3e2ebe77c10bcb1595d1a70d4ab7e06cd4798a8ec4362aae528",
    "rationale": "Fixed detector metrics and zero-event audit"
  },
  {
    "category": "Figure 4 / Table 4: fixed I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_fixed_benchmark_exact\\ivt_fixed_subject_repeat_metrics.csv",
    "file_name": "ivt_fixed_subject_repeat_metrics.csv",
    "extension": ".csv",
    "bytes": 80095,
    "sha256": "58e702e37395f64e36ba3ad04bdd831c6a4c621f2576dd95fc021bfd1b2e5534",
    "rationale": "Fixed detector metrics and zero-event audit"
  },
  {
    "category": "Figure 4 / Table 4: fixed I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_fixed_benchmark_exact\\ivt_fixed_event_count_summary.csv",
    "file_name": "ivt_fixed_event_count_summary.csv",
    "extension": ".csv",
    "bytes": 1887,
    "sha256": "79dd9e976d5195b631e6685238b16ee61c85775735fa64a6983f6dc4c59a5ff3",
    "rationale": "Fixed detector metrics and zero-event audit"
  },
  {
    "category": "Figure 4 / Table 4: fixed I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_fixed_benchmark_exact\\ivt_fixed_clean_vs_degraded_tests.csv",
    "file_name": "ivt_fixed_clean_vs_degraded_tests.csv",
    "extension": ".csv",
    "bytes": 5575,
    "sha256": "9b9d38dc838d031dabe5f37880daf9b517ef17cfc8adec730be80056b2af65b6",
    "rationale": "Fixed detector metrics and zero-event audit"
  },
  {
    "category": "Figure 4 / Table 4: fixed I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_fixed_benchmark_exact\\ivt_fixed_vs_official_common16_audit.csv",
    "file_name": "ivt_fixed_vs_official_common16_audit.csv",
    "extension": ".csv",
    "bytes": 1896,
    "sha256": "ae2deb3b3bda4440e3ac6beb10e1e726fc3868f4dcbfa1696d898c504d2badd6",
    "rationale": "Fixed detector metrics and zero-event audit"
  },
  {
    "category": "Figure 4 / Table 4: fixed I-VT",
    "relative_path": ".\\src\\models\\run_zuco_ivt_event_baseline_fixed.py",
    "file_name": "run_zuco_ivt_event_baseline_fixed.py",
    "extension": ".py",
    "bytes": 62057,
    "sha256": "f095c4bff68aeb51ad79a9e8b5aee3ea6ece18129d4ea10245900a8033654895",
    "rationale": "Fixed detector metrics and zero-event audit"
  },
  {
    "category": "Figure 4 / Table 4: quality-tuned I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_nested_threshold\\ivt_nested_manifest.json",
    "file_name": "ivt_nested_manifest.json",
    "extension": ".json",
    "bytes": 840,
    "sha256": "8906b87307f59ddc7d3da9971d5d2f874bfe293fa31c1e8f78cd0d477203c68a",
    "rationale": "Nested threshold-tuned detector metrics"
  },
  {
    "category": "Figure 4 / Table 4: quality-tuned I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_nested_threshold\\ivt_nested_summary.csv",
    "file_name": "ivt_nested_summary.csv",
    "extension": ".csv",
    "bytes": 1240,
    "sha256": "ecd8b2261da36677300e3179e05f2903dcf57788672f986743e69e6173bfe995",
    "rationale": "Nested threshold-tuned detector metrics"
  },
  {
    "category": "Figure 4 / Table 4: quality-tuned I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_nested_threshold\\ivt_nested_subject_metrics.csv",
    "file_name": "ivt_nested_subject_metrics.csv",
    "extension": ".csv",
    "bytes": 18616,
    "sha256": "4da0e5d0a97aa3dd0de67331ed5a4fdcd261f469392d8a3b2adb217e5a30d142",
    "rationale": "Nested threshold-tuned detector metrics"
  },
  {
    "category": "Figure 4 / Table 4: quality-tuned I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_nested_threshold\\ivt_nested_subject_repeat_metrics.csv",
    "file_name": "ivt_nested_subject_repeat_metrics.csv",
    "extension": ".csv",
    "bytes": 82137,
    "sha256": "005d523a8451c17bfe1537a935277906bd6c15ab60334f8be0030a481c3f921c",
    "rationale": "Nested threshold-tuned detector metrics"
  },
  {
    "category": "Figure 4 / Table 4: quality-tuned I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_nested_threshold\\ivt_nested_threshold_selection_summary.csv",
    "file_name": "ivt_nested_threshold_selection_summary.csv",
    "extension": ".csv",
    "bytes": 590,
    "sha256": "90f1665ec9b3736bd6e16ba781f1933f4b7fa484a79fbf878d6d53e1f4da710d",
    "rationale": "Nested threshold-tuned detector metrics"
  },
  {
    "category": "Figure 4 / Table 4: quality-tuned I-VT",
    "relative_path": ".\\outputs\\ivt_event_baseline_nested_threshold\\ivt_nested_vs_fixed_tests.csv",
    "file_name": "ivt_nested_vs_fixed_tests.csv",
    "extension": ".csv",
    "bytes": 6105,
    "sha256": "fc8e7f29c190415d9ed192268c656b82c5a5f9dc477e9ec0f791eec0ec4b44be",
    "rationale": "Nested threshold-tuned detector metrics"
  },
  {
    "category": "Figure 4 / Table 4: quality-tuned I-VT",
    "relative_path": ".\\src\\models\\run_zuco_ivt_event_baseline_nested_threshold.py",
    "file_name": "run_zuco_ivt_event_baseline_nested_threshold.py",
    "extension": ".py",
    "bytes": 35418,
    "sha256": "92a5b51cd4c9b751b24a5806118f8b528013bf250f9c164230cbcabdee1e505d",
    "rationale": "Nested threshold-tuned detector metrics"
  },
  {
    "category": "Section 6.3: OneStop internal evaluation",
    "relative_path": ".\\outputs\\onestop_internal_validation_v1\\onestop_internal_pooled_metrics.csv",
    "file_name": "onestop_internal_pooled_metrics.csv",
    "extension": ".csv",
    "bytes": 1419,
    "sha256": "d9b9404b54618ae33518620f04bf2fb92e11d3e472598029f81ef4f7a07e4a7f",
    "rationale": "Independent source evaluation and audit"
  },
  {
    "category": "Section 6.3: OneStop internal evaluation",
    "relative_path": ".\\outputs\\onestop_internal_validation_v1\\onestop_internal_fold_metrics.csv",
    "file_name": "onestop_internal_fold_metrics.csv",
    "extension": ".csv",
    "bytes": 4334,
    "sha256": "1d68833017958e3d2a1b0beaa535642a9eeacd6a44dcfda311880c309a360e9e",
    "rationale": "Independent source evaluation and audit"
  },
  {
    "category": "Section 6.3: OneStop internal evaluation",
    "relative_path": ".\\outputs\\onestop_internal_validation_v1\\onestop_internal_bootstrap_cis.csv",
    "file_name": "onestop_internal_bootstrap_cis.csv",
    "extension": ".csv",
    "bytes": 2224,
    "sha256": "cdd8229ab2be36a1879dd00f7e0018e387481f0d57be47813d442152c1cb64ff",
    "rationale": "Independent source evaluation and audit"
  },
  {
    "category": "Section 6.3: OneStop internal evaluation",
    "relative_path": ".\\outputs\\onestop_internal_validation_v1\\onestop_internal_validation_audit.json",
    "file_name": "onestop_internal_validation_audit.json",
    "extension": ".json",
    "bytes": 3830,
    "sha256": "119c9693a9a522b5b299100fe915a34a3059849be9b1f51dec68ed2806c486ef",
    "rationale": "Independent source evaluation and audit"
  },
  {
    "category": "Section 6.3: OneStop internal evaluation",
    "relative_path": ".\\src\\models\\run_onestop_internal_validation.py",
    "file_name": "run_onestop_internal_validation.py",
    "extension": ".py",
    "bytes": 27342,
    "sha256": "fdccf0697c944d82a03294d85eb0832a5d81b58afbcc27d989b04eb77ef0f18a",
    "rationale": "Independent source evaluation and audit"
  },
  {
    "category": "Figure 5 / Table 5: QCSR",
    "relative_path": ".\\outputs\\onestop_qcsr_nested_v1\\qcsr_nested_manifest.json",
    "file_name": "qcsr_nested_manifest.json",
    "extension": ".json",
    "bytes": 4261,
    "sha256": "6951c782ec7ca254a09874cb6a0e082fc82825eab4f5f07cf2b0afcdc0657bcf",
    "rationale": "Clean-only/augmentation/QCSR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCSR",
    "relative_path": ".\\outputs\\onestop_qcsr_nested_v1\\qcsr_robust_summary.csv",
    "file_name": "qcsr_robust_summary.csv",
    "extension": ".csv",
    "bytes": 592,
    "sha256": "d14e372dd4ff6ca956a0b32124ef40d00c7e5600534127c2ed20404951d30995",
    "rationale": "Clean-only/augmentation/QCSR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCSR",
    "relative_path": ".\\outputs\\onestop_qcsr_nested_v1\\qcsr_pooled_condition_metrics.csv",
    "file_name": "qcsr_pooled_condition_metrics.csv",
    "extension": ".csv",
    "bytes": 2684,
    "sha256": "a16e9c23352c119e4466a74780d922a06c348be257db9450a1870981f5d2dbda",
    "rationale": "Clean-only/augmentation/QCSR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCSR",
    "relative_path": ".\\outputs\\onestop_qcsr_nested_v1\\qcsr_participant_condition_metrics.csv",
    "file_name": "qcsr_participant_condition_metrics.csv",
    "extension": ".csv",
    "bytes": 2543,
    "sha256": "84463403e9b6e61a536f7bec8ea257dce36ee55201a587eb17cd1accd08f5c78",
    "rationale": "Clean-only/augmentation/QCSR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCSR",
    "relative_path": ".\\outputs\\onestop_qcsr_nested_v1\\qcsr_bootstrap_summary.csv",
    "file_name": "qcsr_bootstrap_summary.csv",
    "extension": ".csv",
    "bytes": 5187,
    "sha256": "032dc219a21642c8d6b133bb576b05d840baa7474b10675bee55ab7d8bf4e159",
    "rationale": "Clean-only/augmentation/QCSR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCSR",
    "relative_path": ".\\outputs\\onestop_qcsr_nested_v1\\qcsr_bootstrap_deltas.csv",
    "file_name": "qcsr_bootstrap_deltas.csv",
    "extension": ".csv",
    "bytes": 4257,
    "sha256": "806460d9ac029884108b5b816eabecf52dfa38c35ee40063066789df82768854",
    "rationale": "Clean-only/augmentation/QCSR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCSR",
    "relative_path": ".\\src\\models\\run_onestop_qcsr_nested.py",
    "file_name": "run_onestop_qcsr_nested.py",
    "extension": ".py",
    "bytes": 59831,
    "sha256": "58af76c2887f61d1305c4c1044345751e1fd1a287427118e3a572b7ec62ea536",
    "rationale": "Clean-only/augmentation/QCSR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCR",
    "relative_path": ".\\outputs\\onestop_qcr_nested_v1\\qcr_nested_manifest.json",
    "file_name": "qcr_nested_manifest.json",
    "extension": ".json",
    "bytes": 4310,
    "sha256": "0bfabf5d7c048a890f258b095451d8fb479b870be81794e8aa0fb80c2a626d54",
    "rationale": "Clean-only/augmentation/QCR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCR",
    "relative_path": ".\\outputs\\onestop_qcr_nested_v1\\qcr_robust_summary.csv",
    "file_name": "qcr_robust_summary.csv",
    "extension": ".csv",
    "bytes": 594,
    "sha256": "206915582b06e7b44ad12da3c3ea29e2b10bf8dfed8ff09ce88a655bc26d0753",
    "rationale": "Clean-only/augmentation/QCR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCR",
    "relative_path": ".\\outputs\\onestop_qcr_nested_v1\\qcr_pooled_condition_metrics.csv",
    "file_name": "qcr_pooled_condition_metrics.csv",
    "extension": ".csv",
    "bytes": 2681,
    "sha256": "6cbce76e772ad50a1e8e3fc9212fb2e03f6a1d27a022811b1066dbac6c2f58e5",
    "rationale": "Clean-only/augmentation/QCR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCR",
    "relative_path": ".\\outputs\\onestop_qcr_nested_v1\\qcr_participant_condition_metrics.csv",
    "file_name": "qcr_participant_condition_metrics.csv",
    "extension": ".csv",
    "bytes": 2549,
    "sha256": "3a18632427f2f891929a654cf3556a2e85d2014b637fa716eb1845ab94988eea",
    "rationale": "Clean-only/augmentation/QCR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCR",
    "relative_path": ".\\outputs\\onestop_qcr_nested_v1\\qcr_bootstrap_summary.csv",
    "file_name": "qcr_bootstrap_summary.csv",
    "extension": ".csv",
    "bytes": 5170,
    "sha256": "5c08bb5fadf18f76c35e486639d7993c96a5659be674077ae47257cba6b56095",
    "rationale": "Clean-only/augmentation/QCR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCR",
    "relative_path": ".\\outputs\\onestop_qcr_nested_v1\\qcr_bootstrap_deltas.csv",
    "file_name": "qcr_bootstrap_deltas.csv",
    "extension": ".csv",
    "bytes": 3703,
    "sha256": "1d7daf0238faed02f84aa472b675f143276543e0ca8a7352c044861d9f983336",
    "rationale": "Clean-only/augmentation/QCR summaries and inference"
  },
  {
    "category": "Figure 5 / Table 5: QCR",
    "relative_path": ".\\src\\models\\run_onestop_qcr_nested.py",
    "file_name": "run_onestop_qcr_nested.py",
    "extension": ".py",
    "bytes": 61500,
    "sha256": "81c5643b013cfd64704e720b6efe04e1eb06315d4426a2191a6624c96e0ca056",
    "rationale": "Clean-only/augmentation/QCR summaries and inference"
  },
  {
    "category": "Table 6: transferred event baseline",
    "relative_path": ".\\outputs\\webqamgaze_external_validation_stage4\\external_pooled_metrics.csv",
    "file_name": "external_pooled_metrics.csv",
    "extension": ".csv",
    "bytes": 1509,
    "sha256": "c8e62e37167d9c35ac46097c8a0947ecab7cf73222680783912813c225e3601c",
    "rationale": "External pooled/participant/bootstrap evidence"
  },
  {
    "category": "Table 6: transferred event baseline",
    "relative_path": ".\\outputs\\webqamgaze_external_validation_stage4\\external_summary.csv",
    "file_name": "external_summary.csv",
    "extension": ".csv",
    "bytes": 2650,
    "sha256": "2cefc4e22ee2ff3ff3f2b6e056e6cb6582346222ac9b86cafe30bf150a1cc295",
    "rationale": "External pooled/participant/bootstrap evidence"
  },
  {
    "category": "Table 6: transferred event baseline",
    "relative_path": ".\\outputs\\webqamgaze_external_validation_stage4\\external_subject_metrics.csv",
    "file_name": "external_subject_metrics.csv",
    "extension": ".csv",
    "bytes": 99726,
    "sha256": "8e134ddf81818358413a16a7e3dcab922b6c7d885a6b33b3cd10d4b199abf172",
    "rationale": "External pooled/participant/bootstrap evidence"
  },
  {
    "category": "Table 6: transferred event baseline",
    "relative_path": ".\\outputs\\webqamgaze_external_validation_stage4\\external_bootstrap_summary.csv",
    "file_name": "external_bootstrap_summary.csv",
    "extension": ".csv",
    "bytes": 2845,
    "sha256": "c8a81f2e4c5e8ebea96298a46b7a0d668713c3f7c3f1ba4c839c3f0c26cf0bb2",
    "rationale": "External pooled/participant/bootstrap evidence"
  },
  {
    "category": "Table 6: transferred event baseline",
    "relative_path": ".\\outputs\\webqamgaze_external_validation_stage4\\compact_pairwise_bootstrap_differences.csv",
    "file_name": "compact_pairwise_bootstrap_differences.csv",
    "extension": ".csv",
    "bytes": 2392,
    "sha256": "4bdd7d599e67a38e45d21173494a360222dfbda1ec7cb9e5af499a48d1f16250",
    "rationale": "External pooled/participant/bootstrap evidence"
  },
  {
    "category": "Table 6: transferred event baseline",
    "relative_path": ".\\src\\models\\run_onestop_to_webqamgaze_zero_shot.py",
    "file_name": "run_onestop_to_webqamgaze_zero_shot.py",
    "extension": ".py",
    "bytes": 31394,
    "sha256": "6d7df26df6069121f6cf4cf04bb330b4514e717673f193a10807109f2ca9c629",
    "rationale": "External pooled/participant/bootstrap evidence"
  },
  {
    "category": "Figure 7 / Table 9: final English selective prediction",
    "relative_path": ".\\outputs\\final_factorized_selective_prediction_v1\\final_selective_prediction_manifest.json",
    "file_name": "final_selective_prediction_manifest.json",
    "extension": ".json",
    "bytes": 1370,
    "sha256": "15beda2bd3987045e7caa180b91f906b5246948aec0a4e1daf6239b87d50b660",
    "rationale": "QCN-PC coverage, AURC/eAURC, bootstrap, implementation"
  },
  {
    "category": "Figure 7 / Table 9: final English selective prediction",
    "relative_path": ".\\outputs\\final_factorized_selective_prediction_v1\\final_selective_prediction_metrics.csv",
    "file_name": "final_selective_prediction_metrics.csv",
    "extension": ".csv",
    "bytes": 26143,
    "sha256": "0b6451dbf3578b27f659384f681bd454f37b01a7eed39665b1b6d9b8241ecb3e",
    "rationale": "QCN-PC coverage, AURC/eAURC, bootstrap, implementation"
  },
  {
    "category": "Figure 7 / Table 9: final English selective prediction",
    "relative_path": ".\\outputs\\final_factorized_selective_prediction_v1\\final_selective_prediction_aurc.csv",
    "file_name": "final_selective_prediction_aurc.csv",
    "extension": ".csv",
    "bytes": 631,
    "sha256": "685902ade2f3fbf8ea6e3dcaeb6c5e0cc739f41c7fe66a24d5db06a811f09b4e",
    "rationale": "QCN-PC coverage, AURC/eAURC, bootstrap, implementation"
  },
  {
    "category": "Figure 7 / Table 9: final English selective prediction",
    "relative_path": ".\\outputs\\final_factorized_selective_prediction_v1\\final_selective_prediction_bootstrap_deltas.csv",
    "file_name": "final_selective_prediction_bootstrap_deltas.csv",
    "extension": ".csv",
    "bytes": 14197,
    "sha256": "e2cb8d6652e4dd6ab560ea627846739fc95593299bf0bf64ac35a130876bac60",
    "rationale": "QCN-PC coverage, AURC/eAURC, bootstrap, implementation"
  },
  {
    "category": "Figure 7 / Table 9: final English selective prediction",
    "relative_path": ".\\src\\models\\run_final_factorized_selective_prediction.py",
    "file_name": "run_final_factorized_selective_prediction.py",
    "extension": ".py",
    "bytes": 22761,
    "sha256": "af2a7f4f14ad1dd4e5184706c067cc24e5e5b27e998d06299e25b68cf0d547c1",
    "rationale": "QCN-PC coverage, AURC/eAURC, bootstrap, implementation"
  },
  {
    "category": "Figure 7: participant-calibration selective reference",
    "relative_path": ".\\outputs\\webqamgaze_selective_prediction_v1\\selective_prediction_manifest.json",
    "file_name": "selective_prediction_manifest.json",
    "extension": ".json",
    "bytes": 1208,
    "sha256": "4f250bce368f2aabeb59f5020f6087e807b88b7b3a252be816213acb41409cb7",
    "rationale": "Reference coverage and AURC/eAURC"
  },
  {
    "category": "Figure 7: participant-calibration selective reference",
    "relative_path": ".\\outputs\\webqamgaze_selective_prediction_v1\\selective_prediction_metrics.csv",
    "file_name": "selective_prediction_metrics.csv",
    "extension": ".csv",
    "bytes": 44990,
    "sha256": "5bde83118aeb9da4c12e7d05187ea0afbaba1fd4dc82e43bc13e8dbbb9e71a64",
    "rationale": "Reference coverage and AURC/eAURC"
  },
  {
    "category": "Figure 7: participant-calibration selective reference",
    "relative_path": ".\\outputs\\webqamgaze_selective_prediction_v1\\selective_prediction_aurc.csv",
    "file_name": "selective_prediction_aurc.csv",
    "extension": ".csv",
    "bytes": 1128,
    "sha256": "457893a4ad96b340f44c8dbb947e9447748fce6aaea98c379bb9292641a4f94f",
    "rationale": "Reference coverage and AURC/eAURC"
  },
  {
    "category": "Figure 7: participant-calibration selective reference",
    "relative_path": ".\\outputs\\webqamgaze_selective_prediction_v1\\selective_prediction_bootstrap_deltas.csv",
    "file_name": "selective_prediction_bootstrap_deltas.csv",
    "extension": ".csv",
    "bytes": 5704,
    "sha256": "8f68ab2574f277c0d2c60052fd34fb33a570ebf263f405d4b9c51e66e02220a6",
    "rationale": "Reference coverage and AURC/eAURC"
  },
  {
    "category": "Figure 7: participant-calibration selective reference",
    "relative_path": ".\\src\\models\\run_webqamgaze_selective_prediction.py",
    "file_name": "run_webqamgaze_selective_prediction.py",
    "extension": ".py",
    "bytes": 23121,
    "sha256": "f43b97a06316c87c9c44ed51188a3126a53032c628bc8235ed8f4ede459e889a",
    "rationale": "Reference coverage and AURC/eAURC"
  },
  {
    "category": "Figure 6: rendered repository source",
    "relative_path": ".\\outputs\\webqamgaze_external_validation_final_release_v1\\figures\\primary_external_validation_metrics.png",
    "file_name": "primary_external_validation_metrics.png",
    "extension": ".png",
    "bytes": 157491,
    "sha256": "0da2cde600b676251b2d5f3353378fd7384935db43c5d2a621cba6ddd0ac7772",
    "rationale": "Existing image retained for image/hash comparison"
  },
  {
    "category": "Figure 6: rendered repository source",
    "relative_path": ".\\outputs\\webqamgaze_external_validation_stage6_thesis\\figure_external_validation_metrics.png",
    "file_name": "figure_external_validation_metrics.png",
    "extension": ".png",
    "bytes": 157491,
    "sha256": "0da2cde600b676251b2d5f3353378fd7384935db43c5d2a621cba6ddd0ac7772",
    "rationale": "Existing image retained for image/hash comparison"
  },
  {
    "category": "Figure 4 / Table 4: consolidated reporting",
    "relative_path": ".\\outputs\\final_reporting\\table03_key_robustness_results.csv",
    "file_name": "table03_key_robustness_results.csv",
    "extension": ".csv",
    "bytes": 6488,
    "sha256": "0344c6230ee19cdba5176744dc71a68ada5f7fe539bc53ef58de9e003e0f46b1",
    "rationale": "Existing consolidated robustness table/report"
  },
  {
    "category": "Figure 4 / Table 4: consolidated reporting",
    "relative_path": ".\\outputs\\final_reporting\\final_results_summary.md",
    "file_name": "final_results_summary.md",
    "extension": ".md",
    "bytes": 1238,
    "sha256": "f8133660aa94de7792b950108e764bdbb2fd367a360f11da080e861ceaecc40d",
    "rationale": "Existing consolidated robustness table/report"
  }
]

TEXT_EXTENSIONS = {".csv", ".json", ".py", ".txt", ".md"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp932", "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect compact P1 provenance evidence and verify inventory hashes."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/iui_p1_provenance_evidence_v2",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    evidence: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for item in SELECTED:
        relative = str(item["relative_path"])
        normalized = relative[2:] if relative.startswith(".\\") else relative
        path = root / Path(normalized)

        row = dict(item)
        row["resolved_path"] = str(path)
        row["exists"] = path.exists()
        row["actual_bytes"] = None
        row["actual_sha256"] = ""
        row["hash_status"] = "MISSING"
        row["content_embedded"] = False
        row["encoding"] = ""

        entry = {
            "category": item["category"],
            "relative_path": relative,
            "rationale": item["rationale"],
            "expected_sha256": item["sha256"],
            "expected_bytes": int(item["bytes"]),
        }

        if not path.exists():
            failures.append(f"MISSING: {relative}")
            evidence.append(entry)
            index_rows.append(row)
            continue

        actual_hash = sha256(path)
        actual_bytes = path.stat().st_size
        row["actual_bytes"] = actual_bytes
        row["actual_sha256"] = actual_hash
        row["hash_status"] = (
            "PASS"
            if actual_hash.lower() == str(item["sha256"]).lower()
            and actual_bytes == int(item["bytes"])
            else "FAIL"
        )

        if row["hash_status"] != "PASS":
            failures.append(
                f"HASH_OR_SIZE_MISMATCH: {relative} "
                f"expected={item['sha256']}/{item['bytes']} "
                f"actual={actual_hash}/{actual_bytes}"
            )

        entry["actual_sha256"] = actual_hash
        entry["actual_bytes"] = actual_bytes
        entry["hash_status"] = row["hash_status"]

        if path.suffix.lower() in TEXT_EXTENSIONS:
            content, encoding = decode_text(path)
            entry["content"] = content
            entry["encoding"] = encoding
            row["content_embedded"] = True
            row["encoding"] = encoding
        else:
            entry["content"] = None
            entry["encoding"] = None

        evidence.append(entry)
        index_rows.append(row)

    bundle = {
        "artifact_version": "iui_p1_provenance_evidence_v2",
        "project_root": str(root),
        "selected_artifacts": len(SELECTED),
        "embedded_text_artifacts": sum(
            1 for item in index_rows if item["content_embedded"]
        ),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
        "evidence": evidence,
    }

    bundle_path = output / "P1_PROVENANCE_EVIDENCE_BUNDLE.json"
    bundle_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    index_path = output / "P1_PROVENANCE_EVIDENCE_INDEX.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)

    summary_lines = [
        "# P1 Provenance Evidence Collection v2",
        "",
        f"- Status: **{bundle['status']}**",
        f"- Selected artifacts: **{len(SELECTED)}**",
        f"- Embedded text artifacts: **{bundle['embedded_text_artifacts']}**",
        f"- Failures: **{len(failures)}**",
        "",
        "## Outputs",
        "",
        "- `P1_PROVENANCE_EVIDENCE_BUNDLE.json`",
        "- `P1_PROVENANCE_EVIDENCE_INDEX.csv`",
        "",
        "## Failures",
        "",
    ]
    if failures:
        summary_lines.extend(f"- {failure}" for failure in failures)
    else:
        summary_lines.append("- None. All selected files matched the inventory hashes and sizes.")

    (output / "P1_PROVENANCE_EVIDENCE_SUMMARY.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Selected artifacts: {len(SELECTED)}")
    print(f"Embedded text artifacts: {bundle['embedded_text_artifacts']}")
    print(f"Status: {bundle['status']}")
    print(f"Bundle: {bundle_path}")
    print(f"Index: {index_path}")
    if failures:
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
