# gaze_reading_goal フォルダ構成

2026年8月3日時点のプロジェクト構成です。

> 注: `.git/`、仮想環境、外部リポジトリ内の詳細、実験ログ、生成成果物内の個別ファイル、被験者別データは、ファイル数が多いため省略またはパターン表記しています。

```text
gaze_reading_goal/
├── .venv/                         # プロジェクト用Python仮想環境
├── configs/                       # 設定ファイル配置用（現在は空）
├── data/                          # 入力・中間・処理済みデータ
│   ├── raw/
│   │   ├── onestop/
│   │   │   ├── asc_all/
│   │   │   ├── asc_samples/
│   │   │   ├── edf_recovery/
│   │   │   ├── information_seeking/
│   │   │   └── ordinary/
│   │   ├── webqamgaze/            # WebQAmGazeの生データ／リポジトリ
│   │   ├── zuco/                  # ZuCo関連の生データ
│   │   └── zuco2/
│   │       └── {subject_id}/
│   │           └── session_et/
│   │               ├── raw/
│   │               └── corrected/ # 一部被験者のみ
│   ├── interim/
│   │   ├── onestop/
│   │   │   ├── raw_samples_30hz_asc344/
│   │   │   └── raw_samples_30hz_sample/
│   │   └── zuco2/
│   │       ├── all_subjects/
│   │       └── {subject_id}/
│   └── processed/
│       ├── onestop/
│       │   ├── quality_views_asc344_v1/
│       │   ├── quality_views_sample/
│       │   ├── soft_aoi_asc344/
│       │   └── soft_aoi_sample/
│       └── zuco2/
│           ├── hard_aoi_all_subjects/
│           ├── soft_aoi_all_subjects/
│           ├── soft_aoi_template/
│           └── {subject_id}/
│               └── soft_aoi/
├── external/                      # 外部プロジェクト／ベンチマーク
│   ├── OneStop-Eye-Movements/
│   │   ├── .github/
│   │   ├── data_preprocessing/
│   │   ├── docs/
│   │   ├── onestop_paper_analyses/
│   │   └── readme_images/
│   ├── WebQAmGaze/
│   │   ├── datasets/
│   │   ├── generate_set/
│   │   ├── HerokuCode/
│   │   ├── instruction_imgs/
│   │   ├── paper_imgs/
│   │   └── scripts/
│   └── zuco-benchmark/
│       ├── .venv/                 # 外部ベンチマーク専用仮想環境
│       ├── matlab/
│       └── src/
├── logs/                          # 各処理・実験の実行ログ
├── notebooks/                     # Notebook配置用（現在は空）
├── outputs/                       # 分析・学習・評価の生成成果物
│   ├── combined_degradation_30hz/
│   ├── common16_comparison/
│   ├── coordinate_noise_30hz/
│   ├── downsampling_rate_invariant/
│   ├── downsampling_robustness/
│   ├── figures/
│   ├── final_common16_comparison/
│   ├── final_factorized_selective_prediction_v1/
│   ├── final_reporting/
│   ├── final_representation_comparison/
│   ├── frozen_onestop_external_validation_v1/
│   │   └── models/
│   ├── frozen_zuco_external_validation_v1/
│   │   └── models/
│   ├── frozen_zuco_external_validation_v1_audit/
│   ├── goal_measurement_participant_factorization_smoke/
│   ├── ivt_event_baseline_fixed/
│   ├── ivt_event_baseline_fixed_benchmark_exact/
│   ├── ivt_event_baseline_nested_threshold/
│   ├── layer3_layout_audit/
│   ├── layer3_layout_residualization_v1/
│   ├── layer4_crossfit_external_audit_v1/
│   ├── layer4_unlabeled_domain_alignment_v1/
│   ├── linear_feature_group_factorization_v1/
│   ├── missingness_30hz/
│   ├── nested_final_common16/
│   ├── noise_augmentation_30hz/
│   ├── onestop_asc_download/
│   ├── onestop_asc_download_retry/
│   ├── onestop_asc_structure/
│   ├── onestop_audit/
│   ├── onestop_internal_validation_v1/
│   ├── onestop_manifest_audit/
│   ├── onestop_missing_trial_l35_264_t53/
│   ├── onestop_qcr_nested_v1/
│   ├── onestop_qcr_smoke/
│   ├── onestop_qcsr_nested_v1/
│   ├── onestop_qcsr_smoke/
│   ├── onestop_quality_views_asc344_v1/
│   ├── onestop_quality_views_sample/
│   ├── onestop_raw_extraction_asc344/
│   ├── onestop_raw_extraction_sample/
│   ├── onestop_raw_inventory/
│   ├── onestop_soft_aoi_asc344/
│   ├── onestop_soft_aoi_sample/
│   ├── onestop_to_webqamgaze_zero_shot_v1/
│   ├── onestop_trial_messages/
│   ├── predictions/
│   ├── quality_conditioned_canonicalization_v1/
│   ├── quality_conditioned_score_canonicalization_v1/
│   ├── raw_baseline/
│   ├── raw_loso/
│   ├── soft_aoi_ablation_30hz/
│   ├── soft_aoi_combined_degradation_30hz/
│   ├── soft_aoi_compact_selection_30hz/
│   ├── soft_aoi_confound_controls_30hz/
│   ├── soft_aoi_coordinate_noise_30hz/
│   ├── soft_aoi_diagnostics/
│   ├── soft_aoi_missingness_30hz/
│   ├── soft_aoi_official_common16_nested_dual/
│   ├── soft_aoi_official_common16_threshold_tuned/
│   ├── soft_aoi_rate_robustness_30_15_10hz/
│   ├── soft_aoi_rate_robustness_30_15hz/
│   ├── soft_aoi_rawstat_nested_loso_30hz/
│   ├── soft_aoi_word_figures/
│   ├── soft_vs_hard_aoi_clean_30hz/
│   ├── soft_vs_hard_aoi_combined_high_30hz/
│   ├── soft_vs_hard_aoi_coordinate_noise_30hz/
│   ├── soft_vs_hard_aoi_missingness_30hz/
│   ├── soft_vs_hard_aoi_train30_test10/
│   ├── static_offset_30hz/
│   ├── tables/
│   ├── webqamgaze_external_validation_final_release_v1/
│   │   ├── audit/
│   │   ├── figures/
│   │   ├── manifests/
│   │   ├── tables/
│   │   └── text/
│   ├── webqamgaze_external_validation_stage1/
│   ├── webqamgaze_external_validation_stage1b/
│   ├── webqamgaze_external_validation_stage1c/
│   ├── webqamgaze_external_validation_stage2/
│   ├── webqamgaze_external_validation_stage3a/
│   ├── webqamgaze_external_validation_stage3b/
│   ├── webqamgaze_external_validation_stage3c/
│   ├── webqamgaze_external_validation_stage4/
│   ├── webqamgaze_external_validation_stage5_diagnostics/
│   ├── webqamgaze_external_validation_stage6_thesis/
│   ├── webqamgaze_external_validation_stage7_all18/
│   ├── webqamgaze_external_validation_stage8_final/
│   ├── webqamgaze_participant_self_calibration_v1/
│   ├── webqamgaze_participant_trial_count_audit/
│   ├── webqamgaze_repository_audit/
│   ├── webqamgaze_selective_prediction_v1/
│   └── zuco_to_onestop_zero_shot_v1/
├── src/                           # プロジェクトの実装コード
│   ├── analysis/
│   │   ├── audit_layer3_layout_shift.py
│   │   ├── audit_layer4_crossfit_external_alignment.py
│   │   ├── generate_zuco_final_reporting.py
│   │   ├── generate_zuco_final_representation_comparison.py
│   │   ├── run_layer3_layout_residualization.py
│   │   └── run_layer4_unlabeled_domain_alignment.py
│   ├── data/                      # 取得、検査、監査、変換、特徴抽出
│   │   ├── analyze_zuco_et_markers.py
│   │   ├── audit_onestop_asc_structure.py
│   │   ├── audit_onestop_raw_inventory.py
│   │   ├── audit_onestop_reports.py
│   │   ├── audit_onestop_trial_messages.py
│   │   ├── audit_webqamgaze_participant_trial_counts.py
│   │   ├── audit_webqamgaze_repository.py
│   │   ├── audit_zuco_all_subject_raw_et.py
│   │   ├── audit_zuco_raw_et.py
│   │   ├── build_onestop_regular_manifest.py
│   │   ├── build_zuco_raw_official_trial_map.py
│   │   ├── compare_zuco_et_files.py
│   │   ├── diagnose_onestop_missing_trial.py
│   │   ├── diagnose_zuco_soft_aoi_distances.py
│   │   ├── download_exact_osf_files.py
│   │   ├── download_onestop_asc_archives.py
│   │   ├── download_zuco_complete_raw_et.py
│   │   ├── download_zuco_subject.py
│   │   ├── download_zuco_subject_all_raw_et.py
│   │   ├── export_zuco_complete_subjects.py
│   │   ├── export_zuco_session_trials.py
│   │   ├── export_zuco_session_trials_v1.py
│   │   ├── export_zuco_subject_all_blocks.py
│   │   ├── extract_onestop_quality_views.py
│   │   ├── extract_onestop_regular_raw_samples.py
│   │   ├── extract_onestop_soft_aoi_features.py
│   │   ├── extract_zuco_soft_aoi_features.py
│   │   ├── extract_zuco_soft_aoi_template.py
│   │   ├── inspect_webqamgaze_official_schema_stage1c.py
│   │   ├── inspect_zuco_et_file.py
│   │   ├── inspect_zuco_mat.py
│   │   ├── inventory_zuco_subject_files.py
│   │   ├── prepare_webqamgaze_external_validation_stage1.py
│   │   ├── prepare_webqamgaze_external_validation_stage2.py
│   │   ├── prepare_webqamgaze_external_validation_stage3a.py
│   │   ├── prepare_webqamgaze_external_validation_stage3c.py
│   │   ├── resolve_webqamgaze_aoi_stage3b.py
│   │   ├── resolve_webqamgaze_storage_stage1b.py
│   │   ├── run_zuco_hard_aoi_all_subjects.py
│   │   └── run_zuco_soft_aoi_all_subjects.py
│   ├── evaluation/                # 評価処理配置用（現在は空）
│   ├── features/                  # 特徴量処理配置用（現在は空）
│   ├── models/                    # モデル実験、比較、外部検証
│   │   ├── analyze_webqamgaze_external_failure_modes.py
│   │   ├── audit_frozen_zuco_models.py
│   │   ├── audit_zuco_webqamgaze_label_orientation.py
│   │   ├── finalize_webqamgaze_external_validation.py
│   │   ├── freeze_onestop_external_models.py
│   │   ├── freeze_webqamgaze_final_release.py
│   │   ├── freeze_zuco_external_validation_models.py
│   │   ├── generate_webqamgaze_thesis_outputs.py
│   │   ├── prepare_zuco_target_soft_aoi_five_repeat_caches.py
│   │   ├── run_final_factorized_selective_prediction.py
│   │   ├── run_goal_measurement_participant_factorization.py
│   │   ├── run_linear_feature_group_factorization.py
│   │   ├── run_onestop_*.py
│   │   ├── run_webqamgaze_*.py
│   │   ├── run_yac_*.py
│   │   └── run_zuco_*.py
│   └── visualization/
│       └── generate_soft_aoi_word_figures.py
├── .gitignore                     # Git除外設定
├── environment.yml                # Conda環境定義
├── event_baseline_context.zip     # イベントベースライン関連アーカイブ
├── PROJECT_STRUCTURE.md           # 本ファイル
├── README.md                      # プロジェクト説明
└── run_final_five_repeat_pipeline.ps1
                                     # 最終5反復パイプライン実行スクリプト
```

## 被験者ID

`{subject_id}` は、主に次のZuCo 2.0被験者ディレクトリを表します。

```text
YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS,
YLS, YMD, YMS, YRH, YRK, YRP, YSD, YSL, YTL
```

## 各領域の役割

| パス | 役割 |
| --- | --- |
| `data/raw/` | 取得した元データを保存 |
| `data/interim/` | 変換途中の中間データを保存 |
| `data/processed/` | モデル入力などに使用する処理済みデータを保存 |
| `external/` | OneStop、WebQAmGaze、ZuCoベンチマークの外部資産を保存 |
| `src/data/` | データ取得、監査、検査、変換、特徴抽出を実装 |
| `src/models/` | ベースライン、頑健性評価、比較実験、外部検証を実装 |
| `src/analysis/` | 最終集計、表現比較、ドメイン差監査と補正を実装 |
| `src/visualization/` | 可視化処理を実装 |
| `outputs/` | 表、図、予測、モデル、監査結果などの生成物を保存 |
| `logs/` | コマンド実行時の標準出力・エラーログを保存 |
