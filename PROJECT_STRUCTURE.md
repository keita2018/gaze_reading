# gaze_reading_goal フォルダ構成

2026年8月1日時点のプロジェクト構成です。

> 注: 仮想環境、外部リポジトリ、実験ログ、生成済み成果物、被験者別データなど、ファイル数が非常に多い領域は内容を省略またはパターン表記しています。

```text
gaze_reading_goal/
├── .venv/                         # プロジェクト用Python仮想環境
├── configs/                       # 設定ファイル配置用（現在は空）
├── data/                          # 入力データと中間・処理済みデータ
│   ├── raw/
│   │   ├── webqamgaze/            # WebQAmGazeの生データ／リポジトリ
│   │   ├── zuco/                  # ZuCo関連の生データ
│   │   └── zuco2/
│   │       └── {subject_id}/
│   │           └── session_et/
│   │               ├── raw/
│   │               └── corrected/ # 一部被験者のみ
│   ├── interim/
│   │   └── zuco2/
│   │       ├── all_subjects/
│   │       └── {subject_id}/
│   └── processed/
│       └── zuco2/
│           ├── soft_aoi_all_subjects/
│           ├── soft_aoi_template/
│           └── {subject_id}/
│               └── soft_aoi/
├── external/                      # 外部プロジェクト／ベンチマーク
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
│   ├── final_reporting/
│   ├── frozen_zuco_external_validation_v1/
│   │   └── models/
│   ├── missingness_30hz/
│   ├── nested_final_common16/
│   ├── noise_augmentation_30hz/
│   ├── predictions/
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
│   ├── soft_aoi_rate_robustness_30_15hz/
│   ├── soft_aoi_rate_robustness_30_15_10hz/
│   ├── soft_aoi_rawstat_nested_loso_30hz/
│   ├── soft_aoi_word_figures/
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
│   └── webqamgaze_repository_audit/
├── src/                           # プロジェクトの実装コード
│   ├── analysis/
│   │   └── generate_zuco_final_reporting.py
│   ├── data/                      # 取得、検査、監査、変換、特徴抽出
│   │   ├── analyze_zuco_et_markers.py
│   │   ├── audit_webqamgaze_repository.py
│   │   ├── audit_zuco_all_subject_raw_et.py
│   │   ├── audit_zuco_raw_et.py
│   │   ├── build_zuco_raw_official_trial_map.py
│   │   ├── compare_zuco_et_files.py
│   │   ├── diagnose_zuco_soft_aoi_distances.py
│   │   ├── download_exact_osf_files.py
│   │   ├── download_zuco_complete_raw_et.py
│   │   ├── download_zuco_subject.py
│   │   ├── download_zuco_subject_all_raw_et.py
│   │   ├── export_zuco_complete_subjects.py
│   │   ├── export_zuco_session_trials.py
│   │   ├── export_zuco_session_trials_v1.py
│   │   ├── export_zuco_subject_all_blocks.py
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
│   │   └── run_zuco_soft_aoi_all_subjects.py
│   ├── evaluation/                # 評価処理配置用（現在は空）
│   ├── features/                  # 特徴量処理配置用（現在は空）
│   ├── models/                    # モデル実験、比較、外部検証
│   │   ├── analyze_webqamgaze_external_failure_modes.py
│   │   ├── audit_zuco_webqamgaze_label_orientation.py
│   │   ├── finalize_webqamgaze_external_validation.py
│   │   ├── freeze_webqamgaze_final_release.py
│   │   ├── freeze_zuco_external_validation_models.py
│   │   ├── generate_webqamgaze_thesis_outputs.py
│   │   ├── run_webqamgaze_*.py
│   │   ├── run_yac_*.py
│   │   └── run_zuco_*.py
│   └── visualization/
│       └── generate_soft_aoi_word_figures.py
├── environment.yml                # Conda環境定義
├── gitignore                      # Git除外設定
└── README.md                      # プロジェクト説明
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
| `external/` | WebQAmGazeおよびZuCoベンチマークの外部資産を保存 |
| `src/data/` | データ取得、監査、検査、変換、特徴抽出を実装 |
| `src/models/` | ベースライン、頑健性評価、比較実験、外部検証を実装 |
| `src/analysis/` | 最終集計とレポート生成を実装 |
| `src/visualization/` | 可視化処理を実装 |
| `outputs/` | 表、図、予測、モデル、監査結果などの生成物を保存 |
| `logs/` | コマンド実行時の標準出力・エラーログを保存 |
