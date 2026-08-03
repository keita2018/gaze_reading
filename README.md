# gaze_reading_goal

**連続視線から読書目的を推定するための、Event-Free AOI 集約・ラベルなし参加者較正・選択的予測の研究リポジトリ**

本プロジェクトは、注視（fixation）やサッカード（saccade）を事前に明示検出せず、連続視線サンプルを文章中の単語領域へ直接対応付けて試行単位で集約する **Event-Free AOI Aggregation** を実装・評価するものです。

ZuCo 2.0、OneStop、WebQAmGaze を用いて、以下を段階的に検証しています。

- 未知参加者・未知文章への一般化
- サンプリング低下、座標ノイズ、欠損への頑健性
- Soft-AOI、Hard-AOI、I-VT 系イベント特徴の比較
- 別データセットでの独立評価と品質ビュー拡張
- 実験室視線から Web カメラ視線への凍結転送
- ラベルを用いない参加者自己較正
- Quality-Contribution Neutralization with Participant Calibration（QCN-PC）
- 低信頼試行を保留する選択的予測
- 事前固定したトルコ語ターゲットによる確認評価
- モデル・数値・図表・生成元ファイルの provenance 監査

> **プロジェクト状態**: 2026-08-03 時点  
> **研究段階**: P0（確認評価）・P1（論文整合性／provenance 監査）・P2（匿名提出パッケージ作成）まで完了  
> **主な実行環境**: Windows / PowerShell / Python  
> **詳細なフォルダ構成**: [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)

---

## 1. 研究の背景

視線を用いた読書目的推定では、通常、連続座標を注視とサッカードへ分割し、そのイベント列から注視時間、注視回数、回帰、サッカード振幅などを算出します。

しかし、イベント検出は中立的な前処理ではありません。結果は次の要因に依存します。

- サンプリングレート
- 座標ノイズ
- 欠損
- キャリブレーションずれ
- 検出アルゴリズム
- 速度・分散・継続時間の閾値
- 使用機器と画面条件

特に Web カメラ視線では、座標の揺れ、間欠的な欠損、参加者ごとのオフセットが大きく、イベント境界に依存する特徴が不安定になりやすいという課題があります。

本研究では、イベントを確定する前に視線を単語領域へ直接集約し、**読書行動の分布**と**計測品質**を同じ特徴空間で表現します。

---

## 2. 研究課題

本研究では、外部評価を一つの数値だけで扱わず、次の四つの問いへ分解します。

### RQ1：内部一般化と計測頑健性

Event-Free AOI 表現は、未知参加者と未知文章の両方に一般化し、低サンプリング、座標ノイズ、欠損下でも判別構造を保持できるか。

### RQ2：独立評価と頑健学習

同じ特徴スキーマは独立収集された OneStop でも読書目的に関する情報を保持するか。また、どの品質頑健化手法が clean 性能を維持しつつ劣化性能を改善するか。

### RQ3：ラベルなし外部適応と確認

OneStop で凍結したモデルを WebQAmGaze へ target ラベルなしで移送できるか。参加者自己較正と QCN-PC はどの転送誤差を補正し、その効果は事前固定した未使用言語でも確認されるか。

### RQ4：選択的運用

推定マージンを信頼度として用い、低信頼試行ではインタフェースが適応を保留できるか。

---

## 3. 主な提案・実装

### 3.1 Event-Free Soft-AOI Aggregation

時刻 `t` の視線座標を `g_t`、単語 `j` の矩形領域を `A_j` とします。視線座標から矩形までの距離に基づき、各単語への非正規化所属度を計算します。

```text
q_j(t) = exp( - d(g_t, A_j)^2 / (2σ^2) )
```

帯域幅は表示文字サイズに合わせます。

```text
σ = (2/3) × 試行内の単語高さ中央値
```

`max_j q_j(t) >= 0.1` を満たすサンプルを active とし、active サンプルでは所属度を正規化します。

```text
p_j(t) = q_j(t) / Σ_k q_k(t)
```

この処理では、注視・サッカード境界を先に確定しません。試行全体について、次の情報を集約します。

- 単語・行への視線占有分布
- 文章中の進行位置
- 前進・後退・停留の遷移傾向
- 単語間ジャンプと改行
- 有効サンプル率
- active 率
- 単語矩形への hit 率
- 最大所属度
- 最近傍単語までの距離

### 3.2 Full 41 / Compact 27 / Soft–Hard common 25

| 表現 | 特徴数 | 構成 |
| --- | ---: | --- |
| Full Soft-AOI | 41 | 分布 15 + 時間 6 + 順序 8 + membership／品質 12 |
| Compact Soft-AOI | 27 | 分布 15 + membership／品質 12 |
| Soft–Hard common | 25 | 共通分布 13 + membership／品質 12 |

Compact 27 は ZuCo の ablation を踏まえて定義した探索的表現です。最終的な OneStop → WebQAmGaze 転送では、OneStop の OOF 性能と WebQAmGaze での rank transfer が良好だった Full 41 を主モデルとしています。

### 3.3 Hard-AOI 対照

Hard-AOI は各 active サンプルを最大所属度の単語一つへ割り当てます。

```text
j* = argmax_j q_j(t)
```

Soft と Hard で、入力サンプル、単語矩形、active 閾値、分割、分類器を揃えることで、Soft membership 自体の効果と、イベントを用いない試行集約の効果を分離します。

### 3.4 参加者自己較正

凍結モデルの試行スコアを `s_pi`、source 側で固定した基準値を `r_full` とします。対象試行自身を除外した同一参加者内中央値を用いて、参加者ごとのスコア位置を合わせます。

```text
s'_pi = s_pi - median_{j≠i}(s_pj) + r_full
```

- target ラベルは使用しません。
- 評価対象試行は自分自身の中央値計算から除外します。
- 参加者内の試行順位は変えず、参加者固有のスコア位置だけを補正します。
- 複数のラベルなし試行が必要であり、単一試行だけの新規参加者には適用できません。

### 3.5 QCN-PC

**Quality-Contribution Neutralization with Participant Calibration** は、凍結線形 SVM の決定スコアを、事前定義した特徴群の寄与へ厳密に分解します。

```text
s(x) = b + s_content(x) + s_quality(x)
```

Full 41 のうち、先頭 29 特徴を content／distribution／temporal／sequential 群、末尾 12 特徴を membership／quality 群として固定しています。

Quality-Contribution Neutralization（QCN）は、target の品質関連寄与を source 基準へ置き換えます。

```text
s_QCN(x) = s(x) - s_quality(x) + r_quality
```

その後、QCN スコアに参加者自己較正を適用します。

```text
s_QCN-PC,pi
  = s_QCN,pi
  - median_{j≠i}(s_QCN,pj)
  + r_QCN
```

重要な解釈は次のとおりです。

- 参加者自己較正は参加者固有のスコア位置を補正します。
- QCN は事前定義した品質関連寄与を除き、試行順位を変える可能性があります。
- 12 特徴は「純粋なセンサー品質」を因果的に表すと仮定していません。
- QCN の追加効果はデータセット依存であり、常に参加者較正を上回るとは主張しません。

### 3.6 選択的予測

QCN-PC 後のスコア `s_i` と凍結閾値 `τ` の距離を信頼度とします。

```text
confidence_i = |s_i - τ|
```

- **Global selection**: 全試行を一括して信頼度順に保持
- **Participant-balanced selection**: 各参加者内で同じ割合を保持

低信頼試行では、インタフェースが次のように振る舞うことを想定しています。

- 適応しない
- 追加の視線観測を待つ
- 明示操作を求める
- 現状表示を維持する

---

## 4. 使用データセット

| データセット／部分集合 | 参加者 | 試行 | 主な用途 |
| --- | ---: | ---: | --- |
| ZuCo 2.0 all raw | 18 | 13,302 | 内部一般化、人工劣化、Soft／Hard、I-VT 比較 |
| ZuCo Common16 | 16 | 11,324 | 公式 fixation／saccade 特徴との比較 |
| OneStop available ASC | 344 | 18,573 | 独立評価、頑健 source 学習、凍結モデル |
| OneStop quality views | 344 | 130,011 行 | clean + 6 劣化条件 |
| WebQAmGaze English | 131 participant IDs | 1,349 | 開発ターゲット、較正、QCN-PC、選択的予測 |
| WebQAmGaze Turkish | 105 | 1,050 | 事前固定した言語確認ターゲット |

ラベル方向は全実装で次に統一しています。

```text
NR = 1
TSR / IS = 0
```

ZuCo の TSR、OneStop の IS、WebQAmGaze の IS は関連する目標指向読書ですが、同一の実験課題ではありません。外部評価は、センサー、文章、参加者、課題、言語が同時に変化する related-task cross-dataset evaluation として解釈します。

### 想定配置

```text
data/raw/zuco2/
data/raw/onestop/
data/raw/webqamgaze/

external/OneStop-Eye-Movements/
external/WebQAmGaze/
external/zuco-benchmark/
```

生データと外部リポジトリは Git 管理対象外です。各配布元の利用規約、ライセンス、倫理条件、再配布条件に従って取得してください。

---

## 5. 主な結果

### 5.1 ZuCo Common16：未知参加者・未知文章の二重ホールドアウト

| 表現 | Macro F1 | Balanced Accuracy | ROC-AUC | PR-AUC (NR) |
| --- | ---: | ---: | ---: | ---: |
| Compact Event-Free AOI | 0.620 | 0.642 | 0.717 | 0.655 |
| Full Event-Free AOI | 0.618 | 0.645 | 0.723 | 0.662 |
| 公式 `sent_gaze_sacc` | 0.569 | 0.614 | 0.698 | 0.664 |
| RawStat | 0.530 | 0.570 | 0.637 | 0.602 |
| Dummy majority | 0.344 | 0.500 | 0.500 | 0.476 |

Compact は RawStat を有意に上回りました。一方、公式イベント特徴との差は多重比較補正後に有意ではありません。したがって、クリーンな実験室条件で Event-Free AOI が公式イベント特徴より常に優れるとは主張しません。

### 5.2 品質劣化：Soft／Hard AOI と I-VT

| 条件 | Soft-AOI Macro F1 / AUC | Hard-AOI Macro F1 / AUC | Fixed I-VT Macro F1 / AUC | Quality-tuned I-VT Macro F1 / AUC |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.626 / 0.723 | 0.625 / 0.720 | 0.554 / 0.640 | 0.554 / 0.640 |
| 10 Hz | 0.614 / 0.723 | 0.612 / 0.720 | 0.518 / 0.594 | 0.538 / 0.578 |
| 50 px noise | 0.486 / 0.700 | 0.482 / 0.701 | 0.365 / 0.504 | 0.550 / 0.574 |
| 40% random missing | 0.599 / 0.719 | 0.605 / 0.715 | 0.527 / 0.596 | 0.532 / 0.569 |
| 40% burst missing | 0.600 / 0.670 | 0.601 / 0.671 | 0.546 / 0.608 | 0.545 / 0.608 |
| Noise + random | 0.448 / 0.702 | 0.460 / 0.702 | 0.347 / 0.502 | 0.504 / 0.566 |
| Noise + burst | 0.474 / 0.679 | 0.481 / 0.681 | 0.353 / 0.502 | 0.528 / 0.552 |

50 px ノイズでは、固定 I-VT の ROC-AUC が 0.504 まで低下する一方、Soft-AOI は 0.700 を維持しました。

ただし、Soft と Hard の差は条件依存です。頑健性の主因は Soft membership だけではなく、イベント境界に依存しない試行集約と、局所的なイベント数への依存低減にあると解釈しています。

### 5.3 OneStop：品質ビューによる頑健学習

| 学習方法 | Clean Macro F1 | 劣化平均 Macro F1 | Clean からの低下 |
| --- | ---: | ---: | ---: |
| Clean-only | 0.6392 | 0.4583 | -0.1809 |
| Augmentation | 0.6363 | 0.6168 | -0.0194 |
| QCSR | 0.6340 | 0.6172 | -0.0168 |
| QCR | 0.6362 | 0.6169 | -0.0193 |

clean と六つの劣化 view を用いる通常の augmentation が、clean 性能をほぼ維持しながら劣化平均 Macro F1 を約 0.1585 改善しました。

QCSR と QCR は、通常の augmentation に対して実質的な追加改善を示しませんでした。

### 5.4 OneStop → WebQAmGaze English：開発ターゲット

| 方法 | Macro F1 | Balanced Accuracy | ROC-AUC | PR-AUC (NR) | NR 予測率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen Full 41 | 0.5339 | 0.5442 | 0.5623 | 0.5459 | 0.351 |
| Participant median LOO | 0.5767 | 0.5767 | 0.6024 | 0.5776 | 0.511 |
| QCN only | 0.4738 | 0.5295 | 0.5913 | 0.5771 | 0.175 |
| QCN-PC | 0.6034 | 0.6034 | 0.6497 | 0.6060 | 0.506 |

英語は QCN-PC 開発に使用したため、独立確認ではなく開発データ上の証拠です。

QCN-PC は Participant median LOO に対して ROC-AUC を約 0.0473 改善しましたが、100 個のランダム 12 特徴群との対照では 87 パーセンタイルでした。したがって、事前定義した 12 特徴群を有用な操作的分割として扱い、唯一最適または因果的に純粋な品質群とは主張しません。

### 5.5 WebQAmGaze Turkish：事前固定した確認評価

| 方法 | Macro F1 | Balanced Accuracy | ROC-AUC | PR-AUC (NR) | NR 予測率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen Full 41 | 0.5884 | 0.5933 | 0.6451 | 0.6365 | 0.390 |
| Participant median LOO | 0.6351 | 0.6352 | 0.7119 | 0.6911 | 0.518 |
| QCN only | 0.5134 | 0.5648 | 0.6774 | 0.6536 | 0.175 |
| QCN-PC | 0.6343 | 0.6343 | 0.7190 | 0.7044 | 0.506 |

確認前に以下を固定しました。

- OneStop Full-41 モデル
- `C = 0.01`
- 決定閾値 `0.0107692191181`
- 41 特徴の順序
- 29／12 特徴群分割
- source reference
- QCN → 参加者較正の順序
- 対象言語選択規則
- 主比較
- bootstrap 2,000 回
- permutation 2,000 回
- seed `20260803`

トルコ語での主比較は、QCN-PC と Participant median LOO の pooled ROC-AUC 差です。

```text
AUC 差 = +0.0071
95% participant-cluster bootstrap CI = [-0.0216, 0.0346]
Macro F1 差 = -0.0008
within-participant permutation p = 0.0005
```

判定は **PARTIAL_CONFIRMATION（部分的確認）** です。

- QCN-PC は凍結モデルを明確に上回りました。
- QCN-PC 自体は chance を超えていました。
- しかし、参加者中央値較正に対する追加優位性は統計的に分離されませんでした。

したがって、二つの Web カメラターゲットを通じて最も再現的な補正は **参加者自己較正** であり、QCN は残余の rank distortion が大きい場合に有効な、データセット依存の追加補正と解釈します。

### 5.6 選択的予測

#### English：Global selection

| Coverage | Macro F1 | ROC-AUC | 保持試行 | 保持参加者 |
| ---: | ---: | ---: | ---: | ---: |
| 100% | 0.603 | 0.650 | 1,349 | 131 |
| 50% | 0.677 | 0.702 | 675 | 128 |
| 20% | 0.719 | 0.720 | 270 | 95 |

#### Turkish：Participant-balanced selection

| Coverage | Macro F1 | ROC-AUC | 保持試行 | 保持参加者 |
| ---: | ---: | ---: | ---: | ---: |
| 100% | 0.634 | 0.719 | 1,050 | 105 |
| 50% | 0.732 | 0.807 | 525 | 105 |
| 20% | 0.746 | 0.847 | 210 | 105 |

Participant-balanced 50% は、105 名全員を保持しながら Macro F1 を 0.634 から 0.732、ROC-AUC を 0.719 から 0.807 へ改善しました。

Coverage は、別データで運用最適化された値ではなく、記述的な deployment diagnostic です。

---

## 6. プロジェクトの現在状態

### P0：確認実験

完了済みです。

- 確認プロトコルと source モデルの固定
- 英語開発データでの標準ラベルなし対照
- 未使用言語の label-blind 選択
- トルコ語の unlabeled features／scores の凍結
- SHA-256 監査と独立保存
- sealed labels 開封後の事前規定評価
- `PARTIAL_CONFIRMATION` 判定の固定

主な出力:

```text
outputs/iui_p0_protocol_lock_v1/
outputs/iui_p0_standard_controls_v1/
outputs/iui_p0_confirmatory_TR_v1/
outputs/iui_p0_confirmatory_TR_evaluation_v1/
outputs/iui_p0_tr_backup_verification_v1/
```

### P1：論文整合性・provenance

完了済みです。

- 数値、主張、方法、図表の整合性監査
- 全参考文献の確認
- IUI 想定模擬査読
- 90 個の集計成果物・source script のサイズ／SHA-256 検証
- 本文表 160 数値セルの生成元照合
- Figures 4–7 の生成元 CSV からの再構築
- 本文と匿名補足資料の crosswalk

主な出力:

```text
outputs/iui_p1_provenance_candidate_inventory_v1/
outputs/iui_p1_provenance_evidence_v2/
outputs/iui_p1_rebuilt_figures_v1/
```

### P2：匿名提出パッケージ

作成済みです。

- 匿名本文
- 匿名補足資料
- PCS 入力用シート
- ethics reviewer note テンプレート
- 提出前チェックリスト
- 匿名性スキャン
- package manifest
- 匿名成果物パッケージ作成スクリプト

所属機関における公開データ二次解析の倫理審査／非該当判断と、PCS 上の著者情報・conflict 入力は別途確認が必要です。

---

## 7. リポジトリ構成

```text
gaze_reading_goal/
├── configs/              # 確認評価・実験設定
├── data/                 # raw / interim / processed（Git対象外）
├── docs/                 # プロトコル、チェックリスト、倫理文書
├── external/             # 外部リポジトリ（Git対象外）
├── logs/                 # 実行ログ（Git対象外）
├── notebooks/            # 必要なNotebook
├── outputs/              # モデル、予測、表、図、監査結果（Git対象外）
├── scripts/              # PowerShell実行・凍結・検証スクリプト
├── src/
│   ├── analysis/         # 監査、補正、確認評価、provenance
│   ├── data/             # 取得、監査、変換、特徴抽出
│   ├── evaluation/       # 評価処理配置用
│   ├── features/         # 特徴処理配置用
│   ├── models/           # 学習、比較、外部検証
│   └── visualization/    # 可視化
├── submission/           # ローカル提出物配置用（Git対象外）
├── .gitignore
├── environment.yml
├── PROJECT_STRUCTURE.md
├── README.md
└── run_final_five_repeat_pipeline.ps1
```

完全な構成は [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) を参照してください。

---

## 8. 環境構築

### 8.1 プロジェクト用 `.venv`

Windows PowerShell の例です。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

依存関係の正本は `environment.yml` です。

### 8.2 Conda

```powershell
conda env create -f environment.yml
conda activate <environment-name>
```

既存環境を更新する場合:

```powershell
conda env update -f environment.yml --prune
```

### 8.3 ZuCo 公式ベンチマーク用環境

`external/zuco-benchmark` は古い Python と依存関係を必要とするため、プロジェクト本体とは分離した専用環境を使用します。

```text
external/zuco-benchmark/.venv/
```

外部ベンチマーク環境をプロジェクト本体の `.venv` と混在させないでください。

---

## 9. 再現ワークフロー

本プロジェクトには、探索的実験、最終評価、確認評価が含まれます。すべてを一括実行する前に、各スクリプトの入力、既存キャッシュ、出力先を確認してください。

### 9.1 データ取得・監査

```text
src/data/download_zuco_*.py
src/data/download_onestop_asc_archives.py
src/data/audit_zuco_*.py
src/data/audit_onestop_*.py
src/data/audit_webqamgaze_*.py
```

### 9.2 生視線の抽出・共通化

```text
src/data/export_zuco_*.py
src/data/extract_onestop_regular_raw_samples.py
src/data/prepare_webqamgaze_external_validation_stage*.py
```

### 9.3 Event-Free AOI 特徴抽出

```text
src/data/extract_zuco_soft_aoi_features.py
src/data/run_zuco_soft_aoi_all_subjects.py
src/data/run_zuco_hard_aoi_all_subjects.py
src/data/extract_onestop_soft_aoi_features.py
```

### 9.4 ZuCo 内部評価・頑健性評価

```text
src/models/run_zuco_*.py
outputs/soft_aoi_official_common16_nested_dual/
outputs/soft_vs_hard_aoi_*/
outputs/ivt_event_baseline_fixed_benchmark_exact/
outputs/ivt_event_baseline_nested_threshold/
```

### 9.5 OneStop 独立評価・頑健学習

```text
src/data/extract_onestop_quality_views.py
src/models/run_onestop_*.py
src/models/freeze_onestop_external_models.py
```

主な出力:

```text
outputs/onestop_internal_validation_v1/
outputs/onestop_quality_views_asc344_v1/
outputs/onestop_qcsr_nested_v1/
outputs/onestop_qcr_nested_v1/
outputs/frozen_onestop_external_validation_v1/
```

### 9.6 WebQAmGaze 英語開発評価

```text
src/models/run_webqamgaze_*.py
src/models/run_linear_feature_group_factorization.py
src/models/run_final_factorized_selective_prediction.py
src/models/freeze_webqamgaze_final_release.py
```

### 9.7 トルコ語確認評価

確認対象の features／scores は label を結合する前に固定され、独立保存コピーとの一致を確認してから評価されました。

主な実装:

```text
src/analysis/audit_iui_p0_unused_languages.py
src/analysis/create_iui_p0_tr_stage2_clone.py
src/analysis/audit_iui_p0_tr_stage2_outputs.py
src/analysis/audit_iui_p0_tr_stage3_scripts.py
src/analysis/freeze_iui_p0_tr_unlabeled_artifacts_v2.py
src/analysis/verify_iui_p0_tr_unlabeled_freeze.py
src/analysis/verify_iui_p0_tr_backup_copy.py
src/analysis/evaluate_iui_p0_tr_confirmatory.py
```

実行ラッパー:

```text
scripts/run_iui_p0_tr_stage2.ps1
scripts/run_iui_p0_tr_stage3_and_freeze.ps1
scripts/run_iui_p0_tr_freeze_only_v2.ps1
scripts/run_iui_p0_tr_confirmatory_evaluation.ps1
```

> **注意**  
> `run_iui_p0_tr_confirmatory_evaluation.ps1` は sealed label を読み込みます。確認評価は既に完了しているため、通常の開発実験として反復実行したり、結果を見て設定を変更したりしないでください。再実行する場合は監査目的に限定し、固定済み設定を変更しないでください。

### 9.8 P1 provenance 監査

```text
scripts/collect_iui_p1_provenance_candidates.ps1
src/analysis/collect_iui_p1_provenance_evidence_v2.py
src/analysis/rebuild_iui_p1_figures_4_7.py
```

### 9.9 最終5反復パイプライン

```powershell
powershell -ExecutionPolicy Bypass -File .\run_final_five_repeat_pipeline.ps1
```

このスクリプトは大規模な既存データ・中間成果物・キャッシュを前提とします。内部で参照されるパスと環境を確認してから実行してください。

---

## 10. 主要出力

| パス | 内容 |
| --- | --- |
| `outputs/soft_aoi_official_common16_nested_dual/` | ZuCo Common16 二重ホールドアウト |
| `outputs/soft_vs_hard_aoi_*/` | Soft／Hard 比較 |
| `outputs/ivt_event_baseline_fixed_benchmark_exact/` | 固定 I-VT |
| `outputs/ivt_event_baseline_nested_threshold/` | 品質別 tuned I-VT |
| `outputs/onestop_internal_validation_v1/` | OneStop 独立評価 |
| `outputs/onestop_qcsr_nested_v1/` | QCSR |
| `outputs/onestop_qcr_nested_v1/` | QCR |
| `outputs/frozen_onestop_external_validation_v1/` | 凍結 OneStop source モデル |
| `outputs/iui_p0_standard_controls_v1/` | 英語標準対照 |
| `outputs/iui_p0_confirmatory_TR_v1/` | トルコ語 unlabeled freeze と sealed labels |
| `outputs/iui_p0_confirmatory_TR_evaluation_v1/` | トルコ語確認評価 |
| `outputs/final_factorized_selective_prediction_v1/` | 英語 QCN-PC・選択的予測 |
| `outputs/iui_p1_provenance_evidence_v2/` | provenance evidence bundle |
| `outputs/iui_p1_rebuilt_figures_v1/` | Figures 4–7 再構築 |
| `outputs/webqamgaze_external_validation_final_release_v1/` | 最終外部評価リリース |
| `outputs/final_reporting/` | 論文・報告用集計 |
| `outputs/figures/` | 図 |
| `outputs/tables/` | 表 |
| `logs/` | 実行ログ |

`outputs/` は生成物を含むため、原則として Git 管理対象外です。

---

## 11. 再現性・監査上の注意

- ZuCo 主評価では、テスト参加者とテスト文章を同時に除外します。
- 確率的劣化は複数 seed で反復し、参加者内平均後に対応比較します。
- OneStop は参加者 ID を outer／inner split の group として使用します。
- 同一試行の clean／degraded view が train／test をまたがないようにします。
- WebQAmGaze の bootstrap は participant ID 単位です。
- target ラベルは参加者自己較正、QCN-PC、confidence ranking に使用しません。
- PR-AUC（台形積分）と Average Precision を区別します。
- 相対性能差と chance を超えた絶対識別を別の主張として扱います。
- English WebQAmGaze は開発データです。
- Turkish WebQAmGaze は事前固定した確認データですが、同一 WebQAmGaze プラットフォーム内の言語確認であり、完全に独立した第4データセットではありません。
- 確認結果は `PARTIAL_CONFIRMATION` のまま固定し、事後的に再最適化しません。
- 保存済み model、feature order、class order、threshold、score references、manifests、SHA-256 を監査対象とします。

---

## 12. 既知の制約

- ZuCo TSR、OneStop IS、WebQAmGaze IS は関連するが同一タスクではありません。
- OneStop では読書レジームと参加者集団が交絡しています。
- 人工劣化は実際の Web カメラ誤差を完全には再現しません。
- Compact 27 は ZuCo ablation 後に選択された探索的表現です。
- 参加者自己較正には複数のラベルなし試行が必要です。
- QCN-PC は線形分類器と事前定義した 29／12 分割に依存します。
- membership／quality 特徴には行動差も含まれる可能性があります。
- 英語で開発した QCN-PC の追加効果は、トルコ語では参加者自己較正に対して強く確認されませんでした。
- 選択的予測は Coverage 低下時に保持参加者・クラス構成を変える可能性があります。
- 本研究はユーザモデリング層を評価しており、完成した読書支援インタフェースのユーザ実験は実施していません。

---

## 13. データ・プライバシー

本研究は公開済み・匿名化済みデータセットの二次解析です。各データセットの利用規約、倫理手続き、ライセンス、再配布条件に従ってください。

Git または公開成果物へ次を含めないでください。

- 生視線座標
- 直接的な参加者識別子
- participant-level の未匿名予測
- sealed label ファイル
- ローカル絶対パス
- アカウント名
- API key、token、credentials
- 所属機関を特定する confidential reviewer 情報
- 再配布が許可されていない外部データ・コード

匿名補足資料では、集計特徴、モデル設定、schema、相対パス manifest、集計評価、figure source table を使用し、生視線と直接参加者 ID は除外します。

---

## 14. Git 管理方針

追跡対象:

```text
src/
scripts/
configs/
docs/
notebooks/（必要なもののみ）
README.md
PROJECT_STRUCTURE.md
environment.yml
run_final_five_repeat_pipeline.ps1
```

原則として追跡しないもの:

```text
data/
external/
logs/
outputs/
submission/
仮想環境
学習済みモデル
生データ／中間データ
試行別・参加者別予測
ローカル提出物
```

軽量な公開用集計を Git 管理する場合は、個人情報・絶対パス・外部ライセンスを確認した上で、`docs/results/` などの明示的な公開用ディレクトリへコピーしてください。

---

## 15. 論文

本リポジトリは、次の投稿準備中研究を支える実装・監査・成果物を含みます。

> **Reliable Reading-Goal Modeling from Noisy Gaze:  
> Event-Free AOI Aggregation, Label-Free Calibration, and Selective Prediction**

正式な書誌情報は採択・公開後に更新してください。

```bibtex
@inproceedings{TODO,
  title     = {Reliable Reading-Goal Modeling from Noisy Gaze:
               Event-Free AOI Aggregation, Label-Free Calibration,
               and Selective Prediction},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO}
}
```

---

## 16. ライセンス

プロジェクト本体のライセンスは未確定です。公開前に `LICENSE` を追加し、以下との整合を確認してください。

- ZuCo 2.0 の利用条件
- OneStop の利用条件
- WebQAmGaze の利用条件
- `external/` 配下の各リポジトリのライセンス
- 公式ベンチマーク付属コードのライセンス
- 学習済みモデル・派生成果物の再配布条件

---

## 17. 連絡先

匿名査読・公開前のため、著者情報は記載していません。公開時に所属、連絡先、リポジトリ URL を追加してください。
