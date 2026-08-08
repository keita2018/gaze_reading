param(
    [string]$ProjectRoot = ".",
    [string]$SourceTable = "data\processed\onestop\quality_views_asc344_v1\onestop_quality_views_all_features.parquet",
    [string]$FrozenModel = "outputs\frozen_onestop_external_validation_v1\models\soft_full_all.joblib",
    [string]$TargetFeatures = "outputs\webqamgaze_external_validation_stage3c\soft_full_features_primary.csv",
    [string]$TargetLabels = "outputs\webqamgaze_external_validation_stage2\trial_manifest_primary_candidates.csv",
    [switch]$AuditOnly,
    [switch]$RunAugmentedTransformer
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

$AuditOutput = "outputs\onestop_model_class_controls_input_audit_v1"
$TabularOutput = "outputs\onestop_tabular_model_controls_v1"
$TransformerOutput = "outputs\onestop_feature_transformer_control_v1"

Write-Host "[1/3] Auditing input tables..." -ForegroundColor Cyan
python src\models\run_onestop_tabular_model_controls.py `
    --project-root . `
    --source-table $SourceTable `
    --frozen-model $FrozenModel `
    --target-features $TargetFeatures `
    --target-labels $TargetLabels `
    --output-dir $AuditOutput `
    --audit-only

if ($LASTEXITCODE -ne 0) {
    throw "Input audit failed. Inspect $AuditOutput\input_candidate_audit.csv"
}

if ($AuditOnly) {
    Write-Host "Input audit completed." -ForegroundColor Green
    exit 0
}

Write-Host "[2/3] Running LinearSVM / HistGradientBoosting / MLP controls..." -ForegroundColor Cyan
python src\models\run_onestop_tabular_model_controls.py `
    --project-root . `
    --source-table $SourceTable `
    --frozen-model $FrozenModel `
    --target-features $TargetFeatures `
    --target-labels $TargetLabels `
    --output-dir $TabularOutput `
    --models linear_svm hist_gradient_boosting mlp `
    --protocols clean_only augmented `
    --outer-folds 5 `
    --inner-folds 3 `
    --bootstrap 2000 `
    --seed 20260806

if ($LASTEXITCODE -ne 0) {
    throw "Tabular model-class controls failed."
}

Write-Host "[3/3] Running same-input Feature Transformer control..." -ForegroundColor Cyan
python src\models\run_onestop_feature_transformer_control.py `
    --project-root . `
    --source-table $SourceTable `
    --frozen-model $FrozenModel `
    --target-features $TargetFeatures `
    --target-labels $TargetLabels `
    --output-dir $TransformerOutput `
    --protocol clean_only `
    --outer-folds 5 `
    --bootstrap 2000 `
    --d-model 64 `
    --n-heads 4 `
    --n-layers 2 `
    --dropout 0.15 `
    --epochs 40 `
    --batch-size 512 `
    --learning-rate 0.0007 `
    --weight-decay 0.0003 `
    --seed 20260806

if ($LASTEXITCODE -ne 0) {
    throw "Feature Transformer control failed. Confirm that PyTorch is installed."
}

if ($RunAugmentedTransformer) {
    Write-Host "Running optional augmented Feature Transformer..." -ForegroundColor Cyan
    python src\models\run_onestop_feature_transformer_control.py `
        --project-root . `
        --source-table $SourceTable `
        --frozen-model $FrozenModel `
        --target-features $TargetFeatures `
        --target-labels $TargetLabels `
        --output-dir "outputs\onestop_feature_transformer_augmented_control_v1" `
        --protocol augmented `
        --outer-folds 5 `
        --bootstrap 2000 `
        --d-model 64 `
        --n-heads 4 `
        --n-layers 2 `
        --dropout 0.15 `
        --epochs 40 `
        --batch-size 512 `
        --learning-rate 0.0007 `
        --weight-decay 0.0003 `
        --seed 20260806

    if ($LASTEXITCODE -ne 0) {
        throw "Augmented Feature Transformer control failed."
    }
}

Write-Host "" 
Write-Host "Model-class controls completed." -ForegroundColor Green
Write-Host "Tabular results:    $TabularOutput"
Write-Host "Transformer results: $TransformerOutput"
