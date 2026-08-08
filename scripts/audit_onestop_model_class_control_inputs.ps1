param(
    [string]$ProjectRoot = ".",
    [string]$SourceTable = "data\processed\onestop\quality_views_asc344_v1\onestop_quality_views_all_features.parquet",
    [string]$FrozenModel = "outputs\frozen_onestop_external_validation_v1\models\soft_full_all.joblib",
    [string]$TargetFeatures = "outputs\webqamgaze_external_validation_stage3c\soft_full_features_primary.csv",
    [string]$TargetLabels = "outputs\webqamgaze_external_validation_stage2\trial_manifest_primary_candidates.csv"
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

python src\models\run_onestop_tabular_model_controls.py `
    --project-root . `
    --source-table $SourceTable `
    --frozen-model $FrozenModel `
    --target-features $TargetFeatures `
    --target-labels $TargetLabels `
    --output-dir "outputs\onestop_model_class_controls_input_audit_v1" `
    --audit-only

if ($LASTEXITCODE -ne 0) {
    throw "Input audit failed."
}

Write-Host "Input audit completed." -ForegroundColor Green
Write-Host "outputs\onestop_model_class_controls_input_audit_v1\input_candidate_audit.csv"
