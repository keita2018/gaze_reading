param(
    [string]$ProjectRoot = ".",
    [string]$Stage2Dir = "outputs\webqamgaze_tr_confirmatory_stage2_v1",
    [string]$Stage3ADir = "outputs\webqamgaze_tr_confirmatory_stage3a_v1",
    [string]$Stage3CDir = "outputs\webqamgaze_tr_confirmatory_stage3c_v1",
    [string]$FreezeDir = "outputs\iui_p0_confirmatory_TR_v1"
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

python src\analysis\freeze_iui_p0_tr_unlabeled_artifacts_v2.py `
    --project-root . `
    --stage2-dir $Stage2Dir `
    --stage3a-dir $Stage3ADir `
    --stage3c-dir $Stage3CDir `
    --frozen-onestop-model outputs\frozen_onestop_external_validation_v1\models\soft_full_all.joblib `
    --factor-manifest outputs\linear_feature_group_factorization_v1\linear_feature_group_factorization_manifest.json `
    --protocol-snapshot outputs\iui_p0_protocol_lock_v1 `
    --output-dir $FreezeDir `
    --maximum-feature-failure-rate 0.20

if ($LASTEXITCODE -ne 0) {
    throw "Turkish unlabeled feature/score freeze failed."
}

python src\analysis\verify_iui_p0_tr_unlabeled_freeze.py `
    --snapshot $FreezeDir

if ($LASTEXITCODE -ne 0) {
    throw "Turkish unlabeled freeze verification failed."
}

Write-Host ""
Write-Host "P0 Turkish unlabeled freeze completed." -ForegroundColor Green
Write-Host "Copy the entire freeze directory to independent storage before opening labels."
