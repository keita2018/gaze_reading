param(
    [string]$ProjectRoot = ".",
    [string]$WebQAmGazeRoot = "data\raw\webqamgaze",
    [string]$Stage2Dir = "outputs\webqamgaze_tr_confirmatory_stage2_v1",
    [string]$Stage3ADir = "outputs\webqamgaze_tr_confirmatory_stage3a_v1",
    [string]$Stage3BDir = "outputs\webqamgaze_tr_confirmatory_stage3b_v1",
    [string]$Stage3CDir = "outputs\webqamgaze_tr_confirmatory_stage3c_v1",
    [string]$FreezeDir = "outputs\iui_p0_confirmatory_TR_v1"
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

python src\data\prepare_webqamgaze_external_validation_stage3a.py `
    --root $WebQAmGazeRoot `
    --stage2-dir $Stage2Dir `
    --frozen-dir outputs\frozen_zuco_external_validation_v1 `
    --output-dir $Stage3ADir `
    --zuco-reference-width-px 800 `
    --zuco-reference-height-px 600

if ($LASTEXITCODE -ne 0) {
    throw "Turkish Stage 3A failed."
}

python src\data\resolve_webqamgaze_aoi_stage3b.py `
    --root $WebQAmGazeRoot `
    --stage2-dir $Stage2Dir `
    --stage3a-dir $Stage3ADir `
    --output-dir $Stage3BDir `
    --max-resource-files 5000

if ($LASTEXITCODE -ne 0) {
    throw "Turkish Stage 3B failed."
}

python src\data\prepare_webqamgaze_external_validation_stage3c.py `
    --root $WebQAmGazeRoot `
    --stage2-dir $Stage2Dir `
    --stage3a-dir $Stage3ADir `
    --stage3b-dir $Stage3BDir `
    --frozen-dir outputs\frozen_zuco_external_validation_v1 `
    --output-dir $Stage3CDir `
    --rate-hz 30 `
    --sigma-word-height-ratio 0.6666666666666666 `
    --active-threshold 0.1

if ($LASTEXITCODE -ne 0) {
    throw "Turkish Stage 3C failed."
}

python src\analysis\freeze_iui_p0_tr_unlabeled_artifacts.py `
    --project-root . `
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
Write-Host "P0 Turkish Stage 3 and unlabeled freeze completed." -ForegroundColor Green
Write-Host "Do not open sealed\labels_sealed.csv yet."
Write-Host "Copy the entire freeze directory to independent storage first."
