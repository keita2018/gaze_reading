param(
    [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

python src\models\make_onestop_acl2025_compatible_splits.py `
    --manifest data\interim\onestop\onestop_regular_trial_manifest.csv `
    --sequence-index data\processed\onestop\event_free_sequence_cache_v1\sequence_index.parquet `
    --output-dir outputs\onestop_acl2025_compatible_splits_v1 `
    --seed 20260808

if ($LASTEXITCODE -ne 0) {
    throw "ACL-2025-compatible split construction failed."
}

Write-Host ""
Write-Host "Split construction completed." -ForegroundColor Green
Write-Host "Audit: outputs\onestop_acl2025_compatible_splits_v1\split_fraction_summary.csv"
