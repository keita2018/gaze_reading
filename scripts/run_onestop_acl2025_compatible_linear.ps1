param(
    [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

python src\models\run_onestop_acl2025_compatible_linear.py `
    --features data\processed\onestop\quality_views_asc344_v1\onestop_quality_views_all_features.parquet `
    --splits outputs\onestop_acl2025_compatible_splits_v1\split_assignments.parquet `
    --output-dir outputs\onestop_acl2025_compatible_linear_v1 `
    --c-grid 0.01,0.1,1,10 `
    --seed 20260808

if ($LASTEXITCODE -ne 0) {
    throw "ACL-2025-compatible LinearSVM evaluation failed."
}

Write-Host ""
Write-Host "LinearSVM compatibility evaluation completed." -ForegroundColor Green
Write-Host "Summary: outputs\onestop_acl2025_compatible_linear_v1\summary_mean_95ci.csv"
