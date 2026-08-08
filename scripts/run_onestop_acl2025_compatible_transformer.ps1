param(
    [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

python src\models\run_onestop_acl2025_compatible_event_free_transformer.py `
    --cache data\processed\onestop\event_free_sequence_cache_v1 `
    --splits outputs\onestop_acl2025_compatible_splits_v1\split_assignments.parquet `
    --output-dir outputs\onestop_acl2025_compatible_event_free_transformer_v1 `
    --d-model 64 `
    --n-heads 4 `
    --n-layers 2 `
    --ff-multiplier 4 `
    --dropout 0.15 `
    --batch-size 64 `
    --epochs 40 `
    --patience 6 `
    --learning-rate 0.0007 `
    --weight-decay 0.0003 `
    --seed 20260808

if ($LASTEXITCODE -ne 0) {
    throw "ACL-2025-compatible Event-Free Transformer evaluation failed."
}

Write-Host ""
Write-Host "Event-Free Transformer compatibility evaluation completed." -ForegroundColor Green
Write-Host "Summary: outputs\onestop_acl2025_compatible_event_free_transformer_v1\ACL2025_COMPAT_TRANSFORMER_SUMMARY.md"
