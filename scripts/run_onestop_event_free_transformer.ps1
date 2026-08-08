param(
    [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

python src\models\run_onestop_event_free_transformer.py `
    --source-cache data\processed\onestop\event_free_sequence_cache_v1 `
    --target-cache data\processed\webqamgaze\event_free_sequence_cache_en_v1 `
    --protocols clean_only,augmented `
    --outer-folds 5 `
    --validation-fraction 0.10 `
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
    --bootstrap 2000 `
    --seed 20260806 `
    --output-dir outputs\onestop_event_free_transformer_v1

if ($LASTEXITCODE -ne 0) {
    throw "Event-Free Transformer experiment failed."
}

Write-Host ""
Write-Host "Event-Free Transformer experiment completed." -ForegroundColor Green
Write-Host "Summary: outputs\onestop_event_free_transformer_v1\EVENT_FREE_TRANSFORMER_SUMMARY.md"
