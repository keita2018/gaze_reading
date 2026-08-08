param(
    [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

python src\data\build_event_free_sequence_cache.py `
    --dataset onestop `
    --samples-dir data\interim\onestop\raw_samples_30hz_asc344 `
    --manifest data\interim\onestop\onestop_regular_trial_manifest.csv `
    --word-aoi data\interim\onestop\onestop_regular_word_aoi.parquet `
    --participants all `
    --output-dir data\processed\onestop\event_free_sequence_cache_v1 `
    --audit-dir outputs\onestop_event_free_sequence_cache_v1 `
    --audit-only

if ($LASTEXITCODE -ne 0) {
    throw "OneStop event-free sequence input audit failed."
}

python src\data\build_event_free_sequence_cache.py `
    --dataset webqamgaze `
    --web-stage2-dir outputs\webqamgaze_external_validation_stage2 `
    --web-stage3c-dir outputs\webqamgaze_external_validation_stage3c `
    --output-dir data\processed\webqamgaze\event_free_sequence_cache_en_v1 `
    --audit-dir outputs\webqamgaze_event_free_sequence_cache_en_v1 `
    --audit-only

if ($LASTEXITCODE -ne 0) {
    throw "WebQAmGaze event-free sequence input audit failed."
}

python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host ""
Write-Host "Event-Free Transformer input audits completed." -ForegroundColor Green
