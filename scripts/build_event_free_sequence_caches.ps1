param(
    [string]$ProjectRoot = ".",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

$overwriteArgument = @()
if ($Overwrite) {
    $overwriteArgument = @("--overwrite")
}

python src\data\build_event_free_sequence_cache.py `
    --dataset onestop `
    --samples-dir data\interim\onestop\raw_samples_30hz_asc344 `
    --manifest data\interim\onestop\onestop_regular_trial_manifest.csv `
    --word-aoi data\interim\onestop\onestop_regular_word_aoi.parquet `
    --participants all `
    --conditions clean,rate10,noise50,random40,burst40,high_random,high_burst `
    --repeats 1 `
    --max-length 768 `
    --seed 20260806 `
    --output-dir data\processed\onestop\event_free_sequence_cache_v1 `
    --audit-dir outputs\onestop_event_free_sequence_cache_v1 `
    @overwriteArgument

if ($LASTEXITCODE -ne 0) {
    throw "OneStop event-free sequence cache failed."
}

python src\data\build_event_free_sequence_cache.py `
    --dataset webqamgaze `
    --web-stage2-dir outputs\webqamgaze_external_validation_stage2 `
    --web-stage3c-dir outputs\webqamgaze_external_validation_stage3c `
    --max-length 768 `
    --seed 20260806 `
    --output-dir data\processed\webqamgaze\event_free_sequence_cache_en_v1 `
    --audit-dir outputs\webqamgaze_event_free_sequence_cache_en_v1 `
    @overwriteArgument

if ($LASTEXITCODE -ne 0) {
    throw "WebQAmGaze event-free sequence cache failed."
}

Write-Host ""
Write-Host "Event-free sequence caches completed." -ForegroundColor Green
