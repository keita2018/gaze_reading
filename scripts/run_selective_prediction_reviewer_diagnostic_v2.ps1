param(
    [string]$ProjectRoot = ".",
    [int]$Bootstrap = 2000,
    [int]$Seed = 20260809,
    [int]$ProgressEvery = 100
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

$script = "src\models\run_selective_prediction_reviewer_diagnostic_v2.py"
$english = "outputs\linear_feature_group_factorization_v1\target_feature_group_predictions.csv"
$turkish = "outputs\iui_p0_confirmatory_TR_evaluation_v1\confirmatory_joined_trial_predictions.parquet"
$out = "outputs\selective_prediction_reviewer_diagnostic_v2"
$log = "logs\run_selective_prediction_reviewer_diagnostic_v2.txt"

foreach ($path in @($script, $english, $turkish)) {
    if (-not (Test-Path $path -PathType Leaf)) {
        throw "Required file not found: $path"
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    python -u $script `
        --english-predictions $english `
        --turkish-predictions $turkish `
        --output-dir $out `
        --coverages "1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1" `
        --bootstrap $Bootstrap `
        --seed $Seed `
        --progress-every $ProgressEvery 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $prev
}

if ($exitCode -ne 0) {
    Write-Host ""
    Write-Host "Last 100 log lines:"
    Get-Content $log -Tail 100
    throw "Selective-prediction reviewer diagnostic v2 failed with exit code $exitCode."
}

Write-Host ""
Write-Host "SELECTIVE-PREDICTION REVIEWER DIAGNOSTIC V2 COMPLETED" -ForegroundColor Green
Write-Host "Coverage: $out\coverage_metrics.csv"
Write-Host "Deciles: $out\confidence_deciles.csv"
Write-Host "Action tradeoff: $out\action_abstention_tradeoff.csv"
Write-Host "Bootstrap CI: $out\bootstrap_selective_ci.csv"
Write-Host "Extreme coverage: $out\extreme_coverage_anomaly_summary.csv"
