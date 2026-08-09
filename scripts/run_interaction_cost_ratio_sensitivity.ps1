param(
    [string]$ProjectRoot = ".",
    [int]$Bootstrap = 5000,
    [int]$Seed = 20260809,
    [int]$ProgressEvery = 250
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

$script = "src\models\run_interaction_cost_ratio_sensitivity.py"
$english = "outputs\linear_feature_group_factorization_v1\target_feature_group_predictions.csv"
$turkish = "outputs\iui_p0_confirmatory_TR_evaluation_v1\confirmatory_joined_trial_predictions.parquet"
$out = "outputs\interaction_cost_ratio_sensitivity_v1"
$log = "logs\run_interaction_cost_ratio_sensitivity_v1.txt"

foreach ($path in @($script, $english, $turkish)) {
    if (-not (Test-Path $path -PathType Leaf)) { throw "Required file not found: $path" }
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
        --key-lambdas "0.5,1,1.5,2,3,5,10,20" `
        --lambda-min 0.1 `
        --lambda-max 30 `
        --lambda-grid-points 181 `
        --bootstrap $Bootstrap `
        --seed $Seed `
        --progress-every $ProgressEvery 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
}
finally { $ErrorActionPreference = $prev }

if ($exitCode -ne 0) {
    Get-Content $log -Tail 120
    throw "Interaction-cost ratio sensitivity failed with exit code $exitCode."
}

Write-Host ""
Write-Host "INTERACTION-COST RATIO SENSITIVITY COMPLETED" -ForegroundColor Green
Write-Host "Break-even: $out\interaction_cost_break_even.csv"
Write-Host "Dense curves: $out\interaction_cost_lambda_curves.csv"
Write-Host "Bootstrap: $out\interaction_cost_bootstrap_summary.csv"
Write-Host "Optimal coverage: $out\optimal_coverage_by_lambda.csv"
