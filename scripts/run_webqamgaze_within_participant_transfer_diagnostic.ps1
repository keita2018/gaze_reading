param(
    [string]$ProjectRoot = ".",
    [int]$Bootstrap = 10000,
    [int]$Permutations = 10000,
    [int]$Seed = 20260809
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

$script = "src\models\run_webqamgaze_within_participant_transfer_diagnostic.py"
$english = "outputs\linear_feature_group_factorization_v1\target_feature_group_predictions.csv"
$turkish = "outputs\iui_p0_confirmatory_TR_evaluation_v1\confirmatory_joined_trial_predictions.parquet"
$out = "outputs\webqamgaze_within_participant_transfer_diagnostic_v1"
$log = "logs\run_webqamgaze_within_participant_transfer_diagnostic_v1.txt"

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
        --bootstrap $Bootstrap `
        --permutations $Permutations `
        --seed $Seed 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $prev
}

if ($exitCode -ne 0) {
    Get-Content $log -Tail 80
    throw "Within-participant diagnostic failed with exit code $exitCode."
}

Write-Host ""
Write-Host "WITHIN-PARTICIPANT TRANSFER DIAGNOSTIC COMPLETED" -ForegroundColor Green
Write-Host "Summary: $out\within_participant_summary.csv"
Write-Host "Permutation: $out\within_participant_permutation_summary.csv"
Write-Host "Markdown: $out\RESULTS_SUMMARY.md"
