param(
    [int]$Repeats = 5
)

$ErrorActionPreference = "Stop"

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$LogPath
    )

    Write-Host ""
    Write-Host ("=" * 80)
    Write-Host $Name
    Write-Host ("=" * 80)

    # Windows PowerShell converts native stderr to ErrorRecord objects.
    # With ErrorActionPreference=Stop this can terminate at the first
    # traceback line and hide the real exception. Temporarily continue,
    # capture the complete stream, then check the native exit code.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($exitCode -ne 0) {
        Write-Host ""
        Write-Host "Last 80 log lines:"
        Get-Content $LogPath -Tail 80
        throw "$Name failed with exit code $exitCode. See $LogPath"
    }
}

New-Item -ItemType Directory -Force -Path logs | Out-Null

$requiredScripts = @(
    "src\models\prepare_zuco_target_soft_aoi_five_repeat_caches.py",
    "src\models\run_zuco_soft_vs_hard_aoi_coordinate_noise_30hz.py",
    "src\models\run_zuco_soft_vs_hard_aoi_missingness_30hz.py",
    "src\models\run_zuco_soft_vs_hard_aoi_combined_high_30hz.py",
    "src\models\run_zuco_ivt_event_baseline_fixed.py",
    "src\models\run_zuco_ivt_event_baseline_nested_threshold.py",
    "src\analysis\generate_zuco_final_representation_comparison.py",
    "src\data\audit_webqamgaze_participant_trial_counts.py"
)

$missing = @(
    $requiredScripts | Where-Object {
        -not (Test-Path $_)
    }
)
if ($missing.Count -gt 0) {
    throw "Required scripts are missing: $($missing -join ', ')"
}

Invoke-PythonStep `
    -Name "1/8 Prepare missing Soft-AOI repeats 4-5" `
    -Arguments @(
        "src\models\prepare_zuco_target_soft_aoi_five_repeat_caches.py",
        "--repeats", "$Repeats"
    ) `
    -LogPath "logs\prepare_zuco_target_soft_aoi_five_repeat_caches.txt"

Invoke-PythonStep `
    -Name "2/8 Soft vs Hard: coordinate noise, five repeats" `
    -Arguments @(
        "src\models\run_zuco_soft_vs_hard_aoi_coordinate_noise_30hz.py",
        "--repeats", "$Repeats"
    ) `
    -LogPath "logs\run_zuco_soft_vs_hard_aoi_coordinate_noise_30hz_r5.txt"

Invoke-PythonStep `
    -Name "3/8 Soft vs Hard: random/burst missingness, five repeats" `
    -Arguments @(
        "src\models\run_zuco_soft_vs_hard_aoi_missingness_30hz.py",
        "--repeats", "$Repeats"
    ) `
    -LogPath "logs\run_zuco_soft_vs_hard_aoi_missingness_30hz_r5.txt"

Invoke-PythonStep `
    -Name "4/8 Soft vs Hard: high combined degradation, five repeats" `
    -Arguments @(
        "src\models\run_zuco_soft_vs_hard_aoi_combined_high_30hz.py",
        "--repeats", "$Repeats"
    ) `
    -LogPath "logs\run_zuco_soft_vs_hard_aoi_combined_high_30hz_r5.txt"

Invoke-PythonStep `
    -Name "5/8 Fixed I-VT event baseline, five repeats" `
    -Arguments @(
        "src\models\run_zuco_ivt_event_baseline_fixed.py",
        "--conditions",
        "clean,rate10,noise50,random40,burst40,high_random,high_burst",
        "--repeats", "$Repeats"
    ) `
    -LogPath "logs\run_zuco_ivt_event_baseline_fixed_r5.txt"

Invoke-PythonStep `
    -Name "6/8 Nested quality-specific I-VT, five repeats" `
    -Arguments @(
        "src\models\run_zuco_ivt_event_baseline_nested_threshold.py",
        "--conditions",
        "clean,rate10,noise50,random40,burst40,high_random,high_burst",
        "--velocity-thresholds",
        "30,60,120,240,480",
        "--inner-splits", "4",
        "--repeats", "$Repeats"
    ) `
    -LogPath "logs\run_zuco_ivt_event_baseline_nested_threshold_r5.txt"

Invoke-PythonStep `
    -Name "7/8 Regenerate final four-representation comparison" `
    -Arguments @(
        "src\analysis\generate_zuco_final_representation_comparison.py"
    ) `
    -LogPath "logs\generate_zuco_final_representation_comparison_r5.txt"

Invoke-PythonStep `
    -Name "8/8 Audit WebQAmGaze participant/trial counting unit" `
    -Arguments @(
        "src\data\audit_webqamgaze_participant_trial_counts.py"
    ) `
    -LogPath "logs\audit_webqamgaze_participant_trial_counts.txt"

Write-Host ""
Write-Host ("=" * 80)
Write-Host "FINAL FIVE-REPEAT PIPELINE COMPLETED"
Write-Host ("=" * 80)
Write-Host "Final comparison:"
Write-Host "  outputs\final_representation_comparison\final_representation_primary_metrics_wide.csv"
Write-Host "WebQAmGaze count audit:"
Write-Host "  outputs\webqamgaze_participant_trial_count_audit\participant_trial_count_audit.json"
