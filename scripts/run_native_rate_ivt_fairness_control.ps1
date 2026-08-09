param(
    [string]$ProjectRoot = ".",
    [switch]$SkipAudit,
    [switch]$SkipFixed
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

function Invoke-PythonStep {
    param([string]$Name,[string[]]$Arguments,[string]$LogPath)
    Write-Host ""
    Write-Host ("=" * 80)
    Write-Host $Name
    Write-Host ("=" * 80)
    New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        $exitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previousPreference }
    if ($exitCode -ne 0) {
        Write-Host ""
        Write-Host "Last 80 log lines:"
        Get-Content $LogPath -Tail 80
        throw "$Name failed with exit code $exitCode. See $LogPath"
    }
}

$required = @(
    "src\models\audit_zuco_native_rate_ivt.py",
    "src\models\run_zuco_ivt_event_baseline_fixed_native500.py",
    "src\models\run_zuco_ivt_event_baseline_nested_threshold_native500.py",
    "src\models\run_zuco_ivt_event_baseline_fixed.py",
    "src\models\run_zuco_ivt_event_baseline_nested_threshold.py"
)
$missing = @($required | Where-Object { -not (Test-Path $_) })
if ($missing.Count -gt 0) { throw "Required scripts are missing: $($missing -join ', ')" }

if (-not $SkipAudit) {
    Invoke-PythonStep -Name "1/3 Audit original 500-Hz ZuCo streams" -Arguments @("src\models\audit_zuco_native_rate_ivt.py") -LogPath "logs\audit_zuco_native_rate_ivt.txt"
}
if (-not $SkipFixed) {
    Invoke-PythonStep -Name "2/3 Native-rate fixed I-VT (500 Hz except rate10)" -Arguments @("src\models\run_zuco_ivt_event_baseline_fixed_native500.py") -LogPath "logs\run_zuco_ivt_event_baseline_fixed_native500_r5.txt"
}
Invoke-PythonStep -Name "3/3 Native-rate nested threshold-tuned I-VT" -Arguments @("src\models\run_zuco_ivt_event_baseline_nested_threshold_native500.py") -LogPath "logs\run_zuco_ivt_event_baseline_nested_threshold_native500_r5.txt"

Write-Host ""
Write-Host ("=" * 80)
Write-Host "NATIVE-RATE I-VT FAIRNESS CONTROL COMPLETED" -ForegroundColor Green
Write-Host ("=" * 80)
Write-Host "Fixed summary: outputs\ivt_event_baseline_fixed_native500_v1\ivt_fixed_summary.csv"
Write-Host "Tuned summary: outputs\ivt_event_baseline_nested_threshold_native500_v1\ivt_nested_summary.csv"
Write-Host "Threshold summary: outputs\ivt_event_baseline_nested_threshold_native500_v1\ivt_nested_threshold_selection_summary.csv"
