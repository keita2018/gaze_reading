param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectRoot = ".",

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "outputs\iui_p0_protocol_lock_v1"
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

$protocol = "docs\CONFIRMATORY_PROTOCOL.md"
$config = "configs\iui_p0_confirmatory_config.json"

foreach ($required in @($protocol, $config)) {
    if (-not (Test-Path $required)) {
        throw "Required file not found: $required"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
$gitCommit = ""
$gitStatus = ""
try {
    $gitCommit = (git rev-parse HEAD 2>$null).Trim()
    $gitStatus = (git status --porcelain 2>$null) -join "`n"
} catch {
    $gitCommit = "GIT_UNAVAILABLE"
    $gitStatus = "GIT_UNAVAILABLE"
}

Copy-Item $protocol (Join-Path $OutputDir "CONFIRMATORY_PROTOCOL.md") -Force
Copy-Item $config (Join-Path $OutputDir "iui_p0_confirmatory_config.json") -Force

$rootsToHash = @(
    "outputs\frozen_onestop_external_validation_v1",
    "outputs\linear_feature_group_factorization_v1",
    "outputs\webqamgaze_participant_self_calibration_v1",
    "outputs\final_factorized_selective_prediction_v1"
)

$hashRows = New-Object System.Collections.Generic.List[object]

function Add-HashRow([string]$PathValue) {
    if (Test-Path $PathValue -PathType Leaf) {
        $item = Get-Item $PathValue
        $hash = Get-FileHash -Algorithm SHA256 -Path $item.FullName
        $relative = Resolve-Path -Relative $item.FullName
        $hashRows.Add([PSCustomObject]@{
            sha256 = $hash.Hash.ToLower()
            bytes = $item.Length
            path = $relative
        })
    }
}

Add-HashRow $protocol
Add-HashRow $config

foreach ($rootPath in $rootsToHash) {
    if (Test-Path $rootPath) {
        Get-ChildItem -Path $rootPath -File -Recurse |
            Sort-Object FullName |
            ForEach-Object { Add-HashRow $_.FullName }
    } else {
        Write-Warning "Artifact root not found and was not hashed: $rootPath"
    }
}

$hashRows | Export-Csv -Path (Join-Path $OutputDir "file_hash_manifest.csv") -NoTypeInformation -Encoding UTF8
($hashRows | ForEach-Object { "$($_.sha256)  $($_.path)" }) |
    Set-Content -Path (Join-Path $OutputDir "SHA256SUMS") -Encoding UTF8

$pythonVersion = ""
try { $pythonVersion = (python --version 2>&1 | Out-String).Trim() } catch { $pythonVersion = "PYTHON_UNAVAILABLE" }

$environment = [ordered]@{
    freeze_timestamp = $timestamp
    project_root = (Get-Location).Path
    git_commit = $gitCommit
    git_dirty = [bool]($gitStatus)
    git_status_porcelain = $gitStatus
    powershell_version = $PSVersionTable.PSVersion.ToString()
    python_version = $pythonVersion
    protocol_file = $protocol
    config_file = $config
    artifact_roots = $rootsToHash
    hash_count = $hashRows.Count
}

$environment | ConvertTo-Json -Depth 6 |
    Set-Content -Path (Join-Path $OutputDir "release_manifest.json") -Encoding UTF8

$deviationFile = Join-Path $OutputDir "CONFIRMATORY_DEVIATIONS.md"
if (-not (Test-Path $deviationFile)) {
    @"
# Confirmatory Deviations

No deviations recorded at freeze time.

Use one section per deviation:

## YYYY-MM-DD HH:MM
- Reason:
- Protocol rule affected:
- Were target labels/performance already inspected?:
- Exact code/config change:
- Expected influence:
"@ | Set-Content -Path $deviationFile -Encoding UTF8
}

Write-Host ""
Write-Host "P0 confirmatory snapshot created:" -ForegroundColor Green
Write-Host "  $OutputDir"
Write-Host "Git commit: $gitCommit"
Write-Host "Hashed files: $($hashRows.Count)"
Write-Host ""
Write-Host "Next: copy the output directory to independent storage before target evaluation."
