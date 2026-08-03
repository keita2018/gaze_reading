param(
    [string]$ProjectRoot = ".",
    [string]$OutputDir = "outputs\iui_p1_provenance_candidate_inventory_v1"
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

$keywords = @(
    "dual", "holdout", "common16", "soft", "hard", "ivt",
    "robust", "degrad", "noise", "missing", "qcsr", "qcr",
    "augmentation", "selective", "coverage", "aurc", "risk",
    "event_baseline", "event-based", "transferred", "onestop",
    "quality", "bootstrap", "summary", "metrics"
)

$roots = @("outputs", "logs", "src") | Where-Object { Test-Path $_ }
$regex = ($keywords | ForEach-Object { [regex]::Escape($_) }) -join "|"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$rows = foreach ($root in $roots) {
    Get-ChildItem -Path $root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match $regex -and
            $_.Extension.ToLowerInvariant() -in @(
                ".csv", ".json", ".md", ".txt", ".parquet",
                ".py", ".ps1", ".png", ".pdf", ".joblib"
            )
        } |
        ForEach-Object {
            $relative = Resolve-Path -Relative $_.FullName
            $hash = ""
            if ($_.Length -le 50MB) {
                try { $hash = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLowerInvariant() }
                catch { $hash = "HASH_ERROR" }
            } else {
                $hash = "SKIPPED_OVER_50MB"
            }

            [pscustomobject]@{
                relative_path = $relative
                file_name = $_.Name
                extension = $_.Extension.ToLowerInvariant()
                bytes = $_.Length
                modified_utc = $_.LastWriteTimeUtc.ToString("o")
                sha256 = $hash
            }
        }
}

$rows = $rows | Sort-Object relative_path -Unique
$inventory = Join-Path $OutputDir "provenance_candidate_inventory.csv"
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $inventory

$summary = Join-Path $OutputDir "PROVENANCE_CANDIDATE_SUMMARY.md"
@"
# P1 Provenance Candidate Inventory

- Candidate files: $($rows.Count)
- Search roots: $($roots -join ', ')
- Inventory: `provenance_candidate_inventory.csv`

## Priority evidence still required

1. ZuCo dual-holdout pooled and participant-level summaries.
2. Soft/Hard/fixed I-VT/quality-tuned I-VT condition-level metrics.
3. OneStop clean-only/augmentation/QCSR/QCR nested-evaluation summaries.
4. English transferred event-based baseline output.
5. English selective-prediction coverage, AURC/eAURC, and confidence-correctness outputs.
6. Figure-generation scripts or source tables for Figures 4-7.

This inventory contains paths and hashes only; it does not copy or modify source artifacts.
"@ | Set-Content -Encoding UTF8 -Path $summary

Write-Host "P1 provenance candidate inventory completed." -ForegroundColor Green
Write-Host "Candidates: $($rows.Count)"
Write-Host "Inventory: $inventory"
Write-Host "Summary: $summary"
