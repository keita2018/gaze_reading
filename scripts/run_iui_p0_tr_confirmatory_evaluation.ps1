param(
    [string]$ProjectRoot = ".",
    [string]$OriginalSnapshot = "outputs\iui_p0_confirmatory_TR_v1",
    [string]$BackupSnapshot = "D:\silve\hozonn\iui_p0_confirmatory_TR_v1",
    [string]$VerificationOutput = "outputs\iui_p0_tr_backup_verification_v1",
    [string]$EvaluationOutput = "outputs\iui_p0_confirmatory_TR_evaluation_v1"
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

python src\analysis\verify_iui_p0_tr_backup_copy.py `
    --project-root . `
    --original $OriginalSnapshot `
    --backup $BackupSnapshot `
    --output-dir $VerificationOutput

if ($LASTEXITCODE -ne 0) {
    throw "Independent Turkish snapshot verification failed. Labels remain sealed."
}

python src\analysis\evaluate_iui_p0_tr_confirmatory.py `
    --project-root . `
    --snapshot $BackupSnapshot `
    --output-dir $EvaluationOutput `
    --bootstrap 2000 `
    --permutations 2000 `
    --seed 20260803

if ($LASTEXITCODE -ne 0) {
    throw "Turkish confirmatory evaluation failed."
}

Write-Host ""
Write-Host "P0 Turkish confirmatory evaluation completed." -ForegroundColor Green
Write-Host "Results: $EvaluationOutput"
