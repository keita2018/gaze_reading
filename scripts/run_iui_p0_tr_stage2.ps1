param(
    [string]$ProjectRoot = ".",
    [string]$CanonicalWebQAmGazeRoot = "data\raw\webqamgaze",
    [string]$Stage2Output = "outputs\webqamgaze_tr_confirmatory_stage2_v1"
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path $ProjectRoot)

python src\analysis\create_iui_p0_tr_stage2_clone.py `
    --project-root . `
    --source-script src\data\prepare_webqamgaze_external_validation_stage2.py `
    --output-script src\data\prepare_webqamgaze_tr_confirmatory_stage2.py `
    --manifest outputs\iui_p0_tr_stage2_clone_v1\clone_manifest.json

if ($LASTEXITCODE -ne 0) {
    throw "TR Stage-2 clone generation failed."
}

$trFiles = Get-ChildItem -Path $CanonicalWebQAmGazeRoot -Recurse -File `
    -Filter "*_TR_*.json"

Write-Host "Canonical TR JSON files found: $($trFiles.Count)"
if ($trFiles.Count -lt 40) {
    throw "Fewer than 40 TR JSON files found under $CanonicalWebQAmGazeRoot"
}

python src\data\prepare_webqamgaze_tr_confirmatory_stage2.py `
    --root $CanonicalWebQAmGazeRoot `
    --output-dir $Stage2Output

if ($LASTEXITCODE -ne 0) {
    throw "TR Stage-2 preparation failed."
}

python src\analysis\audit_iui_p0_tr_stage2_outputs.py `
    --project-root . `
    --stage2-output $Stage2Output `
    --output-dir outputs\iui_p0_tr_stage2_audit_v1

if ($LASTEXITCODE -ne 0) {
    throw "TR Stage-2 output audit failed."
}

python src\analysis\audit_iui_p0_tr_stage3_scripts.py `
    --project-root . `
    --output outputs\iui_p0_tr_stage3_script_audit_v1\stage3_script_audit.json

if ($LASTEXITCODE -ne 0) {
    throw "Stage-3 script audit failed."
}

Write-Host ""
Write-Host "P0 TR Stage-2 pipeline completed." -ForegroundColor Green
Write-Host "No model performance was calculated."
