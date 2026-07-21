param(
    [string]$Repository = "F:\sohail\Class_Attendance_YuNet_SFace"
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $Repository "venv\Scripts\python.exe"
$Script = Join-Path $Repository "scripts\run_product_phase_2f_config_preflight.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}
if (-not (Test-Path -LiteralPath $Script)) {
    throw "Phase 2F preflight script was not found: $Script"
}

& $Python $Script --repo-root $Repository
if ($LASTEXITCODE -ne 0) {
    throw "Product Phase 2F preflight failed with exit code $LASTEXITCODE."
}
