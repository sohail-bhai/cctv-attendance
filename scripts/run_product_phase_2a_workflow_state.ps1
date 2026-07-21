param(
    [switch]$Preflight,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$Root = "F:\sohail\Class_Attendance_YuNet_SFace"
$Python = Join-Path $Root "venv\Scripts\python.exe"
$Script = Join-Path $Root "scripts\run_product_phase_2a_workflow_state.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python not found: $Python"
}
if (-not (Test-Path -LiteralPath $Script)) {
    throw "Phase 2A runner not found: $Script"
}

if ($Preflight) {
    & $Python $Script preflight
    exit $LASTEXITCODE
}

& $Python $Script run --confirm-backend-stopped BACKEND_STOPPED
if ($LASTEXITCODE -ne 0) {
    throw "Product Phase 2A workflow-state migration failed with exit code $LASTEXITCODE."
}
