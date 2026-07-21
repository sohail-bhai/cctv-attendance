param(
    [string]$RepoRoot = "F:\sohail\Class_Attendance_YuNet_SFace",
    [switch]$Json
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
$Script = Join-Path $RepoRoot "scripts\run_product_phase_2g_processing_preflight.py"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python was not found: $Python"
}
if (-not (Test-Path -LiteralPath $Script -PathType Leaf)) {
    throw "Phase 2G preflight script was not found: $Script"
}

$Arguments = @($Script, "--repo-root", $RepoRoot)
if ($Json) { $Arguments += "--json" }

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Product Phase 2G preflight failed with exit code $LASTEXITCODE."
}
