param(
    [ValidateSet("preflight", "materialize", "verify")]
    [string]$Command = "preflight",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "venv\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "run_product_phase_2k_retrospective_audit.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project virtual-environment Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Phase 2K-C1 Python runner is missing: $runner"
}

if ($Command -eq "verify") {
    if ([string]::IsNullOrWhiteSpace($OutputDir)) {
        throw "-OutputDir is required for verify"
    }
    & $python $runner --repo-root $repoRoot verify --output-dir $OutputDir
} elseif ($Command -eq "materialize") {
    & $python $runner --repo-root $repoRoot materialize
} else {
    & $python $runner --repo-root $repoRoot preflight
}

if ($LASTEXITCODE -ne 0) {
    throw "Product Phase 2K-C1 failed closed with exit code $LASTEXITCODE"
}
