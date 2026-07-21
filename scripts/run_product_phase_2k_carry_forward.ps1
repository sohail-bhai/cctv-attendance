param(
    [ValidateSet("preflight", "materialize", "verify", "validate-capture")]
    [string]$Command = "preflight",
    [string]$OutputDir = "",
    [string]$PackageDir = "",
    [string]$Manifest = "capture_manifest.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "venv\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "run_product_phase_2k_carry_forward.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project virtual-environment Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Phase 2K-B Python runner is missing: $runner"
}

if ($Command -eq "verify") {
    if ([string]::IsNullOrWhiteSpace($OutputDir)) {
        throw "-OutputDir is required for verify"
    }
    & $python $runner --repo-root $repoRoot verify --output-dir $OutputDir
} elseif ($Command -eq "validate-capture") {
    if ([string]::IsNullOrWhiteSpace($PackageDir)) {
        throw "-PackageDir is required for validate-capture"
    }
    & $python $runner --repo-root $repoRoot validate-capture --package-dir $PackageDir --manifest $Manifest
} elseif ($Command -eq "materialize") {
    & $python $runner --repo-root $repoRoot materialize
} else {
    & $python $runner --repo-root $repoRoot preflight
}

if ($LASTEXITCODE -ne 0) {
    throw "Product Phase 2K-B failed closed with exit code $LASTEXITCODE"
}
