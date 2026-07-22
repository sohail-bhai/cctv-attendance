[CmdletBinding()]
param(
    [ValidateSet("preflight", "materialize", "verify")]
    [string]$Command = "preflight",
    [string]$OutputDir = "",
    [string[]]$Validated = @()
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repo "venv\Scripts\python.exe"
$Runner = Join-Path $PSScriptRoot "run_product_phase_2l_stabilization_preflight.py"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Product Phase 2L runner not found: $Runner"
}
if ($Command -eq "verify" -and [string]::IsNullOrWhiteSpace($OutputDir)) {
    throw "verify requires -OutputDir <immutable Phase 2L directory>"
}

$Arguments = @($Runner, "--repo-root", $Repo, $Command)
if ($Command -eq "verify") {
    $Arguments += @("--output-dir", $OutputDir)
}
if ($Command -eq "materialize") {
    foreach ($Check in $Validated) {
        $Arguments += @("--validated", $Check)
    }
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Product Phase 2L command failed with exit code $LASTEXITCODE."
}
