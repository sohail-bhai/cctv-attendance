[CmdletBinding()]
param(
    [ValidateSet("preflight", "materialize", "verify")]
    [string]$Command = "preflight",
    [string]$OutputDir = "",
    [string]$FrontendBuildDir = "",
    [string[]]$Validated = @()
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repo "venv\Scripts\python.exe"
$Runner = Join-Path $PSScriptRoot "run_product_phase_2l_auth_preflight.py"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python environment not found: $Python"
}
if ($Command -eq "verify" -and [string]::IsNullOrWhiteSpace($OutputDir)) {
    throw "verify requires -OutputDir"
}
if ($Command -eq "materialize" -and [string]::IsNullOrWhiteSpace($FrontendBuildDir)) {
    throw "materialize requires -FrontendBuildDir"
}

$Arguments = @($Runner, "--repo-root", $Repo, $Command)
if ($Command -eq "verify") {
    $Arguments += @("--output-dir", $OutputDir)
}
if ($Command -eq "materialize") {
    $Arguments += @("--frontend-build-dir", $FrontendBuildDir)
    foreach ($Check in $Validated) {
        $Arguments += @("--validated", $Check)
    }
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Product Phase 2L-C command failed with exit code $LASTEXITCODE."
}
