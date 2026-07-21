param(
    [switch]$Apply,
    [switch]$Rollback,
    [string]$Backup = "",
    [string]$Confirm = ""
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repo "venv\Scripts\python.exe"
$Script = Join-Path $PSScriptRoot "run_product_phase_2j_stabilization.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python environment not found: $Python"
}
if ($Apply -and $Rollback) {
    throw "Choose either -Apply or -Rollback, not both."
}

$Arguments = @(
    $Script,
    "--repo-root", $Repo
)
if ($Apply) {
    $Arguments += @("--apply", "--confirm", $Confirm)
}
elseif ($Rollback) {
    if ([string]::IsNullOrWhiteSpace($Backup)) {
        throw "Rollback requires -Backup <Phase 2J state backup path>."
    }
    $Arguments += @("--rollback", "--backup", $Backup, "--confirm", $Confirm)
}
elseif (-not [string]::IsNullOrWhiteSpace($Backup)) {
    throw "-Backup is valid only with -Rollback."
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Product Phase 2J stabilization failed with exit code $LASTEXITCODE."
}
