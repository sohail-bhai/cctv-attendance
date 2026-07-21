[CmdletBinding()]
param(
    [string]$RepoRoot = "F:\sohail\Class_Attendance_YuNet_SFace",
    [switch]$Apply,
    [string]$Confirm = "",
    [switch]$Rollback,
    [string]$Backup = ""
)

$ErrorActionPreference = "Stop"

if ($Apply -and $Rollback) {
    throw "Choose either -Apply or -Rollback, not both."
}
if ($Rollback -and [string]::IsNullOrWhiteSpace($Backup)) {
    throw "-Rollback requires -Backup <exact backup JSON path>."
}

$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
$Script = Join-Path $RepoRoot "scripts\run_product_phase_2i_authority.py"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python was not found: $Python"
}
if (-not (Test-Path -LiteralPath $Script -PathType Leaf)) {
    throw "Phase 2I authority runner was not found: $Script"
}

$Arguments = @($Script, "--repo-root", $RepoRoot)
if ($Apply) {
    $Arguments += @("apply", "--confirm", $Confirm)
}
elseif ($Rollback) {
    $Arguments += @("rollback", "--backup", $Backup)
}
else {
    $Arguments += "preflight"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Product Phase 2I authority command failed with exit code $LASTEXITCODE."
}
