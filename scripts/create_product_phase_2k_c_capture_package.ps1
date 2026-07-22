param(
    [ValidateSet("create", "verify", "tooling-validate", "tooling-verify")]
    [string]$Command = "create",
    [string]$OutputRoot = "",
    [string]$SourceDir = "",
    [string[]]$Source = @(),
    [string[]]$CaptureTimestamp = @(),
    [string]$SessionDate = "",
    [string]$Section = "",
    [string]$Period = "",
    [string]$Subject = "",
    [string]$Room = "",
    [string]$Confirm = "",
    [switch]$DryRun,
    [string]$PackageDir = "",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "venv\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "create_product_phase_2k_c_capture_package.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project virtual-environment Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Phase 2K-C0 capture-package runner is missing: $runner"
}

$arguments = @($runner, "--repo-root", $repoRoot, $Command)
if ($Command -eq "verify") {
    if ([string]::IsNullOrWhiteSpace($PackageDir)) { throw "-PackageDir is required for verify" }
    $arguments += @("--package-dir", $PackageDir)
} elseif ($Command -eq "tooling-verify") {
    if ([string]::IsNullOrWhiteSpace($OutputDir)) { throw "-OutputDir is required for tooling-verify" }
    $arguments += @("--output-dir", $OutputDir)
} elseif ($Command -eq "tooling-validate") {
    if (-not [string]::IsNullOrWhiteSpace($OutputRoot)) { $arguments += @("--output-root", $OutputRoot) }
} else {
    if ([string]::IsNullOrWhiteSpace($OutputRoot)) { throw "-OutputRoot is required for create" }
    $arguments += @(
        "--output-root", $OutputRoot,
        "--session-date", $SessionDate,
        "--section", $Section,
        "--period", $Period,
        "--subject", $Subject,
        "--room", $Room,
        "--confirm", $Confirm
    )
    if (-not [string]::IsNullOrWhiteSpace($SourceDir)) {
        $arguments += @("--source-dir", $SourceDir)
    } else {
        foreach ($item in $Source) { $arguments += @("--source", $item) }
    }
    foreach ($item in $CaptureTimestamp) { $arguments += @("--capture-timestamp", $item) }
    if ($DryRun) { $arguments += "--dry-run" }
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Product Phase 2K-C0 capture intake failed closed with exit code $LASTEXITCODE"
}
