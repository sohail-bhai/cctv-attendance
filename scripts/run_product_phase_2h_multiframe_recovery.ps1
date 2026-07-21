param(
    [string]$Repository = "F:\sohail\Class_Attendance_YuNet_SFace",
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $Repository "venv\Scripts\python.exe"
$Runner = Join-Path $Repository "scripts\run_product_phase_2h_multiframe_recovery.py"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python was not found: $Python"
}
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Phase 2H runner was not found: $Runner"
}

$Stdout = [IO.Path]::GetTempFileName()
$Stderr = [IO.Path]::GetTempFileName()
try {
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList @($Runner, "--repo-root", $Repository, "export-review") `
        -WorkingDirectory $Repository `
        -Wait `
        -PassThru `
        -NoNewWindow `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr

    $Output = if (Test-Path -LiteralPath $Stdout) { Get-Content -LiteralPath $Stdout } else { @() }
    $Errors = if (Test-Path -LiteralPath $Stderr) { Get-Content -LiteralPath $Stderr } else { @() }
    $Output | ForEach-Object { Write-Host $_ }
    $Errors | ForEach-Object { Write-Host $_ }
    if ($Process.ExitCode -ne 0) {
        throw "Product Phase 2H export failed with exit code $($Process.ExitCode)."
    }

    $ReviewerLine = $Output | Where-Object { "$_" -like "PRODUCT_PHASE_2H_REVIEWER=*" } | Select-Object -Last 1
    if (-not $ReviewerLine) {
        throw "Phase 2H completed without reporting its reviewer path."
    }
    $Reviewer = ("$ReviewerLine").Substring("PRODUCT_PHASE_2H_REVIEWER=".Length)
    if (-not (Test-Path -LiteralPath $Reviewer -PathType Leaf)) {
        throw "Reported reviewer was not found: $Reviewer"
    }
    if (-not $NoOpen) {
        Start-Process $Reviewer
    }
}
finally {
    Remove-Item -LiteralPath $Stdout, $Stderr -Force -ErrorAction SilentlyContinue
}
