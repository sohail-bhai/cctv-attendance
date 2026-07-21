param(
    [string]$RepoRoot = "F:\sohail\Class_Attendance_YuNet_SFace",
    [switch]$PreflightOnly,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
$Script = Join-Path $RepoRoot "scripts\run_phase_1_2l_1_selection_audit.py"

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host ""
    Write-Host "[$Name]" -ForegroundColor Cyan
    $Started = Get-Date
    $StdOutPath = [System.IO.Path]::GetTempFileName()
    $StdErrPath = [System.IO.Path]::GetTempFileName()
    try {
        $Process = Start-Process `
            -FilePath $Python `
            -ArgumentList $Arguments `
            -WorkingDirectory $RepoRoot `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $StdOutPath `
            -RedirectStandardError $StdErrPath

        $StdOutLines = @()
        $StdErrLines = @()
        if ((Get-Item -LiteralPath $StdOutPath).Length -gt 0) {
            $StdOutLines = @(Get-Content -LiteralPath $StdOutPath -Encoding UTF8)
        }
        if ((Get-Item -LiteralPath $StdErrPath).Length -gt 0) {
            $StdErrLines = @(Get-Content -LiteralPath $StdErrPath -Encoding UTF8)
        }
        foreach ($Line in $StdOutLines) { Write-Host $Line }
        foreach ($Line in $StdErrLines) { Write-Host $Line }
        if ($Process.ExitCode -ne 0) {
            throw "$Name failed with exit code $($Process.ExitCode). No later stage was started."
        }
        $Elapsed = ((Get-Date) - $Started).TotalSeconds
        Write-Host "$Name passed in $([math]::Round($Elapsed, 2)) seconds."
        return [pscustomobject]@{
            ExitCode = $Process.ExitCode
            StdOut = $StdOutLines
            StdErr = $StdErrLines
        }
    }
    finally {
        Remove-Item -LiteralPath $StdOutPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $StdErrPath -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repository not found: $RepoRoot"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $Script -PathType Leaf)) {
    throw "Phase 1.2L.1 Python runner not found: $Script"
}

$OriginalLocation = Get-Location
$OriginalPythonPath = $env:PYTHONPATH
try {
    Set-Location -LiteralPath $RepoRoot
    $env:PYTHONPATH = $RepoRoot

    Write-Host "Phase 1.2L.1 - promotion-evidence/parsimony selection audit" -ForegroundColor Cyan
    Write-Host "Repository: $RepoRoot"
    Write-Host "Phase 1.2L scoring repeated: no"
    Write-Host "MON_P3 recognition repeated: no"
    Write-Host "Manual review repeated: no"
    Write-Host "Production changes allowed: no"
    Write-Host "Candidate family build allowed: no"

    $null = Invoke-CheckedPython -Name "Python compilation" -Arguments @(
        "-m", "py_compile",
        "src\face_attendance\mon_p3_source_ablation.py",
        "scripts\run_phase_1_2l_1_selection_audit.py",
        "tests\test_mon_p3_source_ablation.py"
    )

    $null = Invoke-CheckedPython -Name "Focused Phase 1.2K + 1.2L selection tests" -Arguments @(
        "-m", "unittest", "-v",
        "tests.test_mon_p3_retention_attribution",
        "tests.test_mon_p3_source_ablation"
    )

    $null = Invoke-CheckedPython -Name "Read-only Phase 1.2L.1 preflight" -Arguments @(
        $Script,
        "--repo-root", $RepoRoot,
        "--preflight-only"
    )

    if ($PreflightOnly) {
        Write-Host ""
        Write-Host "Phase 1.2L.1 preflight completed safely." -ForegroundColor Green
        Write-Host "Selection audit written: no"
        Write-Host "Source ablation rescored: no"
        Write-Host "Production embeddings changed: no"
        Write-Host "Official attendance changed: no"
        Write-Host "New embedding family built: no"
        Write-Host "Candidate promoted: no"
        exit 0
    }

    $RunResult = Invoke-CheckedPython -Name "Immutable selection audit" -Arguments @(
        $Script,
        "--repo-root", $RepoRoot
    )

    $OutputLine = $RunResult.StdOut | Where-Object { $_ -match '^PHASE_1_2L_1_OUTPUT=(.+)$' } | Select-Object -Last 1
    if ($null -eq $OutputLine) {
        throw "Phase 1.2L.1 completed but did not report its output directory."
    }
    $OutputDir = ([regex]::Match([string]$OutputLine, '^PHASE_1_2L_1_OUTPUT=(.+)$')).Groups[1].Value.Trim()

    Write-Host ""
    Write-Host "Phase 1.2L.1 completed safely." -ForegroundColor Green
    Write-Host "Output: $OutputDir"
    Write-Host "Corrected build-design composition: ABL-16-ee3a4ff6"
    Write-Host "Kept implicated source: 0140_p2_cp1_medoid_debcd0"
    Write-Host "Production embeddings changed: no"
    Write-Host "Official attendance changed: no"
    Write-Host "New embedding family built: no"
    Write-Host "Candidate promoted: no"
    Write-Host "Full MON_P3 recognition repeated: no"
    Write-Host "Manual review repeated: no"
    Write-Host "Next action: build a new immutable family from the corrected ABL-16 composition."

    $Report = Join-Path $OutputDir "selection_report.md"
    if (-not $NoOpen -and (Test-Path -LiteralPath $Report -PathType Leaf)) {
        Start-Process -FilePath $Report
    }
}
finally {
    if ($null -eq $OriginalPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $OriginalPythonPath
    }
    Set-Location -LiteralPath $OriginalLocation
}
