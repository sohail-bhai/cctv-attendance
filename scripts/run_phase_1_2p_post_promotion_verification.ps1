param(
    [string]$RepoRoot = "F:\sohail\Class_Attendance_YuNet_SFace",
    [switch]$PreflightOnly,
    [switch]$SkipFocusedTests,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
$Cli = Join-Path $RepoRoot "scripts\run_phase_1_2p_post_promotion_verification.py"
$PreflightOutput = Join-Path $RepoRoot "attendance_output\embedding_forensics\phase_1_2p\phase_1_2p_preflight.json"
$OutputRoot = Join-Path $RepoRoot "attendance_output\embedding_forensics\phase_1_2p"

function Invoke-CheckedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    Write-Host ""
    Write-Host "[$Name]" -ForegroundColor Cyan
    $Started = Get-Date
    $StdOutPath = [System.IO.Path]::GetTempFileName()
    $StdErrPath = [System.IO.Path]::GetTempFileName()
    try {
        $Process = Start-Process `
            -FilePath $FilePath `
            -ArgumentList $Arguments `
            -WorkingDirectory $WorkingDirectory `
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
            throw "$Name failed with exit code $($Process.ExitCode)."
        }
        $Elapsed = ((Get-Date) - $Started).TotalSeconds
        Write-Host "$Name passed in $([math]::Round($Elapsed, 2)) seconds." -ForegroundColor Green
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

Write-Host "Phase 1.2P post-promotion operational verification" -ForegroundColor Cyan
Write-Host "Repository: $RepoRoot"
Write-Host "Recognition executed: no"
Write-Host "Official attendance changes allowed: no"
Write-Host "Production changes allowed: no"
Write-Host "Rollback execution allowed: no"

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repository not found: $RepoRoot"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $Cli -PathType Leaf)) {
    throw "Phase 1.2P CLI not found: $Cli"
}

$OriginalLocation = Get-Location
$OriginalPythonPath = $env:PYTHONPATH
try {
    Set-Location -LiteralPath $RepoRoot
    $env:PYTHONPATH = $RepoRoot

    $null = Invoke-CheckedProcess -Name "Python compilation" -FilePath $Python -WorkingDirectory $RepoRoot -Arguments @(
        "-m", "py_compile",
        "src\face_attendance\phase_1_2p_post_promotion_verification.py",
        "scripts\run_phase_1_2p_post_promotion_verification.py",
        "tests\test_phase_1_2p_post_promotion_verification.py"
    )

    if (-not $SkipFocusedTests) {
        $null = Invoke-CheckedProcess -Name "Focused Phase 1.2M-O-P lifecycle tests" -FilePath $Python -WorkingDirectory $RepoRoot -Arguments @(
            "-m", "unittest", "-v",
            "tests.test_phase_1_2p_post_promotion_verification",
            "tests.test_phase_1_2o_explicit_promotion",
            "tests.test_phase_1_2m_candidate_family",
            "tests.test_embedding_version_family"
        )
    }
    else {
        Write-Host ""
        Write-Host "[Focused tests] SKIPPED by explicit switch" -ForegroundColor Yellow
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PreflightOutput) | Out-Null
    $Preflight = Invoke-CheckedProcess -Name "Real Phase 1.2P read-only preflight" -FilePath $Python -WorkingDirectory $RepoRoot -Arguments @(
        $Cli, "preflight",
        "--repo-root", $RepoRoot,
        "--output", $PreflightOutput
    )
    if (-not ($Preflight.StdOut | Where-Object { "$_" -eq "PREFLIGHT_STATUS=PASS" })) {
        throw "Phase 1.2P preflight did not report PASS."
    }

    if ($PreflightOnly) {
        Write-Host ""
        Write-Host "Phase 1.2P preflight completed safely." -ForegroundColor Green
        Write-Host "Preflight output: $PreflightOutput"
        Write-Host "Verification executed: no"
        Write-Host "Production embeddings changed: no"
        Write-Host "Production summary changed: no"
        Write-Host "Official attendance changed: no"
        Write-Host "Recognition executed: no"
        Write-Host "PREFLIGHT_STATUS=PASS"
        exit 0
    }

    $Run = Invoke-CheckedProcess -Name "Exact post-promotion operational verification" -FilePath $Python -WorkingDirectory $RepoRoot -Arguments @(
        $Cli, "run",
        "--repo-root", $RepoRoot,
        "--output-root", $OutputRoot
    )
    $OutputLine = $Run.StdOut | Where-Object { "$_" -like "PHASE_1_2P_OUTPUT=*" } | Select-Object -Last 1
    if (-not $OutputLine) {
        throw "Phase 1.2P run did not report its output directory."
    }
    if (-not ($Run.StdOut | Where-Object { "$_" -eq "PHASE_1_2P_STATUS=PASS" })) {
        throw "Phase 1.2P run did not report PASS."
    }
    $OutputDir = "$OutputLine".Substring("PHASE_1_2P_OUTPUT=".Length)

    Write-Host ""
    Write-Host "Phase 1.2P completed safely." -ForegroundColor Green
    Write-Host "Output: $OutputDir"
    Write-Host "Production embeddings changed: no"
    Write-Host "Production summary changed: no"
    Write-Host "Official attendance changed: no"
    Write-Host "Recognition executed: no"
    Write-Host "Datasets changed: no"
    Write-Host "Rollback backup: verified"
    Write-Host "Next action: return to product workflow development."
    Write-Host "PHASE_1_2P_OUTPUT=$OutputDir"

    if (-not $NoOpen) {
        $Report = Join-Path $OutputDir "verification_report.md"
        if (Test-Path -LiteralPath $Report -PathType Leaf) {
            Start-Process $Report
        }
    }
}
finally {
    Set-Location -LiteralPath $OriginalLocation
    if ($null -eq $OriginalPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $OriginalPythonPath
    }
}
