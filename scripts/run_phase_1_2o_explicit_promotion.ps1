param(
    [string]$RepoRoot = "F:\sohail\Class_Attendance_YuNet_SFace",
    [string]$LabelsPath = "D:\Downloads\tracklet_review_labels_mon-p4-shadow-695277fbd0dcd9d431e1-20260717152714-915b9c.csv",
    [switch]$PreflightOnly,
    [string]$ConfirmPromotion = "",
    [switch]$Rollback,
    [string]$PromotionDir = "",
    [string]$ConfirmRollback = "",
    [switch]$SkipFocusedTests,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
$Cli = Join-Path $RepoRoot "scripts\run_phase_1_2o_explicit_promotion.py"
$PreflightOutput = Join-Path $RepoRoot "attendance_output\embedding_forensics\phase_1_2o\phase_1_2o_promotion_preflight.json"
$ExpectedFamily = "embfam-274b5207b8b71294ff75"

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

Write-Host "Phase 1.2O explicit embedding-family promotion and rollback" -ForegroundColor Cyan
Write-Host "Repository: $RepoRoot"
Write-Host "Candidate family: $ExpectedFamily"
Write-Host "Recognition executed: no"
Write-Host "Official attendance changes allowed: no"
Write-Host "Datasets changes allowed: no"

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repository not found: $RepoRoot"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $Cli -PathType Leaf)) {
    throw "Phase 1.2O CLI not found: $Cli"
}

$Modes = 0
if ($PreflightOnly) { $Modes++ }
if ($ConfirmPromotion) { $Modes++ }
if ($Rollback) { $Modes++ }
if ($Modes -ne 1) {
    throw "Choose exactly one mode: -PreflightOnly, -ConfirmPromotion <family-id>, or -Rollback."
}
if ($Rollback) {
    if (-not $PromotionDir -or -not $ConfirmRollback) {
        throw "Rollback requires -PromotionDir and -ConfirmRollback."
    }
}
else {
    if (-not (Test-Path -LiteralPath $LabelsPath -PathType Leaf)) {
        throw "Completed MON_P4 blind-review labels not found: $LabelsPath"
    }
}

$OriginalLocation = Get-Location
$OriginalPythonPath = $env:PYTHONPATH
try {
    Set-Location -LiteralPath $RepoRoot
    $env:PYTHONPATH = $RepoRoot

    $null = Invoke-CheckedProcess -Name "Python compilation" -FilePath $Python -WorkingDirectory $RepoRoot -Arguments @(
        "-m", "py_compile",
        "src\face_attendance\phase_1_2o_explicit_promotion.py",
        "scripts\run_phase_1_2o_explicit_promotion.py",
        "tests\test_phase_1_2o_explicit_promotion.py"
    )

    if (-not $SkipFocusedTests) {
        $null = Invoke-CheckedProcess -Name "Focused Phase 1.2M-N-O lifecycle tests" -FilePath $Python -WorkingDirectory $RepoRoot -Arguments @(
            "-m", "unittest", "-v",
            "tests.test_phase_1_2o_explicit_promotion",
            "tests.test_mon_p4_shadow_validation",
            "tests.test_phase_1_2m_candidate_family",
            "tests.test_embedding_version_family"
        )
    }
    else {
        Write-Host ""
        Write-Host "[Focused tests] SKIPPED by explicit switch" -ForegroundColor Yellow
    }

    if ($PreflightOnly) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PreflightOutput) | Out-Null
        $Result = Invoke-CheckedProcess -Name "Real Phase 1.2O promotion preflight" -FilePath $Python -WorkingDirectory $RepoRoot -Arguments @(
            $Cli, "preflight",
            "--repo-root", $RepoRoot,
            "--labels", $LabelsPath,
            "--output", $PreflightOutput
        )
        if (-not ($Result.StdOut | Where-Object { "$_" -eq "PREFLIGHT_STATUS=PASS" })) {
            throw "Phase 1.2O preflight did not report PASS."
        }
        Write-Host ""
        Write-Host "Phase 1.2O preflight completed safely." -ForegroundColor Green
        Write-Host "Preflight output: $PreflightOutput"
        Write-Host "Production embeddings changed: no"
        Write-Host "Production summary changed: no"
        Write-Host "Official attendance changed: no"
        Write-Host "Candidate promoted: no"
        Write-Host "PHASE_1_2O_PREFLIGHT=$PreflightOutput"
        exit 0
    }

    if ($Rollback) {
        $RollbackResult = Invoke-CheckedProcess -Name "Explicit Phase 1.2O rollback" -FilePath $Python -WorkingDirectory $RepoRoot -Arguments @(
            $Cli, "rollback",
            "--repo-root", $RepoRoot,
            "--promotion-dir", $PromotionDir,
            "--confirm-promotion-id", $ConfirmRollback
        )
        if (-not ($RollbackResult.StdOut | Where-Object { "$_" -eq "PHASE_1_2O_ROLLBACK_STATUS=PASS" })) {
            throw "Phase 1.2O rollback did not report PASS."
        }
        Write-Host ""
        Write-Host "Phase 1.2O rollback and verification completed safely." -ForegroundColor Green
        exit 0
    }

    if ($ConfirmPromotion -ne $ExpectedFamily) {
        throw "-ConfirmPromotion must exactly equal $ExpectedFamily"
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PreflightOutput) | Out-Null
    $Preflight = Invoke-CheckedProcess -Name "Final read-only Phase 1.2O preflight" -FilePath $Python -WorkingDirectory $RepoRoot -Arguments @(
        $Cli, "preflight",
        "--repo-root", $RepoRoot,
        "--labels", $LabelsPath,
        "--output", $PreflightOutput
    )
    if (-not ($Preflight.StdOut | Where-Object { "$_" -eq "PREFLIGHT_STATUS=PASS" })) {
        throw "Final Phase 1.2O preflight did not report PASS."
    }

    $Promotion = Invoke-CheckedProcess -Name "Atomic Phase 1.2O promotion" -FilePath $Python -WorkingDirectory $RepoRoot -Arguments @(
        $Cli, "promote",
        "--repo-root", $RepoRoot,
        "--labels", $LabelsPath,
        "--confirm-family-id", $ConfirmPromotion
    )
    $PromotionLine = $Promotion.StdOut | Where-Object { "$_" -like "PHASE_1_2O_PROMOTION=*" } | Select-Object -Last 1
    if (-not $PromotionLine) {
        throw "Promotion command did not report its output directory."
    }
    $ActualPromotionDir = "$PromotionLine".Substring("PHASE_1_2O_PROMOTION=".Length)

    $Verify = Invoke-CheckedProcess -Name "Final promoted-state and rollback-backup verification" -FilePath $Python -WorkingDirectory $RepoRoot -Arguments @(
        $Cli, "verify",
        "--repo-root", $RepoRoot,
        "--promotion-dir", $ActualPromotionDir
    )
    if (-not ($Verify.StdOut | Where-Object { "$_" -eq "PHASE_1_2O_VERIFY_STATUS=PASS" })) {
        throw "Final Phase 1.2O promoted-state verification did not report PASS."
    }

    Write-Host ""
    Write-Host "Phase 1.2O completed safely." -ForegroundColor Green
    Write-Host "Promotion output: $ActualPromotionDir"
    Write-Host "Production family: $ExpectedFamily"
    Write-Host "Production embeddings changed: yes, to the verified candidate"
    Write-Host "Production summary changed: yes, to the verified candidate"
    Write-Host "Official attendance changed: no"
    Write-Host "Recognition executed: no"
    Write-Host "Datasets changed: no"
    Write-Host "Rollback backup: verified"
    Write-Host "Candidate promoted: yes"
    Write-Host "PHASE_1_2O_OUTPUT=$ActualPromotionDir"
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
