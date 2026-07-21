param(
    [string]$FamilyDir = "F:\sohail\Class_Attendance_YuNet_SFace\models\versions\embfam-7bc431a3ad762398d4e9",
    [string]$FreezeJson = "D:\Downloads\phase_1_2j_mon_p3_source_freeze.json",
    [string]$FreezeCsv = "D:\Downloads\phase_1_2j_mon_p3_source_freeze.csv",
    [string]$CanonicalVideoRoot = "F:\sohail\Class_Attendance_YuNet_SFace\cctv_videos\prepared_slots\2026-06-22\MON_P3",
    [string]$MirrorVideoRoot = "F:\sohail\Class_Attendance_YuNet_SFace\cctv_videos\MON_P3",
    [string]$ReferenceDiagnosticRun = "F:\sohail\Class_Attendance_YuNet_SFace\attendance_output\diagnostics\2026-06-30__B51__P2__CVO_20260712_221105_286649",
    [switch]$PreflightOnly,
    [switch]$SkipFullRegression,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ExpectedFamilyId = "embfam-7bc431a3ad762398d4e9"
$ExpectedEmbeddingHash = "c32ed31df10b7b9b43b8f19977a2adf0a82fb8e7a71f3e8bceb42c0565fedd49"
$ExpectedSummaryHash = "0df35c3bb9207e191aa49dad5536b8378e812d56463491bd7203885dca5ae9ee"

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo "venv\Scripts\python.exe"
$Cli = Join-Path $Repo "scripts\run_phase_1_2j_mon_p3_shadow.py"
$ProductionEmbeddings = Join-Path $Repo "models\student_embeddings.pkl"
$ProductionSummary = Join-Path $Repo "models\embedding_summary.csv"
$StudentMap = Join-Path $Repo "data\student_faculty_map.json"
$Timetable = Join-Path $Repo "timetable_b51_2026_2027.csv"
$OutputRoot = Join-Path $Repo "attendance_output\shadow_validation\phase_1_2j"
$PreflightJson = Join-Path $Repo "models\versions\phase_1_2j_preflight.json"

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host ""
    Write-Host "[$Stage]" -ForegroundColor Cyan
    $StageWatch = [System.Diagnostics.Stopwatch]::StartNew()
    $Output = [System.Collections.Generic.List[string]]::new()
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        # unittest writes successful verbose progress to stderr. Windows
        # PowerShell 5.1 must use the native process exit code rather than
        # treating that normal stream as a terminating error.
        $ErrorActionPreference = "Continue"
        & $Python @Arguments 2>&1 | ForEach-Object {
            $Text = "$_"
            [void]$Output.Add($Text)
            Write-Host $Text
        }
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    $StageWatch.Stop()
    if ($ExitCode -ne 0) {
        throw "$Stage failed with exit code $ExitCode. No later stage was started."
    }
    Write-Host ("$Stage passed in {0:N2} seconds." -f $StageWatch.Elapsed.TotalSeconds) -ForegroundColor Green
    return $Output.ToArray()
}

foreach ($RequiredPath in @(
    $Python,
    $Cli,
    $FamilyDir,
    (Join-Path $FamilyDir "family_manifest.json"),
    (Join-Path $FamilyDir "evaluation\evaluation_decision.json"),
    (Join-Path $FamilyDir "variants\full_candidate\version_manifest.json"),
    (Join-Path $FamilyDir "variants\full_candidate\student_embeddings.pkl"),
    $FreezeJson,
    $FreezeCsv,
    $CanonicalVideoRoot,
    $MirrorVideoRoot,
    $ReferenceDiagnosticRun,
    $ProductionEmbeddings,
    $ProductionSummary,
    $StudentMap,
    $Timetable
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required Phase 1.2J input not found: $RequiredPath"
    }
}

$FamilyDir = (Resolve-Path -LiteralPath $FamilyDir).Path
$FreezeJson = (Resolve-Path -LiteralPath $FreezeJson).Path
$FreezeCsv = (Resolve-Path -LiteralPath $FreezeCsv).Path
$CanonicalVideoRoot = (Resolve-Path -LiteralPath $CanonicalVideoRoot).Path
$MirrorVideoRoot = (Resolve-Path -LiteralPath $MirrorVideoRoot).Path
$ReferenceDiagnosticRun = (Resolve-Path -LiteralPath $ReferenceDiagnosticRun).Path

$FamilyManifest = Get-Content -LiteralPath (Join-Path $FamilyDir "family_manifest.json") -Raw | ConvertFrom-Json
$FamilyDecision = Get-Content -LiteralPath (Join-Path $FamilyDir "evaluation\evaluation_decision.json") -Raw | ConvertFrom-Json
if ($FamilyManifest.family_id -ne $ExpectedFamilyId) {
    throw "Unexpected family ID. Expected $ExpectedFamilyId, found $($FamilyManifest.family_id)"
}
if ($FamilyDecision.decision -ne "built_unapproved_pending_mon_p3") {
    throw "Family is not pending MON_P3 validation: $($FamilyDecision.decision)"
}
if ($FamilyManifest.production_approved -ne $false -or
    $FamilyManifest.candidate_promoted -ne $false -or
    $FamilyManifest.mon_p3_processed -ne $false) {
    throw "Family violates the unapproved untouched-MON_P3 contract."
}

$InitialEmbeddingHash = (Get-FileHash -LiteralPath $ProductionEmbeddings -Algorithm SHA256).Hash.ToLowerInvariant()
$InitialSummaryHash = (Get-FileHash -LiteralPath $ProductionSummary -Algorithm SHA256).Hash.ToLowerInvariant()
if ($InitialEmbeddingHash -ne $ExpectedEmbeddingHash) {
    throw "Production embedding hash mismatch before Phase 1.2J. Expected $ExpectedEmbeddingHash, found $InitialEmbeddingHash"
}
if ($InitialSummaryHash -ne $ExpectedSummaryHash) {
    throw "Production summary hash mismatch before Phase 1.2J. Expected $ExpectedSummaryHash, found $InitialSummaryHash"
}

Write-Host "Phase 1.2J - guarded MON_P3 production-vs-candidate shadow validation" -ForegroundColor Cyan
Write-Host "Repository: $Repo"
Write-Host "Family: $FamilyDir"
Write-Host "Canonical MON_P3 source: $CanonicalVideoRoot"
Write-Host "Mirror MON_P3 source: $MirrorVideoRoot"
Write-Host "Source-freeze JSON: $FreezeJson"
Write-Host "Source-freeze CSV: $FreezeCsv"
Write-Host "Reference diagnostic: $ReferenceDiagnosticRun"
Write-Host "Candidate mode: immutable unpromoted Variant D"
Write-Host "Official attendance writes: disabled"
Write-Host "Manual review scope: material deltas only"
Write-Host "Preflight only: $PreflightOnly"
Write-Host "Full regression enabled: $(-not $SkipFullRegression)"

$OverallWatch = [System.Diagnostics.Stopwatch]::StartNew()
$PreviousPythonPath = $env:PYTHONPATH
Push-Location $Repo
try {
    $env:PYTHONPATH = $Repo

    Invoke-CheckedPython -Stage "Python compilation" -Arguments @(
        "-m", "py_compile",
        "src\face_attendance\mon_p3_shadow_validation.py",
        "scripts\run_phase_1_2j_mon_p3_shadow.py"
    ) | Out-Null

    Invoke-CheckedPython -Stage "CLI help smoke test" -Arguments @(
        $Cli, "--help"
    ) | Out-Null

    Invoke-CheckedPython -Stage "Focused Phase 1.2J tests" -Arguments @(
        "-m", "unittest", "-v",
        "tests.test_mon_p3_shadow_validation"
    ) | Out-Null

    Invoke-CheckedPython -Stage "Related recognition and lifecycle regression tests" -Arguments @(
        "-m", "unittest", "-v",
        "tests.test_shadow_validation",
        "tests.test_embedding_version_family",
        "tests.test_recovered_embedding_family",
        "tests.test_session_identity",
        "tests.test_tracklets",
        "tests.test_zones",
        "tests.test_roster_integrity"
    ) | Out-Null

    if (-not $SkipFullRegression) {
        Invoke-CheckedPython -Stage "Full unittest discovery" -Arguments @(
            "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"
        ) | Out-Null
    }

    $CommonArguments = @(
        "--repo-root", $Repo,
        "--family-dir", $FamilyDir,
        "--expected-family-id", $ExpectedFamilyId,
        "--freeze-json", $FreezeJson,
        "--freeze-csv", $FreezeCsv,
        "--canonical-video-root", $CanonicalVideoRoot,
        "--mirror-video-root", $MirrorVideoRoot,
        "--reference-diagnostic-run", $ReferenceDiagnosticRun,
        "--student-map", $StudentMap,
        "--timetable", $Timetable,
        "--production-embeddings", $ProductionEmbeddings,
        "--production-summary", $ProductionSummary,
        "--output-root", $OutputRoot
    )

    $PreflightArguments = @(
        $Cli, "preflight"
    ) + $CommonArguments + @(
        "--output-json", $PreflightJson
    )
    $PreflightOutput = Invoke-CheckedPython -Stage "Immutable MON_P3 shadow preflight" -Arguments $PreflightArguments
    if (-not ($PreflightOutput | Where-Object { "$_" -eq "PREFLIGHT_STATUS=PASS" })) {
        throw "Phase 1.2J preflight exited successfully but did not emit PREFLIGHT_STATUS=PASS."
    }

    if ($PreflightOnly) {
        $FinalEmbeddingHash = (Get-FileHash -LiteralPath $ProductionEmbeddings -Algorithm SHA256).Hash.ToLowerInvariant()
        $FinalSummaryHash = (Get-FileHash -LiteralPath $ProductionSummary -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($FinalEmbeddingHash -ne $InitialEmbeddingHash -or $FinalSummaryHash -ne $InitialSummaryHash) {
            throw "Protected production files changed during Phase 1.2J preflight."
        }
        Write-Host ""
        Write-Host "Phase 1.2J preflight completed safely." -ForegroundColor Green
        Write-Host "Preflight JSON: $PreflightJson"
        Write-Host "Recognition executed: no"
        Write-Host "Official attendance changed: no"
        Write-Host "Candidate promoted: no"
        Write-Host "MON_P3 processed: no"
        Write-Host "PREFLIGHT_OUTPUT=$PreflightJson"
        return
    }

    $RunArguments = @(
        $Cli, "run"
    ) + $CommonArguments
    $RunOutput = Invoke-CheckedPython -Stage "MON_P3 dual diagnostic-only shadow run" -Arguments $RunArguments
    $OutputLine = $RunOutput | Where-Object { "$_" -like "PHASE_1_2J_OUTPUT=*" } | Select-Object -Last 1
    if (-not $OutputLine) {
        throw "The successful Phase 1.2J run did not report PHASE_1_2J_OUTPUT."
    }
    $PhaseOutput = ("$OutputLine").Substring("PHASE_1_2J_OUTPUT=".Length)
    if (-not (Test-Path -LiteralPath $PhaseOutput -PathType Container)) {
        throw "Reported Phase 1.2J output does not exist: $PhaseOutput"
    }

    Invoke-CheckedPython -Stage "Final Phase 1.2J integrity verification" -Arguments @(
        $Cli, "verify",
        "--output-dir", $PhaseOutput,
        "--family-dir", $FamilyDir,
        "--production-embeddings", $ProductionEmbeddings,
        "--production-summary", $ProductionSummary
    ) | Out-Null

    $FinalEmbeddingHash = (Get-FileHash -LiteralPath $ProductionEmbeddings -Algorithm SHA256).Hash.ToLowerInvariant()
    $FinalSummaryHash = (Get-FileHash -LiteralPath $ProductionSummary -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($FinalEmbeddingHash -ne $InitialEmbeddingHash) {
        throw "Production embeddings changed during Phase 1.2J."
    }
    if ($FinalSummaryHash -ne $InitialSummaryHash) {
        throw "Production summary changed during Phase 1.2J."
    }

    $SummaryPath = Join-Path $PhaseOutput "shadow_run_summary.json"
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    $OverallWatch.Stop()
    Write-Host ""
    Write-Host "Phase 1.2J completed safely." -ForegroundColor Green
    Write-Host ("Elapsed: {0:N2} seconds" -f $OverallWatch.Elapsed.TotalSeconds)
    Write-Host "Exact output: $PhaseOutput"
    Write-Host "Decision: $($Summary.decision)"
    Write-Host "Material exception tracklets: $($Summary.review_tracklets)"
    Write-Host "Blind review package: $($Summary.review_package)"
    Write-Host "Production embeddings changed: no"
    Write-Host "Production summary changed: no"
    Write-Host "Datasets changed: no"
    Write-Host "Official attendance changed: no"
    Write-Host "Candidate promoted: no"
    Write-Host "MON_P3 processed: yes"
    Write-Host "PHASE_1_2J_OUTPUT=$PhaseOutput"
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}
