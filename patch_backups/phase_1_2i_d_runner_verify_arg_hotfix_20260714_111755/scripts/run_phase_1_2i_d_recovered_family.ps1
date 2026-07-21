param(
    [string]$EnrollmentApprovals = "D:\Downloads\forensic_review_approvals_compact-enrollment-audit-df0b986b716a2cd5.csv",
    [string]$CctvApprovals = "D:\Downloads\forensic_review_approvals_compact-verified-cctv-8785d6b28789ae6f.csv",
    [string]$RecoveryDir = "F:\sohail\Class_Attendance_YuNet_SFace\models\versions\recovery_runs\cctv-recovery-69e11ea931f79330626d",
    [string]$FamilyId = "",
    [switch]$PreflightOnly,
    [switch]$SkipFullRegression,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ExpectedEmbeddingHash = "c32ed31df10b7b9b43b8f19977a2adf0a82fb8e7a71f3e8bceb42c0565fedd49"
$ExpectedSummaryHash = "0df35c3bb9207e191aa49dad5536b8378e812d56463491bd7203885dca5ae9ee"
$ExpectedRecoveryId = "cctv-recovery-69e11ea931f79330626d"

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo "venv\Scripts\python.exe"
$Cli = Join-Path $Repo "scripts\validate_tracklet_ground_truth.py"
$RecoveryCli = Join-Path $Repo "scripts\recover_tue_p2_same_track_evidence.py"
$ForensicRun = Join-Path $Repo "attendance_output\embedding_forensics\embedding_forensics_reviewfix_20260713_140159_455110"
$CompactRun = Join-Path $Repo "attendance_output\embedding_forensics\embedding_forensics_compact_review_20260713_161633_550718"
$EnrollmentPackage = Join-Path $CompactRun "compact_enrollment_review"
$CctvPackage = Join-Path $CompactRun "compact_cctv_review"
$VersionsRoot = Join-Path $Repo "models\versions"
$PreflightJson = Join-Path $VersionsRoot "phase_1_2i_d_preflight.json"
$ProductionEmbeddings = Join-Path $Repo "models\student_embeddings.pkl"
$ProductionSummary = Join-Path $Repo "models\embedding_summary.csv"

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
        # PowerShell 5.1 must rely on the native process exit code instead of
        # turning that normal stream into a terminating error.
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
    $RecoveryCli,
    $ForensicRun,
    $EnrollmentPackage,
    $CctvPackage,
    $EnrollmentApprovals,
    $CctvApprovals,
    $RecoveryDir,
    (Join-Path $RecoveryDir "recovery_manifest.json"),
    (Join-Path $RecoveryDir "same_track_recovery_audit.csv"),
    (Join-Path $RecoveryDir "recovered_embeddings.npz"),
    (Join-Path $RecoveryDir "output_manifest.json"),
    $ProductionEmbeddings,
    $ProductionSummary
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required Phase 1.2I-D input not found: $RequiredPath"
    }
}

$EnrollmentApprovals = (Resolve-Path -LiteralPath $EnrollmentApprovals).Path
$CctvApprovals = (Resolve-Path -LiteralPath $CctvApprovals).Path
$RecoveryDir = (Resolve-Path -LiteralPath $RecoveryDir).Path
New-Item -ItemType Directory -Path $VersionsRoot -Force | Out-Null

$RecoveryManifest = Get-Content -LiteralPath (Join-Path $RecoveryDir "recovery_manifest.json") -Raw | ConvertFrom-Json
if ($RecoveryManifest.recovery_id -ne $ExpectedRecoveryId) {
    throw "Unexpected recovery ID. Expected $ExpectedRecoveryId, found $($RecoveryManifest.recovery_id)"
}
if ($RecoveryManifest.status -ne "recovery_available") {
    throw "Recovery bundle is not available: $($RecoveryManifest.status)"
}

Write-Host "Phase 1.2I-D - recovered immutable candidate-family workflow" -ForegroundColor Cyan
Write-Host "Repository: $Repo"
Write-Host "Python: $Python"
Write-Host "Frozen benchmark: $ForensicRun"
Write-Host "Enrollment approvals: $EnrollmentApprovals"
Write-Host "CCTV approvals: $CctvApprovals"
Write-Host "Recovery bundle: $RecoveryDir"
Write-Host "Recovery ID: $($RecoveryManifest.recovery_id)"
Write-Host "Recovery selected vectors preserved: $($RecoveryManifest.selected_embedding_count)"
Write-Host "Candidate consumption policy: one medoid per reviewed track"
Write-Host "Manual reviews repeated: no"
Write-Host "Output root: $VersionsRoot"
Write-Host "Preflight only: $PreflightOnly"
Write-Host "Full regression enabled: $(-not $SkipFullRegression)"

$InitialEmbeddingHash = (Get-FileHash -LiteralPath $ProductionEmbeddings -Algorithm SHA256).Hash.ToLowerInvariant()
$InitialSummaryHash = (Get-FileHash -LiteralPath $ProductionSummary -Algorithm SHA256).Hash.ToLowerInvariant()
if ($InitialEmbeddingHash -ne $ExpectedEmbeddingHash) {
    throw "Production embedding hash mismatch before tests. Expected $ExpectedEmbeddingHash, found $InitialEmbeddingHash"
}
if ($InitialSummaryHash -ne $ExpectedSummaryHash) {
    throw "Production summary hash mismatch before tests. Expected $ExpectedSummaryHash, found $InitialSummaryHash"
}

$OverallWatch = [System.Diagnostics.Stopwatch]::StartNew()
$PreviousPythonPath = $env:PYTHONPATH
Push-Location $Repo
try {
    $env:PYTHONPATH = $Repo

    Invoke-CheckedPython -Stage "Python compilation" -Arguments @(
        "-m", "py_compile",
        "src\face_attendance\embedding_version_family.py",
        "src\face_attendance\cctv_same_track_recovery.py",
        "scripts\validate_tracklet_ground_truth.py",
        "scripts\recover_tue_p2_same_track_evidence.py"
    ) | Out-Null

    Invoke-CheckedPython -Stage "CLI help smoke test" -Arguments @(
        $Cli, "--help"
    ) | Out-Null

    Invoke-CheckedPython -Stage "Immutable recovery-bundle verification" -Arguments @(
        $RecoveryCli,
        "--verify-only",
        "--family-dir", $RecoveryDir
    ) | Out-Null

    Invoke-CheckedPython -Stage "Focused recovered-family tests" -Arguments @(
        "-m", "unittest", "-v",
        "tests.test_recovered_embedding_family",
        "tests.test_cctv_same_track_recovery",
        "tests.test_embedding_version_family"
    ) | Out-Null

    Invoke-CheckedPython -Stage "Core lifecycle regression tests" -Arguments @(
        "-m", "unittest", "-v",
        "tests.test_compact_forensic_review",
        "tests.test_forensic_review",
        "tests.test_embedding_forensics",
        "tests.test_embedding_lifecycle",
        "tests.test_shadow_validation",
        "tests.test_tracklet_calibration",
        "tests.test_tracklet_review",
        "tests.test_recall_analysis",
        "tests.test_roster_integrity"
    ) | Out-Null

    if (-not $SkipFullRegression) {
        Invoke-CheckedPython -Stage "Full unittest discovery" -Arguments @(
            "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"
        ) | Out-Null
    }

    $CommonFamilyArguments = @(
        "--forensic-run", $ForensicRun,
        "--enrollment-review-package", $EnrollmentPackage,
        "--enrollment-approvals", $EnrollmentApprovals,
        "--cctv-review-package", $CctvPackage,
        "--cctv-approvals", $CctvApprovals,
        "--production-embeddings", $ProductionEmbeddings,
        "--production-summary", $ProductionSummary,
        "--student-map", (Join-Path $Repo "data\student_faculty_map.json"),
        "--dataset", (Join-Path $Repo "dataset"),
        "--augmented-dataset", (Join-Path $Repo "augmented_dataset"),
        "--versions-root", $VersionsRoot,
        "--recovery-dir", $RecoveryDir
    )

    $PreflightArguments = @(
        $Cli,
        "preflight-embedding-family"
    ) + $CommonFamilyArguments + @(
        "--output-json", $PreflightJson
    )
    $PreflightOutput = Invoke-CheckedPython -Stage "Read-only recovered-family preflight" -Arguments $PreflightArguments
    if (-not ($PreflightOutput | Where-Object { "$_" -eq "PREFLIGHT_STATUS=PASS" })) {
        throw "Recovered-family preflight exited successfully but did not emit PREFLIGHT_STATUS=PASS."
    }

    if ($PreflightOnly) {
        $FinalEmbeddingHash = (Get-FileHash -LiteralPath $ProductionEmbeddings -Algorithm SHA256).Hash.ToLowerInvariant()
        $FinalSummaryHash = (Get-FileHash -LiteralPath $ProductionSummary -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($FinalEmbeddingHash -ne $InitialEmbeddingHash -or $FinalSummaryHash -ne $InitialSummaryHash) {
            throw "Protected production files changed during recovered-family preflight."
        }
        Write-Host ""
        Write-Host "Phase 1.2I-D preflight completed safely." -ForegroundColor Green
        Write-Host "Preflight JSON: $PreflightJson"
        Write-Host "Candidate family built: no"
        Write-Host "Manual reviews repeated: no"
        Write-Host "Production promotion performed: no"
        Write-Host "MON_P3 processed: no"
        Write-Host "PREFLIGHT_OUTPUT=$PreflightJson"
        return
    }

    $BuildArguments = @(
        $Cli,
        "build-evaluate-embedding-family"
    ) + $CommonFamilyArguments
    if ($FamilyId) {
        $BuildArguments += @("--family-id", $FamilyId)
    }

    $BuildOutput = Invoke-CheckedPython -Stage "Recovered candidate-family build and leakage-safe evaluation" -Arguments $BuildArguments
    $FamilyLine = $BuildOutput | Where-Object { "$_" -like "FAMILY_OUTPUT=*" } | Select-Object -Last 1
    if (-not $FamilyLine) {
        throw "The successful recovered-family build did not report FAMILY_OUTPUT."
    }
    $FamilyFolder = ("$FamilyLine").Substring("FAMILY_OUTPUT=".Length)
    if (-not (Test-Path -LiteralPath $FamilyFolder -PathType Container)) {
        throw "The reported recovered-family output folder does not exist: $FamilyFolder"
    }

    Invoke-CheckedPython -Stage "Final recovered-family integrity verification" -Arguments @(
        $Cli,
        "verify-embedding-family",
        "--family-dir", $FamilyFolder,
        "--production-embeddings", $ProductionEmbeddings,
        "--production-summary", $ProductionSummary
    ) | Out-Null

    $FinalEmbeddingHash = (Get-FileHash -LiteralPath $ProductionEmbeddings -Algorithm SHA256).Hash.ToLowerInvariant()
    $FinalSummaryHash = (Get-FileHash -LiteralPath $ProductionSummary -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($FinalEmbeddingHash -ne $InitialEmbeddingHash) {
        throw "Production embeddings changed during Phase 1.2I-D."
    }
    if ($FinalSummaryHash -ne $InitialSummaryHash) {
        throw "Production summary changed during Phase 1.2I-D."
    }

    $DecisionPath = Join-Path $FamilyFolder "evaluation\evaluation_decision.json"
    if (-not (Test-Path -LiteralPath $DecisionPath -PathType Leaf)) {
        throw "Evaluation decision not found: $DecisionPath"
    }
    $Decision = Get-Content -LiteralPath $DecisionPath -Raw | ConvertFrom-Json
    $FamilyManifest = Get-Content -LiteralPath (Join-Path $FamilyFolder "family_manifest.json") -Raw | ConvertFrom-Json

    Write-Host ""
    Write-Host "Phase 1.2I-D completed safely." -ForegroundColor Green
    Write-Host ("Elapsed: {0:N2} seconds" -f $OverallWatch.Elapsed.TotalSeconds)
    Write-Host "Exact family output: $FamilyFolder"
    Write-Host "Family ID: $($FamilyManifest.family_id)"
    Write-Host "Decision: $($Decision.decision)"
    Write-Host "Recovered medoids consumed: $($FamilyManifest.same_track_recovered_medoid_count)"
    Write-Host "Variant B status: $($FamilyManifest.variants.B.status)"
    Write-Host "Variant B embeddings: $($FamilyManifest.variants.B.embedding_records)"
    Write-Host "Next step: $($Decision.exact_next_step)"
    Write-Host "Production embeddings SHA-256: $FinalEmbeddingHash"
    Write-Host "Production summary SHA-256: $FinalSummaryHash"
    Write-Host "Manual reviews repeated: no"
    Write-Host "Production promotion performed: no"
    Write-Host "MON_P3 processed: no"
    Write-Host "FAMILY_OUTPUT=$FamilyFolder"

    if (-not $NoOpen) {
        Invoke-Item -LiteralPath $FamilyFolder
    }
} finally {
    if ($null -eq $PreviousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $PreviousPythonPath
    }
    Pop-Location
    $OverallWatch.Stop()
}
