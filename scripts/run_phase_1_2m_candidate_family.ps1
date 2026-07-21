param(
    [string]$EnrollmentApprovals = "D:\Downloads\forensic_review_approvals_compact-enrollment-audit-df0b986b716a2cd5.csv",
    [string]$CctvApprovals = "D:\Downloads\forensic_review_approvals_compact-verified-cctv-8785d6b28789ae6f.csv",
    [string]$RecoveryDir = "F:\sohail\Class_Attendance_YuNet_SFace\models\versions\recovery_runs\cctv-recovery-69e11ea931f79330626d",
    [string]$SourceAblationDir = "F:\sohail\Class_Attendance_YuNet_SFace\attendance_output\embedding_forensics\phase_1_2l\source-ablation-da12b41f85bc813a6de1",
    [string]$SelectionAuditDir = "F:\sohail\Class_Attendance_YuNet_SFace\attendance_output\embedding_forensics\phase_1_2l_1\selection-audit-bacb195efc6751ca2c22",
    [string]$ParentFamilyDir = "F:\sohail\Class_Attendance_YuNet_SFace\models\versions\embfam-7bc431a3ad762398d4e9",
    [string]$FamilyId = "",
    [switch]$PreflightOnly,
    [switch]$SkipFullRegression,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ExpectedEmbeddingHash = "c32ed31df10b7b9b43b8f19977a2adf0a82fb8e7a71f3e8bceb42c0565fedd49"
$ExpectedSummaryHash = "0df35c3bb9207e191aa49dad5536b8378e812d56463491bd7203885dca5ae9ee"
$ExpectedSourceAblationManifestHash = "2a34c5c107e4966fdfafed8a2a2e5d5e2bae1e5ab2a914116c97eacad82c54a3"
$ExpectedSelectionAuditManifestHash = "7baa68fb901e8af8e3bd420282b1d84d96e79382bb43a24a65ea3a91058ac410"
$ExpectedSelectionDecisionHash = "97de993665df3c39dc59f38c7fca30009be1b8f4ae86c06f5ebf9b6b98ceb7d8"
$ExpectedParentFamilyManifestHash = "74e2f134f3de52e3dcb7f58b54d8624eedf2888cf9a998b5b0d15b24acbf25dc"
$ExpectedRecoveryManifestHash = "45894c17d3639a54c2fe6198884e29201fe9c5926d2ea9307c1b1a14b3c12bdb"

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo "venv\Scripts\python.exe"
$Cli = Join-Path $Repo "scripts\run_phase_1_2m_candidate_family.py"
$ForensicRun = Join-Path $Repo "attendance_output\embedding_forensics\embedding_forensics_reviewfix_20260713_140159_455110"
$CompactRun = Join-Path $Repo "attendance_output\embedding_forensics\embedding_forensics_compact_review_20260713_161633_550718"
$EnrollmentPackage = Join-Path $CompactRun "compact_enrollment_review"
$CctvPackage = Join-Path $CompactRun "compact_cctv_review"
$VersionsRoot = Join-Path $Repo "models\versions"
$ProductionEmbeddings = Join-Path $Repo "models\student_embeddings.pkl"
$ProductionSummary = Join-Path $Repo "models\embedding_summary.csv"
$StudentMap = Join-Path $Repo "data\student_faculty_map.json"
$Dataset = Join-Path $Repo "dataset"
$AugmentedDataset = Join-Path $Repo "augmented_dataset"

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
        # PowerShell 5.1 must judge the native exit code, not the output stream.
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

function Assert-FileHash {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected) {
        throw "$Label hash mismatch. Expected $Expected, found $Actual"
    }
}

foreach ($RequiredPath in @(
    $Python,
    $Cli,
    $ForensicRun,
    $EnrollmentPackage,
    $CctvPackage,
    $EnrollmentApprovals,
    $CctvApprovals,
    $RecoveryDir,
    $SourceAblationDir,
    $SelectionAuditDir,
    $ParentFamilyDir,
    $ProductionEmbeddings,
    $ProductionSummary,
    $StudentMap,
    $Dataset,
    $AugmentedDataset,
    (Join-Path $SourceAblationDir "output_manifest.json"),
    (Join-Path $SelectionAuditDir "output_manifest.json"),
    (Join-Path $SelectionAuditDir "selection_decision.json"),
    (Join-Path $ParentFamilyDir "output_manifest.json"),
    (Join-Path $RecoveryDir "output_manifest.json")
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required Phase 1.2M input not found: $RequiredPath"
    }
}

$EnrollmentApprovals = (Resolve-Path -LiteralPath $EnrollmentApprovals).Path
$CctvApprovals = (Resolve-Path -LiteralPath $CctvApprovals).Path
$RecoveryDir = (Resolve-Path -LiteralPath $RecoveryDir).Path
$SourceAblationDir = (Resolve-Path -LiteralPath $SourceAblationDir).Path
$SelectionAuditDir = (Resolve-Path -LiteralPath $SelectionAuditDir).Path
$ParentFamilyDir = (Resolve-Path -LiteralPath $ParentFamilyDir).Path

Assert-FileHash -Path $ProductionEmbeddings -Expected $ExpectedEmbeddingHash -Label "Production embeddings"
Assert-FileHash -Path $ProductionSummary -Expected $ExpectedSummaryHash -Label "Production summary"
Assert-FileHash -Path (Join-Path $SourceAblationDir "output_manifest.json") -Expected $ExpectedSourceAblationManifestHash -Label "Phase 1.2L output manifest"
Assert-FileHash -Path (Join-Path $SelectionAuditDir "output_manifest.json") -Expected $ExpectedSelectionAuditManifestHash -Label "Phase 1.2L.1 output manifest"
Assert-FileHash -Path (Join-Path $SelectionAuditDir "selection_decision.json") -Expected $ExpectedSelectionDecisionHash -Label "Phase 1.2L.1 selection decision"
Assert-FileHash -Path (Join-Path $ParentFamilyDir "output_manifest.json") -Expected $ExpectedParentFamilyManifestHash -Label "Parent family output manifest"
Assert-FileHash -Path (Join-Path $RecoveryDir "output_manifest.json") -Expected $ExpectedRecoveryManifestHash -Label "Recovery output manifest"

Write-Host "Phase 1.2M - corrected immutable candidate-family workflow" -ForegroundColor Cyan
Write-Host "Repository: $Repo"
Write-Host "Corrected composition: ABL-16-ee3a4ff6"
Write-Host "Kept implicated source: 0140 CP1 recovered medoid"
Write-Host "Removed implicated sources: three 0121 P1 crops and 0140 CP2 medoid"
Write-Host "Full MON_P3 recognition repeated: no"
Write-Host "Manual review repeated: no"
Write-Host "Production changes allowed: no"
Write-Host "Final promotion session: a different untouched session is required"

$InitialEmbeddingHash = (Get-FileHash -LiteralPath $ProductionEmbeddings -Algorithm SHA256).Hash.ToLowerInvariant()
$InitialSummaryHash = (Get-FileHash -LiteralPath $ProductionSummary -Algorithm SHA256).Hash.ToLowerInvariant()
$OverallWatch = [System.Diagnostics.Stopwatch]::StartNew()
$PreviousPythonPath = $env:PYTHONPATH
Push-Location $Repo
try {
    $env:PYTHONPATH = $Repo

    Invoke-CheckedPython -Stage "Python compilation" -Arguments @(
        "-m", "py_compile",
        "src\face_attendance\embedding_version_family.py",
        "src\face_attendance\phase_1_2m_candidate_family.py",
        "scripts\run_phase_1_2m_candidate_family.py"
    ) | Out-Null

    Invoke-CheckedPython -Stage "CLI help smoke test" -Arguments @(
        $Cli, "--help"
    ) | Out-Null

    Invoke-CheckedPython -Stage "Focused Phase 1.2K-L-M tests" -Arguments @(
        "-m", "unittest", "-v",
        "tests.test_phase_1_2m_candidate_family",
        "tests.test_mon_p3_source_ablation",
        "tests.test_mon_p3_retention_attribution",
        "tests.test_embedding_version_family",
        "tests.test_recovered_embedding_family",
        "tests.test_cctv_same_track_recovery"
    ) | Out-Null

    if (-not $SkipFullRegression) {
        Invoke-CheckedPython -Stage "Full unittest discovery" -Arguments @(
            "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"
        ) | Out-Null
    }

    $Common = @(
        $Cli,
        "--repo-root", $Repo,
        "--forensic-run", $ForensicRun,
        "--enrollment-review-package", $EnrollmentPackage,
        "--enrollment-approvals", $EnrollmentApprovals,
        "--cctv-review-package", $CctvPackage,
        "--cctv-approvals", $CctvApprovals,
        "--production-embeddings", $ProductionEmbeddings,
        "--production-summary", $ProductionSummary,
        "--student-map", $StudentMap,
        "--dataset", $Dataset,
        "--augmented-dataset", $AugmentedDataset,
        "--versions-root", $VersionsRoot,
        "--recovery-dir", $RecoveryDir,
        "--source-ablation-dir", $SourceAblationDir,
        "--selection-audit-dir", $SelectionAuditDir,
        "--parent-family-dir", $ParentFamilyDir
    )

    $PreflightOutput = Invoke-CheckedPython -Stage "Real read-only Phase 1.2M preflight" -Arguments ($Common + @("--preflight-only"))
    if (-not ($PreflightOutput | Where-Object { "$_" -eq "PREFLIGHT_STATUS=PASS" })) {
        throw "Phase 1.2M preflight exited successfully but did not emit PREFLIGHT_STATUS=PASS."
    }

    if ($PreflightOnly) {
        Write-Host ""
        Write-Host "Phase 1.2M preflight completed safely." -ForegroundColor Green
        Write-Host "Candidate family built: no"
        Write-Host "Production embeddings changed: no"
        Write-Host "Official attendance changed: no"
        Write-Host "Candidate promoted: no"
        return
    }

    $BuildArgs = $Common
    if ($FamilyId) {
        $BuildArgs += @("--family-id", $FamilyId)
    }
    $BuildOutput = Invoke-CheckedPython -Stage "Corrected immutable family build and exact ABL-16 validation" -Arguments $BuildArgs
    $FamilyLine = $BuildOutput | Where-Object { "$_" -like "FAMILY_OUTPUT=*" } | Select-Object -Last 1
    $ValidationLine = $BuildOutput | Where-Object { "$_" -like "PHASE_1_2M_VALIDATION_OUTPUT=*" } | Select-Object -Last 1
    if (-not $FamilyLine -or -not $ValidationLine) {
        throw "Phase 1.2M build did not report both family and validation outputs."
    }
    $FamilyFolder = ("$FamilyLine").Substring("FAMILY_OUTPUT=".Length)
    $ValidationFolder = ("$ValidationLine").Substring("PHASE_1_2M_VALIDATION_OUTPUT=".Length)
    if (-not (Test-Path -LiteralPath $FamilyFolder -PathType Container)) {
        throw "Reported Phase 1.2M family folder does not exist: $FamilyFolder"
    }
    if (-not (Test-Path -LiteralPath $ValidationFolder -PathType Container)) {
        throw "Reported Phase 1.2M validation folder does not exist: $ValidationFolder"
    }

    Invoke-CheckedPython -Stage "Final Phase 1.2M family and ABL-16 verification" -Arguments ($Common + @("--verify-family", $FamilyFolder)) | Out-Null

    $FinalEmbeddingHash = (Get-FileHash -LiteralPath $ProductionEmbeddings -Algorithm SHA256).Hash.ToLowerInvariant()
    $FinalSummaryHash = (Get-FileHash -LiteralPath $ProductionSummary -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($FinalEmbeddingHash -ne $InitialEmbeddingHash) {
        throw "Production embeddings changed during Phase 1.2M."
    }
    if ($FinalSummaryHash -ne $InitialSummaryHash) {
        throw "Production summary changed during Phase 1.2M."
    }

    $FamilyManifest = Get-Content -LiteralPath (Join-Path $FamilyFolder "family_manifest.json") -Raw | ConvertFrom-Json
    $Decision = Get-Content -LiteralPath (Join-Path $FamilyFolder "evaluation\evaluation_decision.json") -Raw | ConvertFrom-Json
    $Validation = Get-Content -LiteralPath (Join-Path $ValidationFolder "validation_summary.json") -Raw | ConvertFrom-Json

    Write-Host ""
    Write-Host "Phase 1.2M completed safely." -ForegroundColor Green
    Write-Host ("Elapsed: {0:N2} seconds" -f $OverallWatch.Elapsed.TotalSeconds)
    Write-Host "Family output: $FamilyFolder"
    Write-Host "Family ID: $($FamilyManifest.family_id)"
    Write-Host "Decision: $($Decision.decision)"
    Write-Host "Variant A/B/C/D records: $($FamilyManifest.variants.A.embedding_records)/$($FamilyManifest.variants.B.embedding_records)/$($FamilyManifest.variants.C.embedding_records)/$($FamilyManifest.variants.D.embedding_records)"
    Write-Host "Exact ABL-16 validation: $($Validation.decision)"
    Write-Host "Validation output: $ValidationFolder"
    Write-Host "Production embeddings changed: no"
    Write-Host "Official attendance changed: no"
    Write-Host "Candidate promoted: no"
    Write-Host "Full MON_P3 recognition repeated: no"
    Write-Host "Manual review repeated: no"
    Write-Host "Next action: reserve a different untouched CVO validation session."
    Write-Host "FAMILY_OUTPUT=$FamilyFolder"
    Write-Host "PHASE_1_2M_VALIDATION_OUTPUT=$ValidationFolder"

    if (-not $NoOpen) {
        Invoke-Item -LiteralPath $ValidationFolder
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
