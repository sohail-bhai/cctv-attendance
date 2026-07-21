param(
    [string]$EnrollmentApprovals = "D:\Downloads\forensic_review_approvals_compact-enrollment-audit-df0b986b716a2cd5.csv",
    [string]$CctvApprovals = "D:\Downloads\forensic_review_approvals_compact-verified-cctv-8785d6b28789ae6f.csv",
    [string]$FamilyId = "",
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo "venv\Scripts\python.exe"
$Cli = Join-Path $Repo "scripts\validate_tracklet_ground_truth.py"
$ForensicRun = Join-Path $Repo "attendance_output\embedding_forensics\embedding_forensics_reviewfix_20260713_140159_455110"
$CompactRun = Join-Path $Repo "attendance_output\embedding_forensics\embedding_forensics_compact_review_20260713_161633_550718"
$EnrollmentPackage = Join-Path $CompactRun "compact_enrollment_review"
$CctvPackage = Join-Path $CompactRun "compact_cctv_review"
$VersionsRoot = Join-Path $Repo "models\versions"

foreach ($RequiredPath in @(
    $Python,
    $Cli,
    $ForensicRun,
    $EnrollmentPackage,
    $CctvPackage,
    $EnrollmentApprovals,
    $CctvApprovals
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required Phase 1.2I-B input not found: $RequiredPath"
    }
}

Write-Host "Phase 1.2I-B - versioned candidate family" -ForegroundColor Cyan
Write-Host "Repository: $Repo"
Write-Host "Python: $Python"
Write-Host "Frozen benchmark: $ForensicRun"
Write-Host "Enrollment approvals: $((Resolve-Path -LiteralPath $EnrollmentApprovals).Path)"
Write-Host "CCTV approvals: $((Resolve-Path -LiteralPath $CctvApprovals).Path)"
Write-Host "Output root: $VersionsRoot"

$Arguments = @(
    $Cli,
    "build-evaluate-embedding-family",
    "--forensic-run", $ForensicRun,
    "--enrollment-review-package", $EnrollmentPackage,
    "--enrollment-approvals", $EnrollmentApprovals,
    "--cctv-review-package", $CctvPackage,
    "--cctv-approvals", $CctvApprovals,
    "--versions-root", $VersionsRoot
)
if ($FamilyId) {
    $Arguments += @("--family-id", $FamilyId)
}

$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Push-Location $Repo
try {
    & $Python @Arguments 2>&1 | Tee-Object -Variable RunOutput
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "Phase 1.2I-B command failed with exit code $ExitCode."
    }
} finally {
    Pop-Location
    $Stopwatch.Stop()
}

$FamilyLine = $RunOutput | Where-Object { "$_" -like "FAMILY_OUTPUT=*" } | Select-Object -Last 1
if (-not $FamilyLine) {
    throw "The successful command did not report FAMILY_OUTPUT."
}
$FamilyFolder = ("$FamilyLine").Substring("FAMILY_OUTPUT=".Length)
if (-not (Test-Path -LiteralPath $FamilyFolder -PathType Container)) {
    throw "The reported family output folder does not exist: $FamilyFolder"
}

Write-Host "Phase 1.2I-B command completed without process errors." -ForegroundColor Green
Write-Host ("Elapsed: {0:N2} seconds" -f $Stopwatch.Elapsed.TotalSeconds)
Write-Host "Exact family output: $FamilyFolder"
Write-Host "Production promotion performed: no"
Write-Host "MON_P3 processed: no"

if (-not $NoOpen) {
    Invoke-Item -LiteralPath $FamilyFolder
}
