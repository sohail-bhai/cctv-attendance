param(
    [string]$RepoRoot = "F:\sohail\Class_Attendance_YuNet_SFace",
    [string]$FamilyDir = "F:\sohail\Class_Attendance_YuNet_SFace\models\versions\embfam-189aede520a3b581a17b",
    [string]$DiagnosticRunDir = "F:\sohail\Class_Attendance_YuNet_SFace\attendance_output\diagnostics\2026-06-30__B51__P2__CVO_20260712_221105_286649",
    [switch]$NoOpen
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
$RecoveryScript = Join-Path $RepoRoot "scripts\recover_tue_p2_same_track_evidence.py"
$ProductionEmbeddings = Join-Path $RepoRoot "models\student_embeddings.pkl"
$ProductionSummary = Join-Path $RepoRoot "models\embedding_summary.csv"
$ExpectedEmbeddingsHash = "c32ed31df10b7b9b43b8f19977a2adf0a82fb8e7a71f3e8bceb42c0565fedd49"
$ExpectedSummaryHash = "0df35c3bb9207e191aa49dad5536b8378e812d56463491bd7203885dca5ae9ee"

foreach ($RequiredPath in @($Python, $RecoveryScript, $FamilyDir, $DiagnosticRunDir, $ProductionEmbeddings, $ProductionSummary)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required Phase 1.2I-C input not found: $RequiredPath"
    }
}

function Assert-ProductionHashes {
    $EmbeddingHash = (Get-FileHash -LiteralPath $ProductionEmbeddings -Algorithm SHA256).Hash.ToLowerInvariant()
    $SummaryHash = (Get-FileHash -LiteralPath $ProductionSummary -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($EmbeddingHash -ne $ExpectedEmbeddingsHash) {
        throw "Production embedding hash mismatch: $EmbeddingHash"
    }
    if ($SummaryHash -ne $ExpectedSummaryHash) {
        throw "Production summary hash mismatch: $SummaryHash"
    }
}

function Invoke-PythonStage {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host "[$Name]"
    $Started = Get-Date
    $PreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Python @Arguments
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }
    if ($ExitCode -ne 0) {
        throw "$Name failed with exit code $ExitCode. No later stage was started."
    }
    $Elapsed = (Get-Date) - $Started
    Write-Host "$Name passed in $([Math]::Round($Elapsed.TotalSeconds, 2)) seconds."
}

Write-Host "Phase 1.2I-C - automated same-track CCTV evidence recovery"
Write-Host "Repository: $RepoRoot"
Write-Host "Source family: $FamilyDir"
Write-Host "Diagnostic run: $DiagnosticRunDir"
Write-Host "Manual review required: no"
Write-Host "MON_P3 processed: no"

Assert-ProductionHashes

Invoke-PythonStage -Name "Python compilation" -Arguments @(
    "-m", "py_compile",
    (Join-Path $RepoRoot "src\face_attendance\cctv_same_track_recovery.py"),
    $RecoveryScript,
    (Join-Path $RepoRoot "tests\test_cctv_same_track_recovery.py")
)

Invoke-PythonStage -Name "Focused same-track recovery tests" -Arguments @(
    "-m", "unittest", "-v", "tests.test_cctv_same_track_recovery"
)

Invoke-PythonStage -Name "Real TUE_P2 same-track recovery" -Arguments @(
    "-B", $RecoveryScript,
    "--repo-root", $RepoRoot,
    "--family-dir", $FamilyDir,
    "--diagnostic-run-dir", $DiagnosticRunDir
)

Assert-ProductionHashes

$RecoveryRoot = Join-Path $RepoRoot "models\versions\recovery_runs"
$Newest = Get-ChildItem -LiteralPath $RecoveryRoot -Directory -Filter "cctv-recovery-*" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -eq $Newest) {
    throw "Recovery command completed but no recovery output was found under $RecoveryRoot"
}

Write-Host ""
Write-Host "Phase 1.2I-C completed safely."
Write-Host "Exact recovery output: $($Newest.FullName)"
Write-Host "Production embeddings changed: no"
Write-Host "Production summary changed: no"
Write-Host "Datasets changed: no"
Write-Host "Official attendance changed: no"
Write-Host "Candidate promoted: no"
Write-Host "MON_P3 processed: no"
Write-Host "Next: upload recovery_manifest.json and same_track_recovery_audit.csv before building another family."

if (-not $NoOpen) {
    Start-Process explorer.exe $Newest.FullName
}
