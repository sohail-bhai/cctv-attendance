# Phase 1.2G — Conservative Tracklet Acceptance Calibration

## What this patch does

This patch adds an **offline, disabled-by-default shadow calibration workflow**. It freezes the 37 accepted and 63 corrected unresolved tracklets as reusable ground truth, performs deterministic identity-grouped holdout validation, and searches only conservative multi-frame acceptance rules.

It does **not**:

- rerun recognition;
- alter `student_embeddings.pkl`;
- change the official `0.48` match threshold;
- change the official `0.08` margin threshold;
- change the three-checkpoint attendance rule;
- alter production attendance;
- enable the candidate configuration.

## Files to replace/add

Copy these three files into the same relative paths under:

`F:\sohail\Class_Attendance_YuNet_SFace`

1. `src\face_attendance\tracklet_calibration.py` — new
2. `scripts\validate_tracklet_ground_truth.py` — replace
3. `tests\test_tracklet_calibration.py` — new

Do not copy the documentation or manifest files into the project unless you want to retain them for reference.

## Prerequisites

The following previous-phase files must already exist:

- `tracklet_labels_TUE_P1.csv` — 37 rows
- `unresolved_labels_TUE_P1_corrected.csv` — 63 rows
- repaired `data\student_faculty_map.json`
- the Phase 1.2F.1 authoritative-evaluation hotfix

## Step 1 — Run focused regression tests

Open PowerShell and paste:

```powershell
Set-Location "F:\sohail\Class_Attendance_YuNet_SFace"
.\venv\Scripts\Activate.ps1

python -m unittest `
  tests.test_tracklet_calibration `
  tests.test_recall_analysis `
  tests.test_tracklet_review `
  tests.test_roster_integrity `
  -v
```

Expected ending:

```text
Ran 19 tests
OK
```

## Step 2 — Run Phase 1.2G

Paste this complete block:

```powershell
& {
    $ErrorActionPreference = "Stop"

    try {
        Set-Location "F:\sohail\Class_Attendance_YuNet_SFace"
        $pythonExe = Join-Path $PWD "venv\Scripts\python.exe"

        $diagnosticRun = Join-Path $PWD `
          "attendance_output\diagnostics\2026-06-30__B51__P1__CVO_20260712_000753_991890"

        $acceptedReview = Join-Path $diagnosticRun `
          "tracklet_review_2026-06-30__B51__P1__CVO_20260712_000753_991890-20260712074431-763c4d"

        $unresolvedReview = Join-Path $diagnosticRun `
          "recall_analysis_20260712_124500_128763\unresolved_tracklet_review_2026-06-30__B51__P1__CVO_20260712_000753_991890-20260712124500-1f3271"

        $outputDir = Join-Path $diagnosticRun (
          "calibration\tracklet_calibration_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        )

        $required = @(
            $pythonExe,
            $diagnosticRun,
            $acceptedReview,
            $unresolvedReview,
            (Join-Path $PWD "tracklet_labels_TUE_P1.csv"),
            (Join-Path $PWD "unresolved_labels_TUE_P1_corrected.csv"),
            (Join-Path $PWD "data\student_faculty_map.json")
        )

        foreach ($path in $required) {
            if (-not (Test-Path $path)) {
                throw "Required path not found: $path"
            }
        }

        $arguments = @(
            "scripts\validate_tracklet_ground_truth.py",
            "calibrate-tracklets",
            "--diagnostic-run", $diagnosticRun,
            "--accepted-review-package", $acceptedReview,
            "--accepted-labels", (Join-Path $PWD "tracklet_labels_TUE_P1.csv"),
            "--unresolved-review-package", $unresolvedReview,
            "--unresolved-labels", (Join-Path $PWD "unresolved_labels_TUE_P1_corrected.csv"),
            "--student-map", (Join-Path $PWD "data\student_faculty_map.json"),
            "--subject-abbr", "CVO",
            "--output-dir", $outputDir,
            "--folds", "5",
            "--min-oof-recoveries", "3",
            "--min-recovery-folds", "3"
        )

        & $pythonExe @arguments

        if ($LASTEXITCODE -ne 0) {
            throw "Phase 1.2G failed with exit code $LASTEXITCODE."
        }

        $summaryPath = Join-Path $outputDir "calibration_summary.json"
        $candidatePath = Join-Path $outputDir "candidate_config.json"
        $manifestPath = Join-Path $outputDir "output_manifest.json"

        foreach ($path in @($summaryPath, $candidatePath, $manifestPath)) {
            if (-not (Test-Path $path)) {
                throw "Expected output was not created: $path"
            }
        }

        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        $candidate = Get-Content $candidatePath -Raw | ConvertFrom-Json

        if ($candidate.enabled -ne $false) {
            throw "Safety check failed: the candidate must remain disabled."
        }
        if ($summary.production_changes -ne $false) {
            throw "Safety check failed: production_changes must remain false."
        }
        if ($summary.thresholds_changed -ne $false) {
            throw "Safety check failed: thresholds_changed must remain false."
        }
        if ($summary.attendance_rule_changed -ne $false) {
            throw "Safety check failed: attendance_rule_changed must remain false."
        }

        Write-Host ""
        Write-Host "PHASE 1.2G COMPLETED SUCCESSFULLY" -ForegroundColor Green
        Write-Host "Output folder: $outputDir"
        Write-Host "OOF recoveries: $($summary.out_of_fold.true_recoveries)"
        Write-Host "OOF false accepts: $($summary.out_of_fold.false_accepts)"
        Write-Host "Final shadow recoveries: $($summary.final_candidate.true_recoveries)"
        Write-Host "Final shadow false accepts: $($summary.final_candidate.false_accepts)"
        Write-Host "Recommendation: $($summary.recommendation.decision)"
        Write-Host "Candidate enabled: $($candidate.enabled)"
        Write-Host ""

        Start-Process explorer.exe $outputDir
    }
    catch {
        Write-Host ""
        Write-Host "PHASE 1.2G STOPPED" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Yellow
    }
}
```

## Expected result for the supplied TUE_P1 benchmark

```text
Frozen reviewed benchmark: 100 tracks
Out-of-fold recovery: 7 correct / 0 false accepts
Final shadow candidate: 8 correct / 0 false accepts
Recommendation: promote_to_multisession_shadow_validation
Candidate enabled: no
Production recognition changed: no
```

The candidate ID should be:

`cal-5cd35b60dd83`

The candidate remains disabled and may only proceed to an untouched-session shadow validation.

## Files to send back after the run

From the newly opened output folder, upload:

1. `calibration_summary.json`
2. `candidate_config.json`
3. `calibration_report.md`
4. `output_manifest.json`

Do not run recognition, rebuild embeddings, or apply the candidate to official attendance yet.
