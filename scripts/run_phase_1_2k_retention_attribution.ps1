param(
    [string]$RepoRoot = "F:\sohail\Class_Attendance_YuNet_SFace",
    [switch]$PreflightOnly,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
$Script = Join-Path $RepoRoot "scripts\run_phase_1_2k_retention_attribution.py"

function Invoke-Stage {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )
    Write-Host ""
    Write-Host "[$Name]" -ForegroundColor Cyan
    $Started = Get-Date
    & $Action
    $ExitCode = $LASTEXITCODE
    if ($null -eq $ExitCode) { $ExitCode = 0 }
    if ($ExitCode -ne 0) {
        throw "$Name failed with exit code $ExitCode. No later stage was started."
    }
    $Elapsed = ((Get-Date) - $Started).TotalSeconds
    Write-Host "$Name passed in $([math]::Round($Elapsed, 2)) seconds."
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $Script -PathType Leaf)) {
    throw "Phase 1.2K script not found: $Script"
}

$OriginalLocation = Get-Location
$OriginalPythonPath = $env:PYTHONPATH
try {
    Set-Location -LiteralPath $RepoRoot
    $env:PYTHONPATH = $RepoRoot

    Write-Host "Phase 1.2K - bounded MON_P3 retention attribution" -ForegroundColor Cyan
    Write-Host "Repository: $RepoRoot"
    Write-Host "Full MON_P3 recognition repeated: no"
    Write-Host "Manual review repeated: no"
    Write-Host "Production changes allowed: no"

    Invoke-Stage -Name "Python compilation" -Action {
        & $Python -m py_compile `
            "src\face_attendance\mon_p3_retention_attribution.py" `
            "scripts\run_phase_1_2k_retention_attribution.py"
    }

    Invoke-Stage -Name "Focused Phase 1.2K tests" -Action {
        & $Python -m unittest -v tests.test_mon_p3_retention_attribution
    }

    Invoke-Stage -Name "Read-only Phase 1.2K preflight" -Action {
        & $Python $Script --repo-root $RepoRoot --preflight-only
    }

    if ($PreflightOnly) {
        Write-Host ""
        Write-Host "Phase 1.2K preflight completed safely." -ForegroundColor Green
        Write-Host "Attribution executed: no"
        Write-Host "Full MON_P3 recognition repeated: no"
        Write-Host "Manual review repeated: no"
        Write-Host "Production embeddings changed: no"
        Write-Host "Official attendance changed: no"
        Write-Host "Candidate promoted: no"
        exit 0
    }

    Invoke-Stage -Name "Exact 15-track variant attribution" -Action {
        & $Python $Script --repo-root $RepoRoot
    }

    Write-Host ""
    Write-Host "Phase 1.2K completed safely." -ForegroundColor Green
    Write-Host "Full MON_P3 recognition repeated: no"
    Write-Host "Manual review repeated: no"
    Write-Host "Production embeddings changed: no"
    Write-Host "Official attendance changed: no"
    Write-Host "Candidate promoted: no"
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
