$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$log = Join-Path $here "review_followup_sequence.log"
$python = Join-Path $env:USERPROFILE "AppData\Local\Programs\Python\Python314\python.exe"

function Write-Log([string]$message) {
    $line = "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $message
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

function Invoke-PythonStep([string]$name, [string]$script) {
    Write-Log ("STEP_START " + $name)
    $consoleLog = Join-Path $here ($name + "_console.log")
    & $python $script 2>&1 | Tee-Object -FilePath $consoleLog -Append
    if ($LASTEXITCODE -ne 0) {
        $tail = Get-Content -LiteralPath $consoleLog -Tail 20 -ErrorAction SilentlyContinue
        if ($tail) {
            Write-Log ("PYTHON_TAIL " + ($tail -join " | "))
        }
        throw "$name failed with exit code $LASTEXITCODE"
    }
    Write-Log ("STEP_DONE " + $name)
}

Set-Location $here
try {
    Write-Log "REVIEW_FOLLOWUP_STARTED"

    while ($true) {
        $seedTask = Get-ScheduledTask -TaskName "SCI_Seed_Replication" -ErrorAction SilentlyContinue
        $postTask = Get-ScheduledTask -TaskName "SCI_Post_Seed_Sequence" -ErrorAction SilentlyContinue
        $rtdetrProcess = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -like "*run_rtdetr_matrix.py*" }
        if (
            ($null -eq $seedTask -or $seedTask.State -ne "Running") -and
            ($null -eq $postTask -or $postTask.State -ne "Running") -and
            $null -eq $rtdetrProcess
        ) {
            break
        }
        Start-Sleep -Seconds 300
    }

    $seedSummary = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $here "seed_replication_summary.json") |
        ConvertFrom-Json
    $requiredSeedKeys = @(
        "escm_ciou_s27", "escm_ciou_s28",
        "lsc_ciou_s27", "lsc_ciou_s28",
        "v8n_fwn_s27", "v8n_fwn_s28",
        "v8n_focaler_s27", "v8n_focaler_s28",
        "v8n_hybrid_s27", "v8n_hybrid_s28"
    )
    $missingSeed = @($requiredSeedKeys | Where-Object { $null -eq $seedSummary.$_ })
    if ($missingSeed.Count -gt 0) {
        throw ("seed replication incomplete: " + ($missingSeed -join ","))
    }

    $rtdetrSummary = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $here "rtdetr_summary.json") |
        ConvertFrom-Json
    foreach ($key in @("rtdetr_clean", "rtdetr_jit10", "rtdetr_drop40")) {
        if ($null -eq $rtdetrSummary.$key) {
            throw "RT-DETR result missing: $key"
        }
    }
    Write-Log "UPSTREAM_QUEUE_VERIFIED"

    Invoke-PythonStep "review_matrix" (Join-Path $here "run_review_matrix.py")

    $realizationStats = Join-Path $root "ultralytics_data\test_annotation_realizations\stats.json"
    if (-not (Test-Path -LiteralPath $realizationStats)) {
        Invoke-PythonStep "make_test_realizations" (Join-Path $here "make_test_annotation_realizations.py")
    }
    Invoke-PythonStep "evaluate_test_realizations" (Join-Path $here "run_test_annotation_realizations.py")
    Invoke-PythonStep "efficiency" (Join-Path $here "measure_efficiency.py")
    Invoke-PythonStep "qualitative" (Join-Path $here "make_qualitative_comparison.py")
    Invoke-PythonStep "review_report" (Join-Path $here "build_review_report.py")

    $marker = [ordered]@{
        completed_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        status = "complete"
        review_matrix_required = 29
    }
    $markerPath = Join-Path $here "review_followup_complete.json"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText(
        $markerPath,
        ($marker | ConvertTo-Json),
        $utf8NoBom
    )
    Write-Log "REVIEW_FOLLOWUP_COMPLETE"
    exit 0
}
catch {
    Write-Log ("REVIEW_FOLLOWUP_FAILED " + $_.Exception.Message)
    exit 1
}
