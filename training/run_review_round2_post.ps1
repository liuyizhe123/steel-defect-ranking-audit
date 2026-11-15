$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $env:USERPROFILE "AppData\Local\Programs\Python\Python314\python.exe"
$log = Join-Path $here "review_round2_post.log"

function Write-Log([string]$message) {
    $line = "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $message
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

Set-Location $here
try {
    while (-not (Test-Path -LiteralPath (Join-Path $here "review_round2_complete.json"))) {
        Start-Sleep -Seconds 60
    }
    Write-Log "ROUND2_POST_STARTED"
    & $python (Join-Path $here "measure_memory_repeats.py") 2>&1 |
        Tee-Object -FilePath (Join-Path $here "efficiency_memory_repeats_console.log") -Append
    if ($LASTEXITCODE -ne 0) {
        throw "measure_memory_repeats failed with exit code $LASTEXITCODE"
    }
    & $python (Join-Path $here "build_review_round2_report.py") 2>&1 |
        Tee-Object -FilePath (Join-Path $here "review_round2_report_console.log") -Append
    if ($LASTEXITCODE -ne 0) {
        throw "build_review_round2_report failed with exit code $LASTEXITCODE"
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText(
        (Join-Path $here "review_round2_post_complete.json"),
        '{"status":"complete"}',
        $utf8NoBom
    )
    Write-Log "ROUND2_POST_COMPLETE"
    exit 0
}
catch {
    Write-Log ("ROUND2_POST_FAILED " + $_.Exception.Message)
    exit 1
}
