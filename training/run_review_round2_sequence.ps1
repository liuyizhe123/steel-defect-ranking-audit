$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $env:USERPROFILE "AppData\Local\Programs\Python\Python314\python.exe"
$log = Join-Path $here "review_round2_sequence.log"

function Write-Log([string]$message) {
    $line = "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $message
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

Set-Location $here
try {
    Write-Log "REVIEW_ROUND2_STARTED"
    & $python (Join-Path $here "run_review_round2.py") 2>&1 |
        Tee-Object -FilePath (Join-Path $here "review_round2_console.log") -Append
    if ($LASTEXITCODE -ne 0) {
        throw "run_review_round2 failed with exit code $LASTEXITCODE"
    }
    & $python (Join-Path $here "build_review_round2_report.py") 2>&1 |
        Tee-Object -FilePath (Join-Path $here "review_round2_report_console.log") -Append
    if ($LASTEXITCODE -ne 0) {
        throw "build_review_round2_report failed with exit code $LASTEXITCODE"
    }
    $marker = [ordered]@{
        completed_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        status = "complete"
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText(
        (Join-Path $here "review_round2_complete.json"),
        ($marker | ConvertTo-Json),
        $utf8NoBom
    )
    Write-Log "REVIEW_ROUND2_COMPLETE"
    exit 0
}
catch {
    Write-Log ("REVIEW_ROUND2_FAILED " + $_.Exception.Message)
    exit 1
}
