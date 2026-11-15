[CmdletBinding()]
param(
    [ValidateSet("verify", "plan")]
    [string]$Mode = "verify",
    [string]$Python = "",
    [switch]$IncludeBootstrap
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

if (-not $Python) {
    $candidate = Join-Path $env:USERPROFILE "AppData\Local\Programs\Python\Python314\python.exe"
    if (Test-Path -LiteralPath $candidate) {
        $Python = $candidate
    }
    else {
        $Python = (Get-Command python.exe -ErrorAction Stop).Source
    }
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:MPLBACKEND = "Agg"
$pandoc = Join-Path (Split-Path -Parent $root) ".tools\pandoc-3.12.1\pandoc.exe"
if (Test-Path -LiteralPath $pandoc) {
    $env:PANDOC = $pandoc
}

$arguments = @((Join-Path $root "training\reproduce_project.py"), $Mode)
if ($IncludeBootstrap -and $Mode -eq "verify") {
    $arguments += "--include-bootstrap"
}

& $Python @arguments
exit $LASTEXITCODE
