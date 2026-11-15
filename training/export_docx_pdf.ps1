param(
    [Parameter(Mandatory = $true)]
    [string]$DocumentPath,
    [Parameter(Mandatory = $true)]
    [string]$PdfPath
)

$ErrorActionPreference = "Stop"
$document = (Resolve-Path -LiteralPath $DocumentPath).Path
$pdf = [System.IO.Path]::GetFullPath($PdfPath)
$parent = Split-Path -Parent $pdf
New-Item -ItemType Directory -Force -Path $parent | Out-Null
if (Test-Path -LiteralPath $pdf) {
    Remove-Item -LiteralPath $pdf -Force
}

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $opened = $word.Documents.Open($document, $false, $true)
    try {
        $opened.SaveAs2($pdf, 17)
    }
    finally {
        $opened.Close(0)
    }
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

$result = Get-Item -LiteralPath $pdf
Write-Output ("PDF_OK {0} {1}" -f $result.FullName, $result.Length)
