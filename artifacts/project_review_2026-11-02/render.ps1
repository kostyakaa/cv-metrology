$ErrorActionPreference = 'Stop'
$reviewFolder = $PSScriptRoot
$renderFolder = Join-Path $reviewFolder 'rendered'
New-Item -ItemType Directory -Force -Path $renderFolder | Out-Null
$reviewApp = New-Object -ComObject PowerPoint.Application
$reviewDeck = $null
try {
    $reviewDeck = $reviewApp.Presentations.Open((Join-Path $reviewFolder 'project_review.pptx'), $true, $false, $false)
    $overflow = @()
    foreach ($slide in $reviewDeck.Slides) {
        $slide.Export((Join-Path $renderFolder ('slide_{0:D2}.png' -f $slide.SlideIndex)), 'PNG', 1600, 900)
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
                $frame = $shape.TextFrame2
                if ($frame.TextRange.BoundHeight -gt ($shape.Height + 3)) {
                    $overflow += "Slide $($slide.SlideIndex): $($shape.Name), bound=$($frame.TextRange.BoundHeight), height=$($shape.Height)"
                }
            }
        }
    }
    $overflow | Set-Content -Encoding utf8 (Join-Path $reviewFolder 'overflow_check.txt')
    Write-Output "Rendered $($reviewDeck.Slides.Count) slides; height overflow candidates: $($overflow.Count)"
} finally {
    if ($null -ne $reviewDeck) { $reviewDeck.Close() }
    $reviewApp.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($reviewApp) | Out-Null
}
