[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$OutDir,
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    if (Test-Path -LiteralPath $OutDir) { throw "Output directory already exists: $OutDir" }
    $dest = (New-Item -ItemType Directory -Path $OutDir).FullName
    $scan = "scans/scan_20261123-171208"
    $base = Join-Path $dest "baseline"
    $datum = Join-Path $base "area.json"
    $axisFit = Join-Path $dest "axes.json"
    $final = Join-Path $dest "final"
    $compare = Join-Path $dest "comparison"
    function Invoke-PythonStep {
        & $Python @args
        if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE while running $($args[0])" }
    }
    Invoke-PythonStep z/area_height.py $scan --area-mm 68 47 --area-margin 7 --below 18 --cell-mm 0.5 --references 5 --fusion range --out $base
    Invoke-PythonStep z/fit_stereo_axes.py $scan --datum-from $datum --out $axisFit
    Invoke-PythonStep z/network_height.py $scan --datum-from $datum --axes $axisFit --extended --textured-ray --multiscale --out $final
    Invoke-PythonStep z/render_metric_height.py $final
    Invoke-PythonStep z/compare_height_engines.py $scan --masks artifacts/u_part_surfaces_20261123/final/surfaces.npz --maps $base artifacts/u_part_surfaces_20261123/nearest $final --out $compare
    Write-Output "Map: $final/area.npz"
    Write-Output "Metrics: $compare/metrics.json"
} finally {
    Pop-Location
}
