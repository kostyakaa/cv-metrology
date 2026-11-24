[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('Check','Trial','Capture','Analyze')][string]$Stage,
    [Parameter(Mandatory=$true)][string]$RunDir,
    [string]$Python = 'python',
    [string]$CameraName = 'UC60 Video',
    [ValidatePattern('^COM[0-9]+$')][string]$Port = 'COM4',
    [string]$Profile = 'calib/calib_cam0_1920x1080_pinhole.json',
    [string]$FrameSource = 'calib/table_frame.json',
    [string]$PlaneSource = 'calib/table_plane.json',
    [ValidateRange(1,8192)][int]$Width = 1920,
    [ValidateRange(1,8192)][int]$Height = 1080,
    [Nullable[double]]$HeadZmm,
    [ValidateRange(1,400)][int]$AreaX = 68,
    [ValidateRange(1,400)][int]$AreaY = 47,
    [ValidateRange(0,50)][int]$Margin = 7,
    [ValidateRange(0,100)][int]$Below = 18,
    [string]$Scan
)

$ErrorActionPreference = 'Stop'
function Invoke-Step {
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python exit code $LASTEXITCODE in $($args[0])."
    }
}
function Decimal-Argument([double]$Value) {
    $Value.ToString([Globalization.CultureInfo]::InvariantCulture)
}

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if ($Stage -eq 'Analyze') {
        if (!$Scan) { throw 'Pass the exact folder printed after b with -Scan.' }
        $scanPath = (Resolve-Path -LiteralPath $Scan).Path
        if (!(Test-Path -LiteralPath (Join-Path $scanPath 'scan.json') -PathType Leaf)) {
            throw "No scan.json in $scanPath"
        }
        if (Test-Path -LiteralPath $RunDir) { throw "Choose a new analysis output: $RunDir" }
        $dir = (New-Item -ItemType Directory -Path $RunDir).FullName
        $base = Join-Path $dir 'baseline'
        $datum = Join-Path $base 'area.json'
        $before = Join-Path $dir 'stereo-original'
        $after = Join-Path $dir 'stereo-corrected'
        $axes = Join-Path $dir 'axes.json'
        $areaArgs = @('--area-mm',$AreaX,$AreaY,'--area-margin',$Margin,'--below',$Below,'--cell-mm','0.5')
        Invoke-Step z/area_height.py $scanPath @areaArgs --references 5 --fusion range --no-pose-refinement --no-fill --out $base
        Invoke-Step z/network_height.py $scanPath @areaArgs --datum-from $datum --out $before
        Invoke-Step z/render_metric_height.py $before
        Invoke-Step z/fit_stereo_axes.py $scanPath --datum-from $datum --out $axes
        if ((Get-Content -Raw -LiteralPath $axes | ConvertFrom-Json).accepted) {
            Invoke-Step z/network_height.py $scanPath @areaArgs --datum-from $datum --axes $axes --out $after
            Invoke-Step z/render_metric_height.py $after
            Write-Output "Stereo with validated axes: $after/area.npz"
        } else {
            Write-Warning "Held-out pairs did not validate the axis correction. See $axes."
        }
        Write-Output "Stereo with original axes: $before/area.npz"
        return
    }

    $profilePath = (Resolve-Path -LiteralPath $Profile).Path
    if (!(Test-Path -LiteralPath $RunDir)) {
        if ($Stage -ne 'Check') { throw 'Run -Stage Check first to prepare the session.' }
        $sourceFrame = (Resolve-Path -LiteralPath $FrameSource).Path
        $sourcePlane = (Resolve-Path -LiteralPath $PlaneSource).Path
        $dir = (New-Item -ItemType Directory -Path $RunDir).FullName
        Copy-Item -LiteralPath $sourceFrame -Destination (Join-Path $dir 'frame.json')
        Copy-Item -LiteralPath $sourcePlane -Destination (Join-Path $dir 'plane.json')
    } else {
        $dir = (Resolve-Path -LiteralPath $RunDir).Path
    }
    $frame = Join-Path $dir 'frame.json'
    $plane = Join-Path $dir 'plane.json'
    $planeMeta = Get-Content -Raw -LiteralPath $plane | ConvertFrom-Json
    if ($planeMeta.head_z_mm -ne $HeadZmm) {
        throw 'Saved plane is for a different head Z. Check -HeadZmm or measure the plane again.'
    }

    $cameraArgs = @('--camera-name',$CameraName,'--port',$Port,'--width',$Width,'--height',$Height,
                    '--calib',$profilePath,'--frame-file',$frame,'--plane-file',$plane)
    if ($null -ne $HeadZmm) { $cameraArgs += @('--head-z-mm',(Decimal-Argument $HeadZmm)) }
    Invoke-Step scripts/scan_plane.py @cameraArgs --preflight

    if ($Stage -eq 'Trial') {
        Write-Output 'At one visible ID0-3 marker, use c, k, p, then q.'
        Invoke-Step scripts/scan_plane.py @cameraArgs --trial
    }
    if ($Stage -eq 'Capture') {
        Write-Output 'At a visible ID0-3 marker use c, k. Jog to the part center, then b. Record the scan path, then q.'
        Invoke-Step scripts/scan_plane.py @cameraArgs --capture-only --grid 5 --baseline 16 --area-mm $AreaX $AreaY --area-margin $Margin --below $Below --feed 120 --slow-feed 120
    }
} finally {
    Pop-Location
}
