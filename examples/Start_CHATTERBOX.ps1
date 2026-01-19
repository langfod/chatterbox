<#
.SYNOPSIS
Starts SkyrimNet ChatterBox application with optional multilingual support.

.DESCRIPTION
Designed to run on Windows 10 (PowerShell 5.1).

Behavior:
- Display banner
- Pause so user can inspect messages
- Start the project using the venv python (if present) in a new window with HIGH priority
- Optionally enable multilingual mode when --multilingual flag is provided

.PARAMETER Multilingual
Enable multilingual text-to-speech mode

.PARAMETER Turbo
Enable turbo mode (faster, English only)

.EXAMPLE
.\2_Start_ChatterBox.ps1
Start ChatterBox in standard mode

.EXAMPLE
.\2_Start_ChatterBox.ps1 -Multilingual
Start ChatterBox in multilingual mode

.EXAMPLE
.\2_Start_ChatterBox.ps1 -Turbo
Start ChatterBox in turbo mode

Notes:
- If the venv python isn't found this script will try the system python in PATH.
- This script uses cmd.exe start /high to set process priority (works on Windows 10).
#>

param(
    [switch]$Multilingual,
    [switch]$Turbo,
    [string]$server = "0.0.0.0",
    [int]$port = 7860
)

function Show-Banner {
    $banner = @'
  ad88888ba   88                                 88                      888b      88                       
 d8"     "8b  88                                 ""                      8888b     88                ,d     
 Y8,          88                                                         88 `8b    88                88     
 `Y8aaaaa,    88   ,d8  8b       d8  8b,dPPYba,  88  88,dPYba,,adPYba,   88  `8b   88   ,adPPYba,  MM88MMM  
   `""""""8b,  88 ,a8"   `8b     d8'  88P'   "Y8  88  88P'   "88"    "8a  88   `8b  88  a8P_____88    88     
         `8b  8888[      `8b   d8'   88          88  88      88      88  88    `8b 88  8PP"""""""    88     
 Y8a     a8P  88`"Yba,    `8b,d8'    88          88  88      88      88  88     `8888  "8b,   ,aa    88,    
  "Y88888P"   88   `Y8a     Y88'     88          88  88      88      88  88      `888   `"Ybbd8"'    "Y888  
                            d8'                                               
                           d8'       ChatterBox                                      
 
'@

    Write-Host $banner
}


function Any_Key_Wait {
    param (
        [string]$msg = "Press any key to continue...",
        [int]$wait_sec = 5
    )
    if ([Console]::KeyAvailable) {[Console]::ReadKey($true) }
    $secondsRunning = $wait_sec;
    Write-Host "$msg" -NoNewline
    While ( !([Console]::KeyAvailable) -And ($secondsRunning -gt 0)) {
        Start-Sleep -Seconds 1;
        Write-Host "$secondsRunning.." -NoNewline; $secondsRunning--
}

}
Clear-Host
Show-Banner

function Find-VsDevShell {
    # Method 1: Try vswhere.exe (most reliable - works regardless of install location)
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -property installationPath 2>$null
        if ($vsPath) {
            $script = Join-Path $vsPath "Common7\Tools\Launch-VsDevShell.ps1"
            if (Test-Path $script) { return $script }
        }
    }
    
    # Method 2: Check common install locations as fallback
    $basePaths = @($env:ProgramFiles, ${env:ProgramFiles(x86)})
    $years = @('2026', '2022', '2019')
    $editions = @('Community', 'Professional', 'Enterprise', 'BuildTools')
    
    foreach ($base in $basePaths) {
        foreach ($year in $years) {
            foreach ($edition in $editions) {
                $path = Join-Path $base "Microsoft Visual Studio\$year\$edition\Common7\Tools\Launch-VsDevShell.ps1"
                if (Test-Path $path) { return $path }
            }
        }
    }
    
    return $null
}

# Find and initialize VS Dev Shell for x64 native tools
$vsDevShellPath = Find-VsDevShell
if ($vsDevShellPath) {
    Write-Host "Found VS Dev Shell: $vsDevShellPath" -ForegroundColor Cyan
    # Save current directory, launch VS dev shell, and return to original directory
    $currentDirectory = $PWD.Path
    & $vsDevShellPath -Arch amd64
    Set-Location -Path $currentDirectory
} else {
    Write-Warning "Visual Studio Dev Shell not found. Some features may not work correctly."
    Write-Host "Install Visual Studio or Build Tools from https://visualstudio.microsoft.com/downloads/" -ForegroundColor Yellow
}


$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$exePath = Join-Path $scriptRoot 'skyrimnet_chatterbox.exe'


    if (!$exePath) {
        Write-Host "No executable found" -ForegroundColor Red
        Read-Host -Prompt "Press Enter to exit"
        exit 1
    }


$exeArgs = "--server $server --port $port"

if ($Multilingual -and $Turbo) {
    Write-Host "Error: Cannot use both --multilingual and --turbo flags together. Turbo only supports English." -ForegroundColor Red
    Read-Host -Prompt "Press Enter to exit"
    exit 1
}
if ($Multilingual) {
    $exeArgs += " --multilingual"
    Write-Host "Multilingual mode enabled" -ForegroundColor Cyan
} elseif ($Turbo) {
    $exeArgs += " --turbo"
    Write-Host "Turbo mode enabled (faster, English only)" -ForegroundColor Cyan
}

Write-Host "`nAttempting to start SkyrimNet CHATTERBOX..." -ForegroundColor Green
Write-Host "`nFlags: $exeArgs" -ForegroundColor Green
# Build the command to run inside the new PowerShell instance. Escape $Host so it's evaluated by the child PowerShell.
$psCommand = "`$Host.UI.RawUI.WindowTitle = 'SkyrimNet CHATTERBOX'; & '$exePath' $exeArgs"

# Launch PowerShell in a new window and keep it open (-NoExit) so errors remain visible.
$proc = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoExit','-Command',$psCommand) -WorkingDirectory $scriptRoot -PassThru
try {
    # Set the PowerShell window process priority to High.
    $proc.PriorityClass = 'High'
    Write-Host "Set PowerShell window process priority to High (Id=$($proc.Id))."
} catch {
    Write-Host "Warning: failed to set process priority: $_" -ForegroundColor Yellow
}

Write-Host "`nSkyrimNet CHATTERBOX should start in another window. Default web server is http://localhost:7860" -ForegroundColor Green
Any_Key_Wait -msg "Otherwise, you may close this window if it does not close itself.`n" -wait_sec 20
