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

function Invoke-Batch($batPath, $arguments) {
    if (-not (Test-Path $batPath)) {
        Write-Host "Batch not found: $batPath" -ForegroundColor Yellow
        return 1
    }
    # Use cmd.exe /c to execute the batch and capture its exit code
    $cmd = "`"$batPath`" $arguments"
    Write-Host "Running: $cmd"
    cmd.exe /c $cmd
    return $LASTEXITCODE
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
        Write-Host “$secondsRunning..” -NoNewLine; $secondsRunning--
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

Write-Host "`nAttempting to start SkyrimNet ChatterBox..." -ForegroundColor Green

# Locate python to run the project. Prefer venv python if present.
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPython = Join-Path $scriptRoot '.venv\Scripts\python.exe'

if (Test-Path $venvPython) {
    $pythonPath = $venvPython
    Write-Host "Using virtualenv python: $pythonPath"
} else {
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $pythonPath = $pyCmd.Source
        Write-Host "Using system python: $pythonPath"
    } else {
        Write-Host "No python executable found. Please create/activate a virtualenv or install Python and ensure it's in PATH." -ForegroundColor Red
        Read-Host -Prompt "Press Enter to exit"
        exit 1
    }
}

# Script to run (relative to repo root)
$scriptToRun = '-m skyrimnet_chatterbox'

# Build Python script arguments
$pythonArgs = "--server $server --port $port"

if ($Multilingual -and $Turbo) {
    Write-Host "Error: Cannot use both --multilingual and --turbo flags together. Turbo only supports English." -ForegroundColor Red
    Read-Host -Prompt "Press Enter to exit"
    exit 1
}
if ($Multilingual) {
    $pythonArgs = "$pythonArgs--multilingual"
    Write-Host "Multilingual mode enabled" -ForegroundColor Cyan
} elseif ($Turbo) {
    $pythonArgs = "$pythonArgs --turbo"
    Write-Host "Turbo mode enabled (faster, English only)" -ForegroundColor Cyan
}


# Start a new PowerShell window, set the console title, and run the python script inside it.
if ($pythonArgs) {
    Write-Host "Starting new PowerShell window to run: $pythonPath $scriptToRun $pythonArgs"
} else {
    Write-Host "Starting new PowerShell window to run: $pythonPath $scriptToRun"
}

# Build the command to run inside the new PowerShell instance. Escape $Host so it's evaluated by the child PowerShell.
# Set environment variables in the child process to fix Triton compilation on Windows
$psCommand = "`$Host.UI.RawUI.WindowTitle = 'SkyrimNet ChatterBox'; $vsInitCommand & '$pythonPath' $scriptToRun $pythonArgs"

# Launch PowerShell in a new window and keep it open (-NoExit) so errors remain visible.
$proc = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoExit','-Command',$psCommand) -WorkingDirectory $scriptRoot -PassThru
try {
    # Set the PowerShell window process priority to High.
    $proc.PriorityClass = 'High'
    Write-Host "Set PowerShell window process priority to High (Id=$($proc.Id))."
} catch {
    Write-Host "Warning: failed to set process priority: $_" -ForegroundColor Yellow
}

Write-Host "`nSkyrimNet ChatterBox should start in another window." -ForegroundColor Green
# Write-Host "If that window closes immediately, run $scriptToRun to capture errors." -ForegroundColor Yellow
Any_Key_Wait -msg "Otherwise, you may close this window if it does not close itself.`n" -wait_sec 20