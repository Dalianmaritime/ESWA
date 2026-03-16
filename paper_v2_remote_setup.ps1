param(
    [string]$PythonExe = "python",
    [string]$VenvName = ".venv-paper-v2",
    [ValidateSet("cpu", "default", "cu121", "cu124")]
    [string]$TorchChannel = "cpu",
    [switch]$RunSmokeTests,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$WorkspaceRoot = Split-Path -Parent $PSCommandPath
$MainRepo = Join-Path $WorkspaceRoot "flipit-simulation-master"
$EnvRepo = Join-Path $WorkspaceRoot "gym-flipit-master"
$RequirementsFile = Join-Path $MainRepo "requirements_paper_v2_remote.txt"
$VenvPath = Join-Path $WorkspaceRoot $VenvName

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host ("==> {0}" -f $Message) -ForegroundColor Cyan
}

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Label
    )
    if (-not (Test-Path $Path)) {
        throw ("{0} not found: {1}" -f $Label, $Path)
    }
}

function Invoke-External {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    Write-Host ("[{0}] {1} {2}" -f (Split-Path $WorkingDirectory -Leaf), $Executable, ($Arguments -join " "))
    Push-Location $WorkingDirectory
    try {
        & $Executable @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw ("Command failed with exit code {0}" -f $LASTEXITCODE)
        }
    }
    finally {
        Pop-Location
    }
}

Assert-PathExists -Path $MainRepo -Label "Main repository"
Assert-PathExists -Path $EnvRepo -Label "gym_flipit repository"
Assert-PathExists -Path $RequirementsFile -Label "Remote requirements file"

Write-Step "Checking Python version"
$VersionText = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
if ($LASTEXITCODE -ne 0) {
    throw ("Unable to run Python executable: {0}" -f $PythonExe)
}
$PythonVersion = [Version]$VersionText.Trim()
if ($PythonVersion.Major -lt 3 -or ($PythonVersion.Major -eq 3 -and $PythonVersion.Minor -lt 10)) {
    throw ("Python 3.10+ is required. Detected: {0}" -f $PythonVersion.ToString())
}

Write-Host ("Workspace root : {0}" -f $WorkspaceRoot)
Write-Host ("Main repo      : {0}" -f $MainRepo)
Write-Host ("Env repo       : {0}" -f $EnvRepo)
Write-Host ("Python version : {0}" -f $PythonVersion.ToString())
Write-Host ("Torch channel  : {0}" -f $TorchChannel)

if ($CheckOnly) {
    Write-Step "Check-only mode"
    Write-Host "Directory structure and Python version look good."
    Write-Host "Run the full setup with:"
    Write-Host ("  powershell -ExecutionPolicy Bypass -File `"{0}`" -RunSmokeTests" -f (Join-Path $WorkspaceRoot "paper_v2_remote_setup.ps1"))
    exit 0
}

Write-Step "Creating virtual environment"
if (-not (Test-Path $VenvPath)) {
    Invoke-External -Executable $PythonExe -Arguments @("-m", "venv", $VenvPath) -WorkingDirectory $WorkspaceRoot
}
else {
    Write-Host ("Virtual environment already exists: {0}" -f $VenvPath)
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"
Assert-PathExists -Path $VenvPython -Label "Virtual environment Python"
Assert-PathExists -Path $VenvPip -Label "Virtual environment pip"

Write-Step "Upgrading pip tooling"
Invoke-External -Executable $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") -WorkingDirectory $WorkspaceRoot

Write-Step "Installing PyTorch"
switch ($TorchChannel) {
    "cpu" {
        Invoke-External -Executable $VenvPip -Arguments @("install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu") -WorkingDirectory $WorkspaceRoot
    }
    "cu121" {
        Invoke-External -Executable $VenvPip -Arguments @("install", "torch", "--index-url", "https://download.pytorch.org/whl/cu121") -WorkingDirectory $WorkspaceRoot
    }
    "cu124" {
        Invoke-External -Executable $VenvPip -Arguments @("install", "torch", "--index-url", "https://download.pytorch.org/whl/cu124") -WorkingDirectory $WorkspaceRoot
    }
    "default" {
        Invoke-External -Executable $VenvPip -Arguments @("install", "torch") -WorkingDirectory $WorkspaceRoot
    }
}

Write-Step "Installing paper V2 dependencies"
Invoke-External -Executable $VenvPip -Arguments @("install", "-r", $RequirementsFile) -WorkingDirectory $WorkspaceRoot

Write-Step "Installing local gym_flipit package"
Invoke-External -Executable $VenvPip -Arguments @("install", "-e", $EnvRepo) -WorkingDirectory $WorkspaceRoot

Write-Step "Running import smoke check"
Invoke-External -Executable $VenvPython -Arguments @(
    "-c",
    "import gym, gymnasium, matplotlib, numpy, torch, yaml; import gym_flipit; print('imports_ok'); print(torch.__version__)"
) -WorkingDirectory $WorkspaceRoot

if ($RunSmokeTests) {
    Write-Step "Running V2 smoke tests"
    Invoke-External -Executable $VenvPython -Arguments @("-m", "pytest", "tests/test_signal_v2_utils.py", "-q") -WorkingDirectory $MainRepo
    Invoke-External -Executable $VenvPython -Arguments @("-m", "pytest", "tests/test_maritime_cheat_attention_env.py", "-q") -WorkingDirectory $MainRepo
    Invoke-External -Executable $VenvPython -Arguments @("-m", "pytest", "tests/test_paper_v2_batch_and_analysis.py", "-q") -WorkingDirectory $MainRepo
}

Write-Step "Setup completed"
Write-Host "Activate the environment with:"
Write-Host ("  {0}" -f (Join-Path $VenvPath "Scripts\Activate.ps1"))
Write-Host ""
Write-Host "Or use the environment Python directly:"
Write-Host ("  {0}" -f $VenvPython)
Write-Host ""
Write-Host "Recommended next steps:"
Write-Host ("  Set-Location `"{0}`"" -f $MainRepo)
Write-Host ("  {0} run_paper_main_v2.py --seeds 42 123 2027 3407 8848 --results-subdir paper_v1_main_5seed --resume-missing" -f $VenvPython)
