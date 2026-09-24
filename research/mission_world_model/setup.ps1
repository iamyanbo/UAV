param([string]$DataRoot = 'D:\uav-research\idea1')
$ErrorActionPreference = 'Stop'
$ProjectDirectory = $PSScriptRoot
$RepositoryDirectory = Split-Path (Split-Path $ProjectDirectory -Parent) -Parent

function Convert-ToWslPath([string]$Path) {
    $resolvedPath = [System.IO.Path]::GetFullPath($Path).Replace('\', '/')
    if ($resolvedPath -notmatch '^[A-Za-z]:/') { throw 'Expected an absolute Windows drive path.' }
    return '/mnt/' + $resolvedPath.Substring(0, 1).ToLowerInvariant() + $resolvedPath.Substring(2)
}

function Invoke-WslChecked([string[]]$Arguments) {
    & wsl.exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "WSL command failed with exit code $LASTEXITCODE" }
}

if ([System.IO.Path]::GetFullPath($DataRoot).TrimEnd('\') -eq [System.IO.Path]::GetPathRoot($DataRoot).TrimEnd('\')) {
    throw 'DataRoot must be a dedicated project directory, not a drive root.'
}
New-Item -ItemType Directory -Path $DataRoot -Force | Out-Null
$linuxData = Convert-ToWslPath $DataRoot
$linuxProject = Convert-ToWslPath $ProjectDirectory

# These are build/runtime dependencies, not a driver or full system upgrade.
Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '-u', 'root', '--exec', 'apt-get', 'update')
Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '-u', 'root', '--exec', 'apt-get', 'install', '-y', 'python3.10-venv', 'ninja-build', 'ffmpeg', 'gcc-10', 'g++-10')
foreach ($environmentName in @('research', 'vlm')) {
    Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '--exec', 'python3', '-m', 'venv', '--system-site-packages', "$linuxData/envs/$environmentName")
    Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '--exec', 'env', "PIP_CACHE_DIR=$linuxData/cache/pip", "$linuxData/envs/$environmentName/bin/python", '-m', 'pip', 'install', 'pip==25.0.1', 'setuptools==75.8.2', 'wheel==0.45.1')
}
# The existing research base is PyTorch2.1/cu118. Verify, never silently migrate it.
Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '--exec', "$linuxData/envs/research/bin/python", "$linuxProject/bootstrap.py", 'environment', '--root', $linuxData)
Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '--exec', 'env', "PIP_CACHE_DIR=$linuxData/cache/pip", "$linuxData/envs/research/bin/python", '-m', 'pip', 'install', '-r', "$linuxProject/requirements-research.txt")
Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '--exec', "$linuxData/envs/research/bin/python", '-m', 'pip', 'install', '--no-build-isolation', '-e', $linuxProject)
Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '--exec', 'env', "PIP_CACHE_DIR=$linuxData/cache/pip", "$linuxData/envs/vlm/bin/python", '-m', 'pip', 'install', 'torch==2.6.0', 'torchvision==0.21.0', '--index-url', 'https://download.pytorch.org/whl/cu118')
Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '--exec', 'env', "PIP_CACHE_DIR=$linuxData/cache/pip", "$linuxData/envs/vlm/bin/python", '-m', 'pip', 'install', '-r', "$linuxProject/requirements-vlm.txt")
Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '--exec', "$linuxData/envs/vlm/bin/python", '-m', 'pip', 'install', '--no-build-isolation', '-e', $linuxProject)
foreach ($command in @('code', 'dino', 'qwen')) {
    Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '--exec', "$linuxData/envs/research/bin/python", "$linuxProject/bootstrap.py", $command, '--root', $linuxData)
}
Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '--exec', 'git', '-C', "$linuxData/upstream/figs", 'submodule', 'update', '--init', 'acados')
foreach ($scene in @('MatrixCityAerial_1.zip', 'MatrixCityAeiral_2.zip')) {
    foreach ($command in @('city', 'extract')) {
        Invoke-WslChecked -Arguments @('-d', 'Ubuntu', '--exec', "$linuxData/envs/research/bin/python", "$linuxProject/bootstrap.py", $command, '--root', $linuxData, '--scene', $scene)
    }
}
Write-Output 'Setup complete. Run integration gates before the overnight comparison.'
