[CmdletBinding()]
param(
  [ValidateSet('start', 'status', 'stop', 'doctor', 'init')]
  [string]$Command = 'start'
)
$ErrorActionPreference = 'Stop'
$quantProject = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $quantProject
try {
  if (-not (Test-Path -LiteralPath 'node_modules/tsx')) {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw 'Node dependency installation failed' }
  }
  & node --import tsx src/cli.ts quant $Command
  if ($LASTEXITCODE -ne 0) { throw "quant $Command failed; see the message above" }
} finally { Pop-Location }
