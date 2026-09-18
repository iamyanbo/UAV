[CmdletBinding()]
param([ValidateSet('status', 'maintain', 'check')][string]$Command = 'status')
$ErrorActionPreference = 'Stop'
$storageProject = Split-Path -Parent $PSScriptRoot
& py -3.10 (Join-Path $storageProject 'domains/finance_realdata/storage_budget.py') $Command --project-root $storageProject
if ($LASTEXITCODE -ne 0) { throw 'Storage budget does not admit more work; see status above.' }
