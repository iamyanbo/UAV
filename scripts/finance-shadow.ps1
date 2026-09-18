$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot ".curi\data\finance-realdata"
$lockPath = Join-Path $runtimeRoot "daily-shadow.lock"
$logPath = Join-Path $runtimeRoot "daily-shadow.log"
New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

try {
  New-Item -ItemType Directory -Path $lockPath -ErrorAction Stop | Out-Null
} catch {
  Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) skipped: another shadow collection is running"
  exit 0
}

try {
  Set-Location -LiteralPath $projectRoot
  Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) starting daily point-in-time collection"

  & node --import tsx src/cli.ts research data sync --direction finance-realdata-v1 --source all *>> $logPath
  if ($LASTEXITCODE -ne 0) { throw "data sync failed with exit code $LASTEXITCODE" }

  & node --import tsx src/cli.ts research data validate --direction finance-realdata-v1 *>> $logPath
  if ($LASTEXITCODE -ne 0) { throw "data validation failed with exit code $LASTEXITCODE" }

  & node --import tsx src/cli.ts research data shadow --direction finance-realdata-v1 *>> $logPath
  if ($LASTEXITCODE -ne 0) { throw "shadow recording failed with exit code $LASTEXITCODE" }

  Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) completed daily point-in-time collection"
} catch {
  Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) failed: $($_.Exception.Message)"
  throw
} finally {
  Remove-Item -LiteralPath $lockPath -Recurse -Force -ErrorAction SilentlyContinue
}
