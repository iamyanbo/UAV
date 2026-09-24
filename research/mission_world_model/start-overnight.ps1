param(
    [string]$DataRoot = 'D:\uav-research\idea1',
    [ValidateRange(0.1, 8)][double]$Hours = 8
)
$ErrorActionPreference = 'Stop'
$resolvedDataRoot = [System.IO.Path]::GetFullPath($DataRoot)
if (-not (Test-Path -LiteralPath $resolvedDataRoot -PathType Container)) {
    throw 'Run setup first; the dedicated research data directory does not exist.'
}
$repositoryDirectory = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$launcherDirectory = Join-Path $resolvedDataRoot 'launcher-logs'
New-Item -ItemType Directory -Path $launcherDirectory -Force | Out-Null
$runId = 'overnight-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 6)
$runDirectory = Join-Path (Join-Path $resolvedDataRoot 'runs') $runId
$supervisor = Join-Path (Split-Path $PSScriptRoot -Parent) 'rgb_flight/run.py'
$arguments = @('-3.10', ('"{0}"' -f $supervisor), '--stage', 'all', '--hours', $Hours.ToString([Globalization.CultureInfo]::InvariantCulture), '--root', ('"{0}"' -f $resolvedDataRoot), '--run-id', $runId)
$process = Start-Process -FilePath (Get-Command py.exe).Source -ArgumentList $arguments -WorkingDirectory $repositoryDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $launcherDirectory ($runId + '.stdout.log')) -RedirectStandardError (Join-Path $launcherDirectory ($runId + '.stderr.log'))
[pscustomobject]@{
    supervisor_pid = $process.Id
    started_utc = $process.StartTime.ToUniversalTime().ToString('o')
    run = $runDirectory
    report = Join-Path $runDirectory 'REPORT.md'
    stop_marker = Join-Path $runDirectory 'STOP'
    hours = $Hours
    autonomous_agent = $false
} | ConvertTo-Json
