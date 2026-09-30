param(
    [Parameter(Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$UserName = 'yanbocheng',
    [string]$Command
)

$ErrorActionPreference = 'Stop'
$server = '129.97.250.143'
$fingerprint = 'SHA256:Z6PrkFpfq1OhWJ9iFqvsDDNObGzmuzXq22M19P99scQ'

$sshClient = 'C:/Program Files/Git/usr/bin/ssh.exe'
if (-not (Test-Path -LiteralPath $sshClient)) { $sshClient = (Get-Command ssh -ErrorAction Stop).Source }
$keyFile = Join-Path $env:USERPROFILE '.ssh/uav_viplab_ed25519'
$knownFile = Join-Path $env:USERPROFILE '.ssh/uav_viplab_known_hosts'
if (-not (Test-Path -LiteralPath $keyFile)) { throw 'Lab SSH key is missing on this PC.' }
if (-not (Test-Path -LiteralPath $knownFile)) {
    '129.97.250.143 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJhqqkte4MMGMXaJzgTLzEp1jqTgU7s8Qh49jJkQQp+1' | Set-Content -LiteralPath $knownFile -Encoding ascii
}

Write-Host "Connecting to $server as $UserName."
$sshArguments = @('-i', $keyFile, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', "UserKnownHostsFile=$knownFile", '-o', 'HostKeyAlgorithms=ssh-ed25519', '-l', $UserName, $server)
if ($Command) { $sshArguments += $Command }
& $sshClient @sshArguments
exit $LASTEXITCODE
