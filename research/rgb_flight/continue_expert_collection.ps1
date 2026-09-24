param(
    [Parameter(Mandatory=$true)][string]$FirstRunId,
    [Parameter(Mandatory=$true)][int]$FirstProcessId,
    [string]$Root = 'D:/uav-research/idea1',
    [int]$ValidTarget = 250,
    [int]$EpisodeLimit = 300
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$runsRoot = Join-Path $Root 'runs'
$python = 'C:/WINDOWS/py.exe'
$labels = '/home/iamyanbo/uav-rgb-flight/launches/20260922T002848Z/manifests/evaluator_labels/train.json'
$field = '/home/iamyanbo/uav-rgb-flight/launches/20260922T002848Z/obstacle-field.npz'
$currentRunId = $FirstRunId
$currentProcessId = $FirstProcessId

while ($true) {
    $statePath = Join-Path (Join-Path $runsRoot $currentRunId) 'state.json'
    while ($true) {
        $state = $null
        if (Test-Path -LiteralPath $statePath) {
            try { $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json } catch { $state = $null }
        }
        if ($state -and $state.status -ne 'running') { break }
        if (-not (Get-Process -Id $currentProcessId -ErrorAction SilentlyContinue)) {
            throw "Collection process $currentProcessId exited without a terminal receipt: $statePath"
        }
        Start-Sleep -Seconds 30
    }

    if ($state.status -notin @('stage_finished', 'checkpointed') -or
        $state.collection.status -notin @('collection_complete', 'checkpointed_between_episodes')) {
        throw "Collection stopped at $currentRunId with status $($state.status) / $($state.collection.status)"
    }
    $valid = [int]$state.collection.valid_expert_episodes
    $processed = [int]$state.collection.processed_episodes
    Write-Output "Collection ${currentRunId}: $valid valid from $processed attempted"
    if ($valid -ge $ValidTarget) { break }
    if ($processed -ge 10000) { throw 'Training manifest exhausted before valid expert target' }
    if ($processed -ge $EpisodeLimit) { $EpisodeLimit = [Math]::Min(10000, $EpisodeLimit + 100) }

    $previousRemoteRun = [string]$state.jobs[-1].remote_run
    if ($previousRemoteRun -notmatch '^/home/iamyanbo/uav-rgb-flight/runs/[A-Za-z0-9_-]+$') {
        throw 'Missing or invalid remote campaign run path'
    }
    $currentRunId = 'rgb-visual-expert-continue-' + (Get-Date -Format 'yyyyMMddTHHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6)
    $arguments = @('-3.10', 'research/rgb_flight/run.py', '--stage', 'collect-expert-campaign',
        '--backend', 'spark', '--root', $Root, '--run-id', $currentRunId, '--hours', '8',
        '--remote-evaluator-labels', $labels, '--remote-obstacle-field', $field,
        '--maximum-speed-mps', '3', '--episode-limit', [string]$EpisodeLimit,
        '--remote-campaign', ($previousRemoteRun + '/visual-goal-campaign'))
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Root ($currentRunId + '.stdout.log')) `
        -RedirectStandardError (Join-Path $Root ($currentRunId + '.stderr.log')) -PassThru
    $currentProcessId = $process.Id
    Write-Output "Started $currentRunId with limit $EpisodeLimit and process $currentProcessId"
}

$datasetRunId = 'rgb-visual-dataset-250-' + (Get-Date -Format 'yyyyMMddTHHmmss')
$datasetArguments = @('-3.10', 'research/rgb_flight/run.py', '--stage', 'prepare-visual-dataset',
    '--backend', 'spark', '--root', $Root, '--run-id', $datasetRunId, '--hours', '2',
    '--remote-dataset', '/home/iamyanbo/uav-rgb-flight/launches')
$dataset = Start-Process -FilePath $python -ArgumentList $datasetArguments -WorkingDirectory $repoRoot `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Root ($datasetRunId + '.stdout.log')) `
    -RedirectStandardError (Join-Path $Root ($datasetRunId + '.stderr.log')) -PassThru -Wait
if ($dataset.ExitCode -ne 0) { throw "Visual dataset preparation failed in $datasetRunId" }
Write-Output "Prepared the visual dataset in $datasetRunId after $valid distinct valid experts"
