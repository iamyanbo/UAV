param(
    [string]$RunId = 'rgb-visual-expert-pilot-20260922'
)

$ErrorActionPreference = 'Stop'
$studyRoot = 'D:/uav-research/idea1'
$manifestState = Join-Path $studyRoot 'runs/rgb-visual-manifests-20260922/state.json'
$remoteRoot = '/home/iamyanbo/uav-rgb-flight/launches/20260922T002848Z'
$remoteManifest = "$remoteRoot/manifests/MANIFEST.json"
$remoteLabels = "$remoteRoot/manifests/evaluator_labels/train.json"
$remoteField = "$remoteRoot/obstacle-field.npz"
$expectedFieldHash = 'aa10aee79ede86978904de4638fb924e1205b28f014004c06c6bef34738deda9'
$sshKey = Join-Path $env:USERPROFILE '.ssh/gx10_codex_ed25519'
$deadline = (Get-Date).AddHours(8)

while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $manifestState) {
        $state = Get-Content -LiteralPath $manifestState -Raw | ConvertFrom-Json
        if ($state.status -ne 'running') { break }
    }
    Start-Sleep -Seconds 30
}

if (-not $state -or $state.status -ne 'stage_finished' -or -not $state.deterministic_visual_goal_manifests_complete) {
    throw 'Full manifest job did not complete successfully; no expert flight was launched.'
}

$manifestText = & ssh -o BatchMode=yes -o ConnectTimeout=8 -i $sshKey 'iamyanbo@10.31.12.8' "cat $remoteManifest"
if ($LASTEXITCODE -ne 0) { throw 'Cannot read completed Spark manifest receipt.' }
$manifest = $manifestText | ConvertFrom-Json
if ($manifest.status -ne 'deterministic_manifests_complete' -or $manifest.diagnostic_only -or
    $manifest.counts.train -ne 10000 -or $manifest.counts.validation -ne 150 -or
    $manifest.counts.test -ne 200 -or $manifest.obstacle_field_sha256 -ne $expectedFieldHash -or
    -not $manifest.goal_regions_disjoint -or -not $manifest.complete_pair_disjointness) {
    throw 'Manifest counts, split isolation or field provenance failed; no expert flight was launched.'
}

& py -3.10 research/rgb_flight/run.py --stage collect-expert-campaign --root $studyRoot --hours 8 --run-id $RunId --remote-evaluator-labels $remoteLabels --remote-obstacle-field $remoteField --maximum-speed-mps 3 --episode-limit 10
exit $LASTEXITCODE
