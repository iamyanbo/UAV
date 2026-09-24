# Operator and continuation handoff

> Historical diagnostic instructions. The current program is [RGB-only continuous flight](../rgb_flight/PROJECT.md). Default launchers have been redirected to its preflight; old direct supervisor commands require `--legacy-diagnostic`. Preserve existing checkpoints and raw evidence.

Read [PROJECT.md](PROJECT.md) for the research question and verified boundaries. This runner executes code; it does not ask an LLM to redesign the experiment.

## Start, inspect, stop

From the repository, `npm run idea1:start` starts a hidden eight-hour supervisor and prints its run/report/stop paths. Do not launch a second GPU run. Read that run's `REPORT.md`, then `summary.json` and the current stage's `stdout.log`. Completed stages have `result.json`; active processes have `process.json` with PID and birth identity.

Create the printed `STOP` file to cancel. The backend terminates only its owned process group. **`npm run uav:stop` belongs to the historical autonomous pipeline, not this runner.** Do not kill unrelated Python, WSL or GPU processes to make room.

## When a gate fails

Read the exact exception and resource receipt first. Do not increase memory ceilings, reduce model widths or replace the city. Fix local adapter code, never pinned upstream files or an already running source snapshot. Run the unit tests and appropriate real-data gate again. No automatic repair loop is enabled.

The package fingerprint names the model directory. Each checkpoint also binds the dataset, processed caches and—for compact arms—the exact reference checkpoint. A changed teacher cannot silently become the parent of an existing student. Checkpoints retain optimizer, mixed-precision scaler and all RNG states; they are saved after the first five updates, every three minutes and at completion. Re-running identical code resumes them; changed code starts a separate namespace. Preserve old evidence.

The committed v1 data writer shortened decimal horizon filenames (`h0.5` became `h0.png`). `observation_artifact` resolves that convention, with regression tests. The saved timestamps and controls retain the correct half-second horizon. Do not rename raw files or invalidate their hashes.

## Read results at their actual scope

Seed 997 and three-update checks are diagnostics, not study results. Study seeds are 0–2. Inspect per-mission regret, feature baselines, raw held-out predictions, mission-switch trajectories and measured latency—not just training loss. A full reference means the complete observed **hierarchy**, not access to the simulator's complete scene.

Before a paper claim, add strong aerial/semantic-retrieval baselines, native reference checks, independent collision geometry, better semantic annotations and genuinely held-out sites. Current visibility measures previously observed target surfaces; it is not complete building inspection. Negative results reject only this tested configuration, not world models, JEPA, VLMs or 3DGS as a field.
