# Pipeline storage budget

The shared maximum is **40,000,000,000 bytes (40 GB) on disk**, with 1 GB
reserved as operating headroom. `storage-policy.json` names the shared stores;
all `.curi*` profiles in this project are automatically included. This counts
market downloads, immutable snapshots, research workspaces, evidence, databases,
sessions, logs and archives. Repository dependencies and unrelated projects
are outside the managed research budget. Additional participating projects can
be listed in `additional_projects`.

```powershell
.\scripts\storage.ps1 status
.\scripts\storage.ps1 maintain
```

The runtime counts hard links once and uses actual compressed file size on
Windows. `maintain` applies standard NTFS compression to the managed directories,
including compression inheritance for new files. Paths, file contents and
scientific hashes remain unchanged; ordinary Python, SQLite and Parquet readers
continue to work. Apparent file sizes can therefore remain much larger than
the disk space they occupy. On other filesystems, allocated blocks are counted;
automatic NTFS compression is unavailable.

Acquisition, snapshot staging and model turns reserve capacity under a shared
lock. Reservations are released on completion or reclaimed after the owning
process exits. Model turns are checked at startup and every minute as well.
Admission may try lossless compression, then refuses new work if the budget still
cannot accommodate it. Active monitoring only measures; it does not run compression.
The orchestrator receives measured usage and can choose
which symbols, history windows and acquisition cadence are worth the space.
The cap cannot be raised above 40 GB through the policy.

Refreshes with unchanged dataset content reuse the existing snapshot; their
request results and last-check time are retained in the current pointer.
Repeated identical raw-source references are deduplicated in new manifests.
Historical failed attempts remain in old manifests instead of being copied
into every future manifest. Existing evidence is not deleted or rewritten.

This is application admission control, **not a Windows filesystem quota**.
Reservations are conservative estimates for downloads and worker output;
an arbitrary shell command or a single unexpectedly large write can outrun
the periodic monitor. A strict write-by-write ceiling for untrusted workers
would additionally require a size-limited volume or OS quota. The runtime does
not claim to enforce that boundary.

The active-turn monitor runs asynchronously so scanning cannot block model RPC
events or cancellation. Automatic compression during admission under pressure has a
30-minute retry interval; explicit `maintain` bypasses it. Admission still checks
actual usage on every attempt and remains closed when capacity is insufficient.

A failed measurement is recorded as unknown, with the original diagnostic in
`.curi-storage/monitor-<pid>.json`, `monitor-events.jsonl` and the worker trace.
The already admitted research session continues and the monitor retries; new
acquisitions and turns still require successful admission. Directory removal
during a scan is tolerated, while inaccessible directories are reported rather
than counted as empty. Windows API bindings are reused throughout each scan.

Reservations and the 1 GB headroom govern admission of new work. They are not
proof that existing files have filled the allowance. Active work is interrupted
only when measured allocated bytes reach the configured limit, with the measured
bytes and limit retained in its failure record. A lease-release error is recorded
without discarding a completed research handoff.
## Content-addressed snapshots and retention

Historical options downloads keep content-addressed provider pages and verified
completed expiry partitions under `cache/acquisitions/`. Each partition is a
publication boundary; the next acquisition resumes the same requested scope.
The configured byte allowance bounds new downloads in a step, not total research
depth. Hitting it preserves completed pages and reports continuing acquisition,
without counting an ordinary progress checkpoint as a failed request. The watcher
owns one acquisition process, and an OS lock prevents concurrent writers from
publishing competing pointers. Cache, raw payloads and published snapshots remain
inside the same storage allowance.

Snapshot directories keep their paths, bytes and hashes, but identical table
files are stored once under `objects/` and hard-linked into each snapshot. Readers
see ordinary files; the budget counts linked bytes once. New snapshots are linked
as they are written, and `snapshot_store.py dedupe` links existing ones after
verifying each file against its manifest hash.

Retention removes only snapshots nothing depends on. A snapshot is kept when any
local profile's task, canonical evaluation, trial, shadow prediction, evidence
bundle, paper plan or current pointer names it; when it is the newest snapshot of
a UTC day within `retention.keep_daily_days` (30); or when it is younger than
`retention.grace_hours` (48). A removed snapshot leaves its gzipped manifest under
`retired/` and a line in `retired/retention-log.jsonl`. Content objects no
snapshot links to are then removed.

```powershell
npm run storage:retention        # dry run: what would be kept and why
npm run storage:retention:apply  # retention, dedupe and orphan collection
```

The supervisor runs the same maintenance once a day and records results in
`.curi*/maintenance/maintenance.log`, alongside hourly removal of finished task
worktrees with sealed evidence and gzip of week-old traces of finished runs.
