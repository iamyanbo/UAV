# CURI research recovery and cleanup

September 20, 2026, Toronto time; export timestamp September 21 UTC. Requested scope: organize the accumulated pipeline material, recover useful research into reports, consolidate a literature review, and develop structural world-model/JEPA/VLM ideas.

## Completed organization

The pipeline's supervisor, watcher and active executor were stopped through its own stop command. The ledger was backed up before stopping. Continuous mode was then disabled. A subsequent database inspection found **zero active or waiting-external runs**. One queued task, `TASK-d5253311-331`, remains preserved; the direction's historical database status is still `active`. These labels do not indicate a running process. The immediate-stop and watcher-stop markers remain in place. No autonomous campaign was restarted.

The current reading entry point is [README.md](README.md). The new [general review](UAV%20General%20Literature%20Review%20-%202026-09-20.md) separates external literature from internal experiments. The structural directions were later deleted at the user's request; README.md identifies the current research discussion.

| Material | Action | Recovery/provenance |
|---|---|---|
| Source ledger and cached source texts | Exported into reports | [Source catalog](recovered/curi-2026-09-20/SOURCE-CATALOG.md) |
| Investigations, outcomes, notes, syntheses and later corrections | Exported as raw JSON plus readable records | [Recovered research](recovered/curi-2026-09-20/README.md) |
| Research Markdown buried in sealed task artifacts | Copied, deduplicated by SHA-256, linked to task/run IDs | [Document catalog](recovered/curi-2026-09-20/DOCUMENT-CATALOG.md) |
| 35 historical injection files | Moved out of the pipeline into the reports archive | [Historical bundles](archive/pipeline-2026-09-20/injections/) |
| Original injection paths | Preserved through a Windows directory junction | `curi-uav/presearch/injections` points to the archive |
| 30 monitor records whose PIDs were no longer live | Moved to the archive | [Stale monitor records](archive/pipeline-2026-09-20/stale-monitor-records/) |
| Pipeline README, AGENTS and presearch entry point | Updated to identify stopped/manual research mode | [Original versions](archive/pipeline-2026-09-20/original-entrypoints/) |
| Existing source code and user modifications | Preserved | The repository already had substantial uncommitted work |
| Model weights, experiment data, Git worktrees, original evidence and database | Preserved in the pipeline | Their recorded paths remain valid |

No material files were deleted. Disabling continuous mode removed only its runtime flag using the normal CLI. This was a recoverable organization pass, not a purge of generated code or a rewrite of the engine. Finance-origin compatibility code, dependencies and older operator backups were not deleted: their dependency/recovery roles were not established, and removing them would be a separate code-cleanup task.

## What was recovered

The snapshot contains:

- 46 source records: 45 in `retrieved` state and one unreadable metadata request.
- 45 cached normalized source texts. These include abstracts and code/page extracts; they are not 45 complete papers.
- 39 investigations and 23 investigation plans.
- 17 outcomes, including bounded and inconclusive results.
- Two formal syntheses, both with `needs_evidence` reviews.
- 131 notes and 210 later record reviews, including historical-context dispositions.
- 24 task records, one artifact program, one checkpoint and 1,086 artifact metadata records.
- 70 distinct recovered Markdown artifact documents. Byte-identical copies from multiple runs share one exported file, with every associated task/run retained in the manifest.

The [export manifest](recovered/curi-2026-09-20/manifest.json) records original paths, byte lengths and SHA-256 hashes for 115 copied files: 45 source texts plus 70 artifact documents. All copied-file hashes and all 17 exported-table row counts were verified. No requested source/artifact file in this export selection was missing. This is a complete export of the selected live-ledger tables and available stored Markdown artifacts, not a forensic reconstruction of every historical database backup or unsealed workspace.

Authentication stores, `.env`, provider credentials, raw inference sessions and dependency folders were not copied into reports. The original database backup stays inside the pipeline's operator-backup directory rather than becoming a published research artifact.

## Useful findings worth retaining

### A. Architecture comparisons that prevent repeating weak proposals

The pipeline accumulated fast/slow VLA, action-chunk correction, calibration, recurrent memory, JEPA factorization and semantic mapping leads. These help establish what is already a normal architectural ingredient. The recovered source cards mostly have retrieval provenance rather than complete architecture analysis; their high-value role is precise discovery and follow-up. [Source catalog](recovered/curi-2026-09-20/SOURCE-CATALOG.md)

The older investigations also distinguish state-based quadrotor dynamics from visual semantic navigation. This is consequential: calling a small state predictor “JEPA for UAVs” cannot establish that visual prediction or language grounding has been solved. The new literature review checks this distinction against SkyJEPA's primary methods.

### B. Actual code-level mapping and uncertainty knowledge

The timing-map-backup audit located the correct certifiably-correct-mapping repository and its nested nvblox implementation. It separated implemented covariance deflation from which overload a ROS node called, documented map reset behavior and kept missing producer/controller evidence explicit. This is considerably more useful than a generic recommendation to “add map uncertainty.” These are inherited source findings; the new pass recovered and read the audit rather than rerunning all its repository inspections. [Recovered audit](recovered/curi-2026-09-20/documents/TASK-6b134303-92b/fc17d09c80da-Joint%20Timing-Map-Backup%20Source%20Audit%202026-09-20.md), [investigation continuation](recovered/curi-2026-09-20/investigations/INV-f3131fb7-08c.md)

### C. More careful interpretation of semantic-navigation failures

The VLN-on-the-Fly investigation distinguishes incorrect target commitment, stopping short of a correct target, and choosing not to execute. It also notes that repeated correlated semantic answers cannot be treated as independent confirmations. That is useful interpretation of a system's evidence, even though a generic consensus or identity-memory wrapper is not a new model contribution. Numerical claims remain inherited until the source is rechecked for a decisive comparison. [Recovered failure analysis](recovered/curi-2026-09-20/investigations/INV-8b7ec76e-945.md)

### D. Why the early “supported” head result is not a strong contribution

The first synthesis's independent review records nearly deterministic calibration targets, a mismatch between binary and multiclass tasks, a documented AUC bug, absent sealed code in that evidence boundary, and cached-feature rather than complete inference timing. These are useful corrections. Its positive headline should not survive without those qualifications. [Synthesis with review attached](recovered/curi-2026-09-20/component_syntheses/SYN-e90e6e79-37b.md)

### E. Useful implementation scaffolding, limited research evidence

The later program produced a compact visual model stack, trained checkpoints, a small MuJoCo setup and raw metrics. Its pairwise ranking-loss comparison did not establish convincing visual navigation. These assets may reduce future engineering work; their existence does not justify keeping that loss as the research agenda. The later synthesis also lacked adequate outcome linkage in its review. [Later synthesis and review](recovered/curi-2026-09-20/component_syntheses/SYN-aeba3eac-03c.md), [recovered project account](recovered/curi-2026-09-20/documents/TASK-d58661b2-414/0ad1e3586e87-PROJECT.md)

## Corrections to how the material should be read

Original wording is preserved inside the recovered records, including stale claims, overconfident novelty language and finance-template residue. A recovery export is not an endorsement. Later reviews are shown above original investigations/outcomes/notes/syntheses where the ledger associates them. Artifact documents retain their exact bytes; consult their task's outcome and reviews before relying on their claims.

Some imported documents use relative links written for their original worktree. Those links were not rewritten because that would alter the preserved artifact. Their catalog and manifest supply the original paths and identifiers. Links in the newly authored entry point, review, structural proposals and this cleanup account are checked separately.

The new synthesis changes the role of the pipeline: source retrieval, exact comparisons and scoped execution can be useful. Open-ended autonomous novelty generation is stopped. No small or invalid pilot is used to reject world models, JEPA, VLM/VLA or geometry as a family.

## Recovery and maintenance

The pre-stop SQLite backup is `C:/Users/yanbo/downloads/curi-uav/.curi/operator-backups/reports-cleanup-2026-09-20-before.sqlite`. It was created with SQLite's online backup mechanism, not by copying a potentially incomplete WAL-backed database file. Do not overwrite the live database to undo document organization; database restoration is unnecessary for that purpose.

The [cleanup manifest](archive/pipeline-2026-09-20/cleanup-manifest.json) lists every moved injection and monitor record. To undo the injection move, first verify that `curi-uav/presearch/injections` is the compatibility junction, remove only that junction without recursing into its target, then move the archived directory back to its original path. Restore entry-point files individually from `original-entrypoints` if desired. The reports archive must be kept alongside the pipeline while the junction is used.

The [recovery tool](tools/recover-curi-research.mjs) opens the database read-only for inspection/export and verifies exported hashes with `node tools/recover-curi-research.mjs verify`. Export refuses to overwrite an existing snapshot. The [organization script](tools/organize-curi.ps1) previews by default, validates exact source/destination roots, verifies moved injection hashes, and refuses a repeat archive operation.

No experiments or new model training were launched in this pass. The three structural proposals are designs with explicit prior-art limits; they are the next intellectual work product, not a report of scientific success.
