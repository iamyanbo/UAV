# 06 — GPU resource handling and crash-report placement

## Qwen restoration after optimization

The September 29 21:17 window completed the first accepted 8,192-row batch:
64 optimizer steps, final KL 0.00553, behavior error 9.54e-7 and 819 independent
world updates. It then failed memory admission while restoring Qwen to the GPU
because inactive optimizer workspaces remained cached. The implementation now
releases unused CUDA cache before Qwen restoration. This was a lifecycle/resource
failure after actual learning work, not proof that the selected models cannot fit.

## Inherited 60% cap

`runs/city-window-20260930T013511Z` started September 29 at 21:35 EDT and stopped
at its inherited 60% memory allowance. Subsequent explicit user authorization
permits the whole of lab GPU 0 using measured headroom. GPU 1 and unrelated jobs
remain outside the study. Changing this allowance did not solve recording,
timing or optimization errors; these are separate failures.

## OS-generated SSD crash report

During an earlier failed native shutdown, Ubuntu's crash handler wrote a
1,180,700,268-byte report to the SSD, contrary to the intended HDD-only placement.
The study's report was copied to the HDD, fsynced and SHA-verified before its SSD
copy was removed. Other users' reports were preserved. Evidence remains at
`runs/native-crash-1790732067681943566` on the lab HDD.

The study startup hook now disables core dumps and Linux dumpability before
imports. It is copied into operator snapshots and pinned in launch receipts.
Native shutdown resumes the clock and drains/cancels SDK tasks before closing
sockets. Evidence and source: `../plans/LAB_OVERNIGHT_20260929.md` and
`scripts/lab_sitecustomize.py`. No new SSD crash report is claimed to have occurred
in the September 30 PPO rejection.
