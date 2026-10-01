# 05 — startup readiness and first-dispatch failures

The first September 29 overnight launcher used an unauthenticated socket probe
against the authenticated Qwen endpoint. The probe interrupted its handshake.
The launcher was corrected. Evidence is recorded in
`../plans/LAB_OVERNIGHT_20260929.md`; do not describe this as model collapse.

The replacement `runs/city-window-20260930T004718Z` started September 29 around
20:47 EDT and hit the freshness watchdog before recording a PPO row. There was
one reserved dispatch and zero confirmed rows. Its checkpoint and physical
budget charge were preserved, rather than pretending the attempt never happened.

Actual budget bookkeeping had been omitted from the preceding flight check.
The repaired qualification includes it. The city ledger uses WAL NORMAL for
per-action updates after a durable full-batch reservation, then FULL at the
boundary. The RGB memory index defers automatic WAL checkpointing during flight.
Crash-ledger reconciliation remains explicit; unsafe rows are not fabricated.

Measured subsequent qualification: 375 learner intervals, maximum source age
145.8 ms and maximum reservation time 1.18 ms, below the unchanged 250 ms bound.
That narrow result did not establish every later cold/cache/resource path was
bounded; see incident 02 for the subsequent integration failures.
