# 01 — disk accounting interrupted training

- Run: `runs/city-window-20260930T034237Z`.
- End: September 30, 01:11:17 EDT.
- Failure: `du -sb` scanned a derived feature `.pending` file while its writer
  atomically renamed it. A checked subprocess failure stopped the operator,
  which interrupted its trainer. The subsequent KeyboardInterrupt was cleanup.
- Cause: a live filesystem scan was treated as authoritative during concurrent
  writes. This was not evidence of disk exhaustion or a server outage.
- Preserved reference: `city-training/latest.pt`, SHA-256
  `352aa617c1f4e645621a8f98d2949df92b75520f83b84fa3e3c24f8c53a391b7`.
- Learning state: eight accepted batches, 65,536 rows, 6,552 world updates and
  844 pending rows. These counts do not imply navigation competence.
- Repair: conservative write leases, explicit accounting, quiescent
  reconciliation and disk admission outside live dispatch.
- Evidence: implementation progress and environment/data/reward plan in
  `../plans/`. Later real collection passed this failure point; not a guarantee
  against every future storage fault. Mount, permission and actual I/O errors
  remain distinct fatal errors.
