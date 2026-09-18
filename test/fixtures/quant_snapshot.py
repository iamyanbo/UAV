"""Small synthetic snapshot for canonical-evaluator integration tests only."""
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
rows = []
for day in range(16):
    at = pd.Timestamp("2026-01-01", tz="America/New_York").tz_convert("UTC") + pd.Timedelta(days=day)
    for symbol, factor in [("AAA", 1), ("BBB", .5)]:
        rows.append(dict(symbol=symbol, observation_at=at, available_at=at + pd.Timedelta(days=1),
                         close=100 + day * factor, volume=1000000, dividends=0))
path = root / "prices.parquet"
pd.DataFrame(rows).to_parquet(path)
payload = path.read_bytes()
manifest = dict(snapshot_id="TEST-ONLY", as_of="2026-02-01T20:00:00Z", validation_state="valid",
                files=[dict(path=path.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())])
(root / "manifest.json").write_text(json.dumps(manifest))
