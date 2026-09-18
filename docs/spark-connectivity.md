# DGX Spark connectivity

The launcher no longer depends on Wi‑Fi or Windows mDNS. `scripts/spark.ps1`
tries, in order: `SPARK_HOST`/`-SparkHost`, the wired reservation
`10.13.2.8`, the last known Wi‑Fi DHCP address, and the Spark hostname. Every
candidate is verified with the configured SSH key before it is used.

## One-time wired setup

The Spark's wired interface is `enP7s7` (MAC `30:c5:99:3f:6c:a9`) and its
NetworkManager profile is **Wired connection 3**, already configured for
autoconnect + DHCP. Connect that port to the same router/LAN as this PC's
Ethernet port, then create a DHCP reservation for that MAC. The recommended
reservation is `10.13.2.8`; use the address actually assigned by the router if
it chooses another one.

Persist the address on this PC after the reservation is made:

```powershell
[Environment]::SetEnvironmentVariable('SPARK_HOST', '10.13.2.8', 'User')
```

Open a new PowerShell window and verify:

```powershell
.\scripts\spark.ps1 status
```

SSH is socket-activated on the Spark (`ssh.socket` is enabled), so no Wi‑Fi
login or desktop session is required. If the router cannot provide a DHCP
reservation, use a direct Ethernet static pair instead (Spark `192.168.50.2/24`,
PC Ethernet `192.168.50.1/24`) and set `SPARK_HOST=192.168.50.2`; configuring
those addresses requires administrator access on both machines.
