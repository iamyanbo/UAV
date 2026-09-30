#!/usr/bin/env bash
set -euo pipefail

username=${1:-yanbocheng}
key_file=${UAV_LAB_SSH_KEY:-$HOME/.ssh/uav_viplab_ed25519}
known_file=${UAV_LAB_KNOWN_HOSTS:-$HOME/.ssh/uav_viplab_known_hosts}
[[ -f $key_file && -f $known_file ]] || { printf 'Lab key or pinned known-host file is missing.\n' >&2; exit 2; }

server=129.97.250.143
fingerprint=SHA256:Z6PrkFpfq1OhWJ9iFqvsDDNObGzmuzXq22M19P99scQ

printf 'Connecting to %s as %s.\n' "$server" "$username"
exec ssh -i "$key_file" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$known_file" -o HostKeyAlgorithms=ssh-ed25519 -l "$username" "$server"
