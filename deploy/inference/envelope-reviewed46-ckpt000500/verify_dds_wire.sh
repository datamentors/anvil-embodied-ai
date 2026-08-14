#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${RUNTIME_ENV_FILE:-${DEPLOY_DIR}/runtime.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: runtime environment not found: ${ENV_FILE}" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

duration="${1:-15}"
iface="${DDS_IFACE:?DDS_IFACE is required}"
local_ip="${DDS_LOCAL_IP:?DDS_LOCAL_IP is required}"
peer_ip="${DDS_PEER_IP:?DDS_PEER_IP is required}"

if [[ ! "${duration}" =~ ^[0-9]+$ ]] || (( duration < 5 || duration > 120 )); then
  echo "Usage: $0 [capture-seconds: 5..120]" >&2
  exit 2
fi

command -v tcpdump >/dev/null || {
  echo "ERROR: tcpdump is required" >&2
  exit 1
}
ip -4 -o addr show dev "${iface}" | grep -q "${local_ip}/24" || {
  echo "ERROR: ${iface} does not own ${local_ip}/24" >&2
  exit 1
}

pcap="$(mktemp --suffix=.dds-domain204.pcap)"
trap 'rm -f "${pcap}"' EXIT

if (( EUID == 0 )); then
  capture=(timeout --signal=INT --kill-after=2 "${duration}s" tcpdump)
else
  echo "[wire] tcpdump requires sudo only for this read-only packet capture."
  sudo -v
  # Keep timeout inside sudo so it can deliver SIGINT to the privilege-dropped
  # tcpdump process. An outer timeout only signals sudo and eventually returns
  # 137 even though the capture itself is valid.
  capture=(sudo -n timeout --signal=INT --kill-after=2 "${duration}s" tcpdump)
fi

# DDS/RTPS domain 204 uses the 58400 base-port range. Capture only that range,
# while the real Devbox sensors are actively flowing in echo mode.
set +e
"${capture[@]}" -U -s 96 -nn -i "${iface}" -w - \
  'udp portrange 58400-58473' >"${pcap}" 2>/dev/null
capture_rc=$?
set -e
if [[ "${capture_rc}" != "0" && "${capture_rc}" != "124" ]]; then
  echo "ERROR: packet capture failed (status ${capture_rc})" >&2
  exit 1
fi

count_packets() {
  local filter="$1"
  tcpdump -nn -r "${pcap}" "${filter}" 2>/dev/null | awk 'END {print NR + 0}'
}

total="$(count_packets 'udp portrange 58400-58473')"
multicast="$(count_packets 'udp portrange 58400-58473 and dst net 224.0.0.0/4')"
foreign="$(count_packets "udp portrange 58400-58473 and not (host ${local_ip} and host ${peer_ip})")"

if (( total == 0 )); then
  echo "ERROR: no domain-204 DDS traffic captured; run this while echo mode sees real sensors" >&2
  exit 1
fi
if (( multicast != 0 )); then
  echo "ERROR: captured ${multicast} multicast DDS packet(s)" >&2
  exit 1
fi
if (( foreign != 0 )); then
  echo "ERROR: captured ${foreign} DDS packet(s) involving a host other than ${local_ip}/${peer_ip}" >&2
  exit 1
fi

echo "DDS_WIRE_PASS packets=${total} multicast=0 foreign_peers=0 endpoints=${local_ip}<->${peer_ip}"
