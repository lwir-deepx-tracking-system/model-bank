#!/bin/bash
# live_bind.sh must never give an empty page address (11-22): LWIR_LIVE_BIND, Tailscale, an assigned LAN address, else 127.0.0.1.
# Uses a fake `ip`. Run: bash scripts/net/test_live_bind.sh
set -u
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/bin"
cat > "$work/bin/ip" <<'IP'
#!/bin/sh
for last; do :; done
if [ "$last" = tailscale0 ]; then printf '%s' "${FAKE_TS:-}"; else printf '%s' "${FAKE_ALL:-}"; fi
IP
chmod +x "$work/bin/ip"
TS='4: tailscale0    inet 100.64.1.2/32 scope global tailscale0'
ALL_LAN='3: enP4p65s0    inet 192.168.77.2/24 brd 192.168.77.255 scope global enP4p65s0'
pass=0
fail=0
check_bind() {   # name, wanted address
  local name="$1" want="$2" got
  got=$(PATH="$work/bin:$PATH" bash -c ". \"$HERE/scripts/live_bind.sh\"; lwir_live_bind" 2>/dev/null)
  if [ "$got" = "$want" ]; then echo "PASS  $name -> '$got'"; pass=$((pass + 1)); else echo "FAIL  $name: wanted '$want' got '$got'"; fail=$((fail + 1)); fi
}
FAKE_TS="$TS" check_bind "Tailscale address present (unchanged behaviour)" 100.64.1.2
FAKE_TS="" FAKE_ALL="$ALL_LAN" LWIR_LIVE_LAN_BIND=192.168.77.2 check_bind "no Tailscale, assigned LAN address" 192.168.77.2
FAKE_TS="" FAKE_ALL="" LWIR_LIVE_LAN_BIND=192.168.77.2 check_bind "no Tailscale, LAN address not assigned -> loopback" 127.0.0.1
FAKE_TS="" FAKE_ALL="" check_bind "nothing at all -> loopback (the server can still start)" 127.0.0.1
FAKE_TS="$TS" LWIR_LIVE_BIND=10.42.0.1 check_bind "explicit LWIR_LIVE_BIND wins" 10.42.0.1
FAKE_TS="" FAKE_ALL="" LWIR_LIVE_NETWORK=lan LWIR_LIVE_LAN_BIND=192.168.77.2 check_bind "explicit LAN-only mode keeps its old rule (empty = do not start)" ""
echo "RESULT pass=$pass fail=$fail"
[ "$fail" = 0 ]
