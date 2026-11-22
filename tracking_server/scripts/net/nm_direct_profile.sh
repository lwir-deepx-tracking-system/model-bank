#!/bin/bash
# NetworkManager profile for the second Ethernet port (the direct cable port). No root needed (polkit lets the netdev group edit
# system connections). It only ever adds / changes / removes the profile 'lwir-direct-link' bound to that one interface: the first
# port, its profile and Tailscale are never touched.
#
#   nm_direct_profile.sh apply [--rollback-after SECONDS]   create/refresh the profile; with the option, a timer deletes it again
#                                                           unless 'confirm' is run before it fires
#   nm_direct_profile.sh confirm                            keep the change (stops the rollback timer)
#   nm_direct_profile.sh restore                            delete the profile (the older profiles for this port take over again)
#   nm_direct_profile.sh status
#
# Addresses on the port (a manual profile, not NetworkManager's "shared" mode; DHCP + DNS come from lwir-direct-link.service):
#   10.42.0.1/24      the direct-cable address (PCs get 10.42.0.x by DHCP)
#   169.254.42.1/16   link-local fallback: a PC that gave up on DHCP and uses 169.254.x.x can still reach the board
#   192.168.50.2/24   the team PC GUI link (PC 192.168.50.1, --gui-host) exactly as in profile 'lwir-pc-link'
#   192.168.77.2/24   the older direct LAN address (profile 'LWIR direct LAN')
set -u
NAME="${LWIR_DIRECT_PROFILE:-lwir-direct-link}"
IFACE="${LWIR_DIRECT_IFACE:-enP4p65s0}"
ADDRS="${LWIR_DIRECT_ADDRESSES:-10.42.0.1/24,169.254.42.1/16,192.168.50.2/24,192.168.77.2/24}"
PRIORITY="${LWIR_DIRECT_PRIORITY:-200}"   # above lwir-pc-link (100) and 'LWIR direct LAN' (0): ours wins for this port
TIMER=lwir-net-rollback
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

exists() { nmcli -t -f NAME con show | grep -Fxq "$NAME"; }

case "${1:-status}" in
  apply)
    shift
    after=""
    [ "${1:-}" = "--rollback-after" ] && after="${2:-180}"
    # never act on the first port
    [ "$IFACE" = "enP3p49s0" ] && { echo "refusing: $IFACE is the remote-access port" >&2; exit 2; }
    ip link show "$IFACE" > /dev/null 2>&1 || { echo "no interface $IFACE" >&2; exit 2; }
    if [ -n "$after" ]; then
      export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
      systemctl --user stop "$TIMER.timer" "$TIMER.service" 2> /dev/null
      systemd-run --user --quiet --on-active="${after}s" --timer-property=AccuracySec=1s --unit="$TIMER" bash "$SELF" restore \
        || { echo "could not arm the rollback timer: nothing changed" >&2; exit 3; }
      echo "rollback timer armed: profile $NAME is deleted in ${after}s unless '$SELF confirm' runs first"
    fi
    if exists; then nmcli con delete "$NAME" > /dev/null || exit 4; fi
    nmcli con add type ethernet con-name "$NAME" ifname "$IFACE" \
      connection.autoconnect yes connection.autoconnect-priority "$PRIORITY" \
      ipv4.method manual ipv4.addresses "$ADDRS" ipv4.never-default yes ipv4.may-fail no \
      ipv6.method ignore connection.mdns no connection.llmnr no > /dev/null || exit 4
    echo "profile $NAME created for $IFACE: $ADDRS (priority $PRIORITY)"
    nmcli -f connection.id,connection.interface-name,connection.autoconnect,connection.autoconnect-priority,ipv4.method,ipv4.addresses,ipv4.never-default,ipv6.method con show "$NAME" ;;
  confirm)
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    systemctl --user stop "$TIMER.timer" "$TIMER.service" 2> /dev/null
    echo "rollback timer stopped: profile $NAME stays" ;;
  restore)
    if exists; then nmcli con delete "$NAME" && echo "profile $NAME deleted"; else echo "profile $NAME not present"; fi ;;
  status)
    nmcli -t -f NAME,DEVICE,TYPE,AUTOCONNECT-PRIORITY con show | grep -E "^($NAME|static-link|lwir-pc-link|LWIR direct LAN):" || true
    nmcli -t -f DEVICE,STATE,CONNECTION dev | grep -E "^($IFACE|enP3p49s0):" || true
    ip -4 -o addr show dev "$IFACE" ;;
  *) echo "usage: $0 apply [--rollback-after SECONDS] | confirm | restore | status" >&2; exit 2 ;;
esac
