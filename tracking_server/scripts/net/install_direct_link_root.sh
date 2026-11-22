#!/bin/bash
# The ONE step that needs root (everything else runs as the normal user): DHCP + captive-portal DNS on the direct cable port
# and the port 80 redirect. Run once on the board:   sudo bash scripts/net/install_direct_link_root.sh
#   --uninstall   remove it again (service stopped and disabled, files and the nft table deleted)
#   --check       only say what is installed
# LWIR_ROOT_ETC / LWIR_ROOT_UNIT_DIR / LWIR_ROOT_DRY=1 are for tests (install into a scratch folder, touch no service).
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
ETC="${LWIR_ROOT_ETC:-/etc}"
UNIT_DIR="${LWIR_ROOT_UNIT_DIR:-/etc/systemd/system}"
DRY="${LWIR_ROOT_DRY:-0}"
UNIT=lwir-direct-link.service

[ "$DRY" = 1 ] || [ "$(id -u)" = 0 ] || { echo "run with sudo: sudo bash $0" >&2; exit 1; }

case "${1:-install}" in
  --check)
    for f in "$ETC/lwir/direct-dnsmasq.conf" "$ETC/lwir/direct-link.nft" "$UNIT_DIR/$UNIT"; do [ -f "$f" ] && echo "present: $f" || echo "missing: $f"; done
    [ "$DRY" = 1 ] || { systemctl is-enabled "$UNIT" 2>&1 | sed 's/^/enabled: /'; systemctl is-active "$UNIT" 2>&1 | sed 's/^/active: /'; }
    exit 0 ;;
  --uninstall)
    if [ "$DRY" != 1 ]; then systemctl disable --now "$UNIT" 2>/dev/null || true; /usr/sbin/nft delete table ip lwir_direct 2>/dev/null || true; fi
    rm -f "$UNIT_DIR/$UNIT" "$ETC/lwir/direct-dnsmasq.conf" "$ETC/lwir/direct-link.nft"
    rmdir "$ETC/lwir" 2>/dev/null || true
    [ "$DRY" = 1 ] || systemctl daemon-reload
    echo "removed"; exit 0 ;;
  install) ;;
  *) echo "usage: $0 [--check|--uninstall]" >&2; exit 2 ;;
esac

if [ "$DRY" != 1 ]; then
  command -v dnsmasq >/dev/null || { echo "dnsmasq is not installed (apt install dnsmasq-base dnsmasq)" >&2; exit 3; }
  command -v nft >/dev/null || { echo "nft is not installed (apt install nftables)" >&2; exit 3; }
  getent passwd dnsmasq >/dev/null || { echo "user 'dnsmasq' missing: apt install dnsmasq" >&2; exit 3; }
  /usr/sbin/dnsmasq --test --conf-file="$HERE/direct-dnsmasq.conf" || { echo "dnsmasq rejects the config" >&2; exit 4; }
  /usr/sbin/nft -c -f "$HERE/direct-link.nft" || { echo "nft rejects the rule file" >&2; exit 4; }
fi
install -d -m 755 "$ETC/lwir" "$UNIT_DIR"
install -m 644 "$HERE/direct-dnsmasq.conf" "$ETC/lwir/direct-dnsmasq.conf"
install -m 644 "$HERE/direct-link.nft" "$ETC/lwir/direct-link.nft"
install -m 644 "$HERE/$UNIT" "$UNIT_DIR/$UNIT"
if [ "$DRY" != 1 ]; then
  systemctl daemon-reload
  systemctl enable --now "$UNIT"
  sleep 2
  systemctl --no-pager --full status "$UNIT" | head -12
fi
echo "installed. DHCP/DNS: 10.42.0.1 on the direct port; port 80 there -> 8091 (portal). Remove again with: sudo bash $0 --uninstall"
