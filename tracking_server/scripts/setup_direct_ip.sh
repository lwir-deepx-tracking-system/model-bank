#!/bin/bash
# Configure, but do not activate, an isolated static Ethernet profile.
# Usage on PC:        sudo bash setup_direct_ip.sh <wired-interface> 192.168.77.1/24
# Usage on Orange Pi: sudo bash setup_direct_ip.sh <wired-interface> 192.168.77.2/24
set -euo pipefail
[ "$#" -eq 2 ] || { echo "usage: sudo bash $0 <wired-interface> <ipv4/prefix>"; exit 2; }
IFACE="$1"
ADDRESS="$2"
PROFILE='LWIR direct LAN'
command -v nmcli >/dev/null || { echo 'NetworkManager nmcli is required'; exit 2; }
if nmcli -g connection.id connection show "$PROFILE" >/dev/null 2>&1; then
  nmcli connection modify "$PROFILE" connection.interface-name "$IFACE" \
    ipv4.method manual ipv4.addresses "$ADDRESS" ipv4.never-default yes \
    ipv6.method disabled connection.autoconnect yes
else
  nmcli connection add type ethernet ifname "$IFACE" con-name "$PROFILE" \
    ipv4.method manual ipv4.addresses "$ADDRESS" ipv4.never-default yes \
    ipv6.method disabled connection.autoconnect yes
fi
echo "Prepared $PROFILE on $IFACE with $ADDRESS. Verify the active interface has this address; another NetworkManager profile may own the cable."
