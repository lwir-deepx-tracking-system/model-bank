# Shared web listener selection for the manual launcher and watchdog.
lwir_live_lan_bind() {
  local address="${LWIR_LIVE_LAN_BIND:-}"
  [ -n "$address" ] || return 0
  ip -4 -o addr show 2>/dev/null | awk -v wanted="$address" '
    $2 != "tailscale0" {
      split($4, cidr, "/")
      split(cidr[1], octets, ".")
      lan = octets[1] == 10 || (octets[1] == 172 && octets[2] >= 16 && octets[2] <= 31) || (octets[1] == 192 && octets[2] == 168)
      link = octets[1] == 169 && octets[2] == 254 && octets[3] >= 1 && octets[3] <= 254
      if ((lan || link) && cidr[1] == wanted) { print cidr[1]; exit }
    }'
}

lwir_live_bind() {
  case "${LWIR_LIVE_NETWORK:-default}" in
    default)
      local address="${LWIR_LIVE_BIND:-}"
      if [ -z "$address" ]; then
        address=$(ip -4 -o addr show tailscale0 2>/dev/null | awk 'NR == 1 { split($4, parts, "/"); print parts[1] }')
      fi
      if [ -n "$address" ]; then
        printf '%s\n' "$address"
      else
        # 11-22: neither Tailscale nor an assigned LAN address (the internet and Tailscale are gone, no cable): the loopback listener keeps the
        # server startable and readable by keepalive.sh; the direct-cable listeners and the remembered Tailscale address are opened by the server itself
        # (src/web/direct_listeners.py, IP_FREEBIND), so they answer as soon as their address appears, without a restart.
        address=$(lwir_live_lan_bind)
        printf '%s\n' "${address:-127.0.0.1}"
      fi
      ;;
    lan) lwir_live_lan_bind ;;
    *) return 1 ;;
  esac
}
