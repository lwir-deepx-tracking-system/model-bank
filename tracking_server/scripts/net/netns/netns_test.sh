#!/bin/bash
# Namespace test of the whole direct-cable chain, as a normal user (no root, no change to the host network):
#   bash scripts/net/netns/netns_test.sh        (from tracking_server/)
# "board" = a user+network namespace with a veth port named enP4p65s0 (same name as the real port, so the nft rule is the real one);
# "PC"    = a second network namespace on the other end of the veth.
# Runs the real dnsmasq (the real conf with scratch paths), lwir_netd, the real src/web/direct_listeners.py (page stub) and the
# real nft rule. NetworkManager is replaced by `ip addr add` (it only assigns the addresses).
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
T="$(mktemp -d /tmp/lwir_netns.XXXXXX)"
if [ "${1:-}" != "--inner" ]; then
  exec unshare --user --map-root-user --net --mount --pid --fork --mount-proc bash "$0" --inner "$T" "$ROOT"
fi
T="$2"; ROOT="$3"
PASS=0; FAIL=0
ok()   { echo "PASS  $1"; PASS=$((PASS + 1)); }
bad()  { echo "FAIL  $1  ($2)"; FAIL=$((FAIL + 1)); }
check() { local d="$1"; shift; local out; if out=$("$@" 2>&1); then ok "$d"; else bad "$d" "$(echo "$out" | tail -4 | tr '\n' ' ')"; fi; }
expect() { local d="$1" want="$2"; shift 2; local out; out=$("$@" 2>&1); if echo "$out" | grep -q -- "$want"; then ok "$d"; else bad "$d" "wanted '$want', got: $(echo "$out" | tail -4 | tr '\n' ' ')"; fi; }

ip link set lo up
mount -t sysfs sysfs /sys   # /sys must show THIS network namespace (netd reads /sys/class/net/<port>/carrier)
unshare --net sleep 3000 & CPID=$!
sleep 0.5
C() { nsenter --net=/proc/$CPID/ns/net -- "$@"; }
ip link add enP4p65s0 type veth peer name client0
ip link set client0 netns $CPID
C ip link set lo up

# resolv.conf of the PC points to the board (only visible in this private mount namespace)
echo "nameserver 10.42.0.1" > $T/resolv.conf
mount --bind $T/resolv.conf "$(readlink -f /etc/resolv.conf)"

# ---- board side services, started BEFORE the cable / addresses exist
sed -e "s#/var/lib/misc/lwir-direct.leases#$T/leases#" -e "s#/run/lwir-direct-dnsmasq.pid#$T/dnsmasq.pid#" $ROOT/scripts/net/direct-dnsmasq.conf > $T/dnsmasq.conf
dnsmasq --test --conf-file=$T/dnsmasq.conf > $T/dnsmasq.test 2>&1 && ok "dnsmasq accepts the real config" || bad "dnsmasq config test" "$(cat $T/dnsmasq.test)"
gcc -shared -fPIC -o $T/nosetgroups.so $ROOT/scripts/net/netns/nosetgroups.c   # setgroups() is denied in an unprivileged user namespace
LD_PRELOAD=$T/nosetgroups.so dnsmasq --keep-in-foreground --conf-file=$T/dnsmasq.conf --user=root --group=root --log-facility=- > $T/dnsmasq.log 2>&1 & DPID=$!
nft -f $ROOT/scripts/net/direct-link.nft && ok "nft rule file loads (port 80 -> 8091 on the direct port)" || bad "nft load" "$?"
export LWIR_DIRECT_IFACE=enP4p65s0 LWIR_NETD_LOCK=$T/netd.lock LWIR_NET_STATE=$T/state.json LWIR_PORTAL_SILENCE_S=6 LWIR_NET_CHECK_S=1
(cd $ROOT && exec python3 -B -m src.net.netd >> $T/netd.log 2>&1) & NPID=$!
(cd $ROOT && exec python3 -B scripts/net/netns/page_stub.py >> $T/page.log 2>&1) & PPID_=$!
sleep 2
kill -0 $DPID 2>/dev/null && ok "dnsmasq runs although 10.42.0.1 does not exist yet (bind-dynamic)" || bad "dnsmasq alive before the address exists" "$(tail -3 $T/dnsmasq.log)"
kill -0 $NPID 2>/dev/null && ok "lwir_netd runs before the cable" || bad "netd alive" "$(tail -3 $T/netd.log)"
expect "page listeners are bound before any address exists (IP_FREEBIND)" "10.42.0.1:8090" ss -ltn
expect "page listener only on the allowed addresses (nothing on 0.0.0.0:8090)" "192.168.50.2:8090" ss -ltn
ss -ltn | grep -q "0.0.0.0:8090\|\*:8090" && bad "no wildcard listener" "found 0.0.0.0:8090" || ok "no wildcard listener on 8090"
sleep 1.2
expect "supervisor: no internet in the namespace -> lan-only" '"mode": "lan-only"' cat $T/state.json

# ---- the cable is plugged: NetworkManager's part = link up + addresses
ip link set enP4p65s0 up; C ip link set client0 up
for a in 10.42.0.1/24 169.254.42.1/16 192.168.50.2/24 192.168.77.2/24; do ip addr add $a dev enP4p65s0; done
sleep 2
# ---- the PC asks for an address (DHCP)
cat > $T/dhscript <<'EOF'
#!/bin/bash
case "$reason" in BOUND|RENEW|REBIND|REBOOT)
  { echo "ip=$new_ip_address"; echo "mask=$new_subnet_mask"; echo "routers=$new_routers"; echo "dns=$new_domain_name_servers"; } > __ENVFILE__
  pfx=$(python3 -c "import ipaddress,sys;print(ipaddress.ip_network('0.0.0.0/%s'%sys.argv[1]).prefixlen)" "$new_subnet_mask")
  ip addr add "$new_ip_address/$pfx" dev "$interface" 2>/dev/null ;; esac
exit 0
EOF
sed -i "s#__ENVFILE__#$T/dhcp.env#" $T/dhscript
chmod +x $T/dhscript
export ENVFILE=$T/dhcp.env
C dhclient -d -v -1 -lf $T/c.leases -pf $T/c.pid -sf $T/dhscript client0 > $T/dhclient.log 2>&1 & CLPID=$!
for i in $(seq 1 25); do [ -s $T/dhcp.env ] && break; sleep 1; done
expect "DHCP: the PC gets a 10.42.0.x address" "ip=10.42.0." cat $T/dhcp.env
expect "DHCP: DNS server is the board" "dns=10.42.0.1" cat $T/dhcp.env
grep -q "routers=$" $T/dhcp.env && ok "DHCP: no router handed out (PC keeps its own internet route)" || bad "no router" "$(cat $T/dhcp.env | tr '\n' ' ')"
C ip route | grep -q default && bad "PC has no default route from us" "$(C ip route | tr '\n' ' ')" || ok "PC got no default route"
PCIP=$(grep '^ip=' $T/dhcp.env | cut -d= -f2)

# ---- the page at the fixed address (listener started before the address existed)
expect "page at http://10.42.0.1:8090/netinfo from the PC" '"client_local": true' C curl -s -m 4 http://10.42.0.1:8090/netinfo
expect "netinfo reports lan-only" '"mode": "lan-only"' C curl -s -m 4 http://10.42.0.1:8090/netinfo
expect "netinfo offers the PC program download" '"pc_download": true' C curl -s -m 4 http://10.42.0.1:8090/netinfo
C curl -s -m 4 -o $T/pc.zip http://10.42.0.1:8090/pc/lwir_connect.zip
check "PC program zip downloads and holds the .bat and the .ps1" python3 -c "import zipfile;n=zipfile.ZipFile('$T/pc.zip').namelist();print(n);assert 'lwir_connect.ps1' in n and 'lwir_connect.bat' in n"
expect "PC .ps1 downloads as is" "param(" C curl -s -m 4 http://10.42.0.1:8090/pc/lwir_connect.ps1
expect "unknown /pc/ file is refused" "404" C curl -s -m 4 -o /dev/null -w "%{http_code}" --path-as-is http://10.42.0.1:8090/pc/../src/app.py
# allowlist: a source address outside 10.42.0.0/24 talking to 10.42.0.1 is dropped
C ip addr add 192.168.99.5/24 dev client0
C ip route add 10.42.0.1/32 dev client0 src 192.168.99.5 2>/dev/null
ip route add 192.168.99.0/24 dev enP4p65s0   # so the board could answer: only the allowlist stops it
out=$(C curl -s -m 3 --interface 192.168.99.5 -o /dev/null -w "%{http_code}" http://10.42.0.1:8090/netinfo 2>&1); [ "$out" != 200 ] && ok "allowlist: client 192.168.99.5 is not served on 10.42.0.1 (got $out)" || bad "allowlist" "got $out"
C ip route del 10.42.0.1/32 dev client0 2>/dev/null; ip route del 192.168.99.0/24 dev enP4p65s0; C ip addr del 192.168.99.5/24 dev client0
C ip route add 224.0.0.0/4 dev client0 2>/dev/null
C ip route add 255.255.255.255/32 dev client0 2>/dev/null
kill $CLPID 2>/dev/null
# team PC with a fixed address keeps working (GUI link)
C ip addr add 192.168.50.1/24 dev client0
check "team link: ping 192.168.50.2 from 192.168.50.1" C ping -c1 -W2 -I 192.168.50.1 192.168.50.2
expect "team link: page at http://192.168.50.2:8090/ from 192.168.50.1" '"client_local": true' C curl -s -m 4 --interface 192.168.50.1 http://192.168.50.2:8090/netinfo
C ip addr del 192.168.50.1/24 dev client0

# ---- DNS of the connectivity checks
expect "DNS: www.msftconnecttest.com -> board" "10.42.0.1" C dig +short +time=2 +tries=1 @10.42.0.1 www.msftconnecttest.com
expect "DNS: dns.msftncsi.com -> 131.107.255.255" "131.107.255.255" C dig +short +time=2 +tries=1 @10.42.0.1 dns.msftncsi.com
expect "DNS: captive.apple.com -> board" "10.42.0.1" C dig +short +time=2 +tries=1 @10.42.0.1 captive.apple.com
expect "DNS: connectivitycheck.gstatic.com -> board" "10.42.0.1" C dig +short +time=2 +tries=1 @10.42.0.1 connectivitycheck.gstatic.com
expect "DNS: any other name is refused (nothing is forwarded)" "REFUSED" C dig +time=2 +tries=1 @10.42.0.1 www.example.com

# ---- captive portal (real DNS through the board's dnsmasq, port 80 through the nft redirect)
P="C curl -s -m 4 -i"
expect "portal: Windows probe before the page was shown -> 302 to the page" "Location: http://10.42.0.1:8090/" $P http://www.msftconnecttest.com/connecttest.txt
expect "portal: the browser Windows opens (/redirect) -> 302 to the page" "302" $P http://www.msftconnecttest.com/redirect
expect "portal: Windows probe after the visit -> 200 'Microsoft Connect Test'" "Microsoft Connect Test" $P http://www.msftconnecttest.com/connecttest.txt
expect "portal: legacy NCSI probe after the visit -> 'Microsoft NCSI'" "Microsoft NCSI" $P http://www.msftncsi.com/ncsi.txt
expect "portal: Android probe after the visit -> 204" "204" $P http://connectivitycheck.gstatic.com/generate_204
expect "portal: Apple probe is never 'Success' (the sign-in window must stay) -> 302" "302" $P http://captive.apple.com/hotspot-detect.html
expect "portal: status for the PC watcher says shown" '"shown": true' C curl -s -m 4 http://10.42.0.1:8091/lwir-status
sleep 7
expect "portal: after the silence time the probe is redirected again" "302" $P http://www.msftconnecttest.com/connecttest.txt
$P http://www.msftconnecttest.com/redirect > /dev/null
expect "portal: shown again, probe answered normally" "Microsoft Connect Test" $P http://www.msftconnecttest.com/connecttest.txt
# cable re-plugged: carrier down/up clears the table
C ip link set client0 down; sleep 2; C ip link set client0 up; sleep 3
C ip addr show dev client0 | grep -q "$PCIP" || C ip addr add $PCIP/24 dev client0
C ip route add 224.0.0.0/4 dev client0 2>/dev/null; C ip route add 255.255.255.255/32 dev client0 2>/dev/null   # routes of a downed link are gone
expect "portal: after the cable was re-plugged the probe is redirected again" "302" $P http://www.msftconnecttest.com/connecttest.txt
grep -q "carrier" $T/netd.log && ok "netd logged the carrier change" || bad "carrier log" "$(tail -3 $T/netd.log)"
# bare address typed on port 80
expect "portal: http://10.42.0.1/ (no port) -> page" "Location: http://10.42.0.1:8090/" $P http://10.42.0.1/

# ---- discovery
expect "beacon: the PC hears the board's UDP beacon" '"addr": "10.42.0.1"' C python3 $ROOT/scripts/net/netns/client_checks.py beacon
expect "discover: broadcast question is answered" '"port": 8090' C python3 $ROOT/scripts/net/netns/client_checks.py discover
expect "discover: unicast question is answered" '"_from": "10.42.0.1"' C python3 $ROOT/scripts/net/netns/client_checks.py discover 10.42.0.1
expect "mDNS: orangepi.local A answered with 10.42.0.1, AAAA answered negatively (NSEC)" '"address": "10.42.0.1"' C python3 $ROOT/scripts/net/netns/client_checks.py mdns orangepi.local
HN=$(hostname | cut -d. -f1)
expect "mDNS: <hostname>.local answered" '"address": "10.42.0.1"' C python3 $ROOT/scripts/net/netns/client_checks.py mdns $HN.local

# ---- the PowerShell logic, ported (every step on its own)
F="C python3 $ROOT/pc_connect/check_find_pi.py"
expect "find: step 1 fixed address" "FOUND 10.42.0.1 via fixed" $F --steps fixed
expect "find: step 2 mDNS name (fixed address skipped)" "FOUND 10.42.0.1 via mdns" $F --steps mdns
expect "find: step 3 UDP beacon (fixed and mDNS skipped)" "FOUND 10.42.0.1 via beacon" $F --steps beacon --beacon-seconds 5
expect "find: all steps in order, first hit wins" "FOUND 10.42.0.1 via fixed" $F
expect "find: nothing there -> NOTFOUND" "NOTFOUND" $F --fixed 10.42.0.99 --mdns-name nothing.local --steps fixed,mdns --port 8090
# PC that has only a link-local address (DHCP failed)
C ip addr flush dev client0
C ip addr add 169.254.7.7/16 dev client0
expect "find: link-local PC reaches the board at 169.254.42.1 (steps 1-3 off)" "FOUND 169.254.42.1 via linklocal" $F --steps linklocal
expect "find: link-local PC, beacon answers too" "FOUND 169.254.42.1 via beacon" $F --steps beacon --beacon-seconds 5
C ip addr flush dev client0

# ---- unplug / replug: listeners and services survive, address comes back
ip addr flush dev enP4p65s0; ip link set enP4p65s0 down
sleep 2
for a in 10.42.0.1/24 169.254.42.1/16 192.168.50.2/24 192.168.77.2/24; do ip addr add $a dev enP4p65s0; done
ip link set enP4p65s0 up
C ip addr add 10.42.0.77/24 dev client0; C ip link set client0 up
sleep 3
expect "replug: the page answers again without any restart" '"client_local": true' C curl -s -m 4 http://10.42.0.1:8090/netinfo
expect "replug: portal answers again" "302" $P http://www.msftconnecttest.com/connecttest.txt

# ---- supervisor both ways (fake internet host on loopback)
ip addr add 1.1.1.1/32 dev lo
nc -l -k 443 > /dev/null 2>&1 & NCPID=$!
sleep 4
expect "supervisor: internet reachable -> online" '"mode": "online"' cat $T/state.json
kill $NCPID 2>/dev/null; ip addr del 1.1.1.1/32 dev lo
sleep 6
expect "supervisor: internet gone again -> lan-only" '"mode": "lan-only"' cat $T/state.json
expect "netinfo follows the state file" '"mode": "lan-only"' C curl -s -m 4 http://10.42.0.1:8090/netinfo

# ---- the root installer in dry mode (files only)
D=$T/rootdest
LWIR_ROOT_DRY=1 LWIR_ROOT_ETC=$D/etc LWIR_ROOT_UNIT_DIR=$D/units bash $ROOT/scripts/net/install_direct_link_root.sh > $T/inst.out 2>&1 && [ -f $D/etc/lwir/direct-dnsmasq.conf ] && [ -f $D/units/lwir-direct-link.service ] && ok "installer (dry run) places the files" || bad "installer dry run" "$(cat $T/inst.out)"
LWIR_ROOT_DRY=1 LWIR_ROOT_ETC=$D/etc LWIR_ROOT_UNIT_DIR=$D/units bash $ROOT/scripts/net/install_direct_link_root.sh --uninstall > /dev/null 2>&1 && [ ! -e $D/etc/lwir/direct-dnsmasq.conf ] && ok "installer --uninstall removes them" || bad "uninstall" ""

echo "---- netd log"; tail -n 12 $T/netd.log
echo "---- dnsmasq log"; tail -n 5 $T/dnsmasq.log
echo "RESULT pass=$PASS fail=$FAIL"
kill $DPID $NPID $PPID_ $CPID $CLPID 2>/dev/null
[ "$FAIL" = 0 ]
