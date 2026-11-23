# 인터넷·Tailscale 없이 쓰기: 직결 케이블 (2026-11-22)

목표: 인터넷과 Tailscale 이 없어도 서버가 돌고, 보드의 **둘째 랜 포트에 PC를 케이블로 꽂기만 하면** 주소를 치지 않고 화면이 PC에서 열린다.

## 한눈에

```
PC ──랜 케이블── 보드 둘째 포트 (enP4p65s0)             첫째 포트 (enP3p49s0) 와 Tailscale 은 그대로 (원격 접속)
                 │
                 ├ NetworkManager 프로필 lwir-direct-link : 10.42.0.1/24 + 169.254.42.1/16 + 192.168.50.2/24 + 192.168.77.2/24
                 ├ lwir-direct-link.service (root, 한 번 설치) : 우리 dnsmasq = DHCP(10.42.0.20~200) + 연결 확인 이름 DNS, 80번 → 8091 전달(nft)
                 ├ lwir_netd (일반 사용자, 서버와 함께 켜짐) : 5초마다 인터넷·Tailscale 점검(online | lan-only), '로그인 창' 응답기(8091),
                 │                                           UDP 신호(47890), mDNS(orangepi.local)
                 └ 추적 서버 : 10.42.0.1:8090 외 여러 주소에 동시에 귀를 연다 (IP_FREEBIND: 케이블이 없어도 시작되고, 주소가 생기면 저절로 응답)
```

PC에서 일어나는 일: DHCP 로 10.42.0.x 를 받는다 → 운영체제가 인터넷 연결 확인 주소(예 `www.msftconnecttest.com`)를 묻는다 → 보드가 자기 주소로 답하고 `302 → http://10.42.0.1:8090/` 를 준다 → 운영체제가 "로그인이 필요한 네트워크"로 보고 브라우저를 연다(운영체제 동작은 아래 표). 한 번 화면을 보여 준 PC는 10분 동안 정상 답을 받아 같은 창이 다시 뜨지 않는다. 케이블을 다시 꽂으면 처음부터.

## 주소 · 이름 (사용자가 칠 것은 없다)

| 길 | 주소 | 비고 |
| --- | --- | --- |
| 자동 | 운영체제의 로그인 창 | 아래 "운영체제별" 참고 |
| 고정 | `http://10.42.0.1:8090/` | 직결 포트의 고정 주소. DHCP 가 안 될 때도 `169.254.42.1:8090` |
| 이름 | `http://orangepi.local:8090/` | 우리 mDNS 응답기 (avahi 설치 필요 없음). `<호스트이름>.local` 도 됨 |
| 프로그램 | 화면 영상 위 **PC 연결 프로그램** 링크 | `lwir_connect.bat` + `.ps1` (zip). 더블클릭하면 위 길을 차례로 찾아 브라우저를 연다 |

## 사용자 점검표 (케이블 시험, 6줄)

1. 보드의 **둘째 랜 포트**(지금 비어 있는 것, 이름 enP4p65s0)에 PC를 랜 케이블 하나로 직접 꽂는다. 첫째 포트 케이블과 Tailscale 은 건드리지 않는다.
2. PC 어댑터는 "IP 주소 자동으로 받기"(기본값)로 둔다. 팀 PC GUI 용 고정 주소 192.168.50.1 은 예전처럼 그대로 동작한다.
3. 10~30초 기다린다. 보여야 하는 것: Windows 는 브라우저가 저절로 열리거나 "추가 로그인 정보 필요" 알림(누르면 브라우저)이 뜬다.
4. 안 열리면 브라우저에 `10.42.0.1:8090` (처음 한 번)을 치거나, 이 PC에 받아 둔 `lwir_connect.bat` 을 더블클릭한다.
5. 화면에 추적 영상이 나오고 짐벌 조작이 되면 성공이다. 인터넷이 없으면 화면 위쪽에 "LAN 모드" 표시와 "PC 연결 프로그램" 링크가 보인다.
6. 알림·창이 귀찮으면 보드에서 `touch ~/.lwir_portal_off` (바로 적용, 지우면 다시 켜짐). 케이블을 빼면 모든 것이 원래대로다.

## 처음 한 번만 root 가 필요하다 (사용자 sudo)

보드의 `orangepi` 에는 암호 없는 sudo 가 없다(`sudo -n` 거부). 그래서 일반 사용자로 되는 것(NetworkManager 프로필, 서버, lwir_netd, 화면, PC 프로그램)은 모두 적용했고, root 가 필요한 둘(우리 DHCP/DNS 서비스, 80번 전달)은 스크립트 하나로 묶었다.

```
ssh -t <보드 계정>@<보드 주소> "sudo bash ~/model-bank/model-bank/tracking_server/scripts/net/install_direct_link_root.sh"
```

- 설치되는 것: `/etc/lwir/direct-dnsmasq.conf`, `/etc/lwir/direct-link.nft`, `/etc/systemd/system/lwir-direct-link.service` (켜고 부팅 때 자동 시작).
- 새 패키지는 필요 없다 (이미 있는 `dnsmasq`, `dnsmasq-base`, `nftables`, 사용자 `dnsmasq` 를 쓴다). avahi 는 설치하지 않는다(mDNS 는 lwir_netd 가 대신 답한다).
- 확인: `sudo bash …/install_direct_link_root.sh --check`. 지우기: `… --uninstall` (서비스·파일·nft 표가 지워진다).
- 이 단계 전에도 케이블을 꽂으면 NetworkManager 프로필이 주소(10.42.0.1)를 주고 서버가 응답한다. 없는 것은 PC에 자동 주소(DHCP)와 로그인 창뿐이다.

## 왜 NetworkManager "shared" 방식이 아닌가

`ipv4.method=shared` 는 포트에 주소를 하나만 남긴다(팀 PC GUI 링크 192.168.50.2 와 옛 직결 주소 192.168.77.2 가 사라진다), IP 전달과 NAT 를 켜서 직결 PC가 첫째 포트·Tailscale 쪽으로 나갈 수 있게 하고, 추가 설정을 전체에 적용되는 `/etc/NetworkManager/dnsmasq-shared.d` 에서만 읽는다(root 필요). 그래서 프로필은 `manual`(주소 4개)로, DHCP·DNS 는 이 포트에만 귀를 여는 우리 dnsmasq 로 했다. 전달·NAT 없음, 라우터 정보도 안 준다(PC는 자기 인터넷 길을 그대로 쓴다). `ip_forward` 는 0 그대로다.

## 운영체제별 동작 (문서 근거, 그리고 증명되지 않은 것)

- **Windows 10/11 (NCSI)**: Microsoft 문서(NCSI FAQ, "NCSI active probes")는 `www.msftconnecttest.com` 이름 풀이 → `GET http://www.msftconnecttest.com/connecttest.txt` 가 본문 `Microsoft Connect Test` 인지 확인 → `dns.msftncsi.com` 이름 풀이(문서의 정상 답 131.107.255.255)를 적는다. 하나라도 어긋나면 "제한된 연결/로그인 필요"로 본다. 같은 문서가 "Windows 는 로그인이 필요한 네트워크에서 기본 브라우저를 연다. 이 동작은 의도된 것이다. 예전에는 작은 알림만 띄웠다"고 쓰고, 열리는 주소는 `http://www.msftconnecttest.com/redirect` 이다. 우리 답이 그 경로를 받으면 화면 주소로 302 를 준다. Windows 11 은 HTTP 확인만 쓴다(이름 풀이는 보낼 곳을 찾는 데만). **그러나** 한 외부 글(Eric Lawrence, 2022)은 Wi-Fi 목록에 링크만 보일 뿐 브라우저가 저절로 열리지 않는다고 쓰고, Microsoft 문서는 공중 Wi-Fi 를 예로 들 뿐 **유선 이더넷에서도 같은지 쓰지 않았다.** 사용자가 보게 될 것: (가) 브라우저가 저절로 열림, (나) 작업 표시줄 네트워크 아이콘에 경고 + "추가 로그인 정보 필요" 알림(누르면 브라우저가 열림), (다) 둘 다 없음(이 경우 `10.42.0.1:8090` 또는 PC 연결 프로그램). 어느 쪽인지는 실제 PC와 케이블로만 알 수 있다.
- **macOS / iOS**: `captive.apple.com/hotspot-detect.html` 이 `Success` 가 아니면 Captive Network Assistant 창이 저절로 뜬다고 알려져 있다(Wi-Fi 기준). 유선 이더넷에서도 뜨는지 문서를 못 찾았다: **증명 안 됨.** 이 창은 정상 답이 오면 스스로 닫히므로 보드는 Apple 확인에는 정상 답을 주지 않는다(`LWIR_PORTAL_APPLE_SILENCE=1` 로 바꿀 수 있음).
- **Android / ChromeOS / Firefox / Ubuntu 데스크톱(NetworkManager)**: `generate_204`, `detectportal.firefox.com`, `nmcheck.gnome.org`, `connectivity-check.ubuntu.com` 도 같은 방식으로 답한다. 유선에서 창이 뜨는지는 확인하지 않았다.
- 모든 운영체제 공통: 창이 안 떠도 고정 주소·이름·PC 연결 프로그램이 있다.

## PC 연결 프로그램 (Windows 우선)

- 화면에서 받는다: 영상 위 "PC 연결 프로그램" (zip = `lwir_connect.bat` + `lwir_connect.ps1` + `lwir_connect.sh`). **LAN·직결로 들어온 PC에게만** 링크와 내려받기(`/pc/…`)를 준다. Tailscale 로 들어온 PC는 이미 되는 주소를 갖고 있어 이 프로그램이 필요 없고, 보드의 파일을 인터넷 쪽에 내놓을 이유도 없다(Tailscale·공개 주소는 404).
- exe 가 아니라 스크립트다(서명 없는 exe 는 SmartScreen 이 막는다). 내려받은 파일은 "인터넷에서 받은 파일" 표시가 붙어 처음 실행할 때 한 번 "실행" 을 눌러야 할 수 있다.
- 찾는 순서: `10.42.0.1:8090` → `orangepi.local` → UDP 신호 5초(듣기 + `LWIR_DISCOVER` 질문: Windows 방화벽은 내가 먼저 묻고 온 답은 통과시키므로 방화벽 창 없이 된다) → 링크로컬 `169.254.42.1`. 찾으면 기본 브라우저로 연다. 각 주소가 정말 이 서버인지 `/netinfo`(옛 서버는 `/healthz`)로 확인한다.
- `lwir_connect.bat -Install`: **묻고 나서** Windows 시작 때 뒤에서 지켜보다가 보드가 연결되면 화면을 연다(시작 프로그램 바로가기, `-Uninstall` 로 지움, `-Status` 로 확인). 운영체제 로그인 창이 이미 화면을 열었으면(보드가 알려 준다) 또 열지 않고, 12초 안에 안 열리면 직접 연다. 그래서 로그인 창을 꺼 둔 PC도 한 번 설치하면 된다.
- 팀 PC GUI(pc-application): 이 저장소에는 그 프로그램의 설치 위치·실행 명령이 문서로 없어 **경로를 추측하지 않았다.** 프로그램 경로를 한 줄로 적은 `gui.txt`(스크립트 옆 또는 `%LOCALAPPDATA%\LWIRConnect\gui.txt`, 또는 환경 변수 `LWIR_GUI`)가 있으면 화면을 연 뒤 시작할지 묻는다. 주의: 팀 GUI 는 고정 주소 쌍(PC 192.168.50.1 ↔ 보드 192.168.50.2, `--gui-host`)으로 영상을 받는다. DHCP(10.42.0.x)로 연결한 PC에는 영상이 가지 않는다.

## 위험과 스위치

- **다른 DHCP 서버를 방해할 수 있다:** 둘째 포트를 공유기·스위치·회사 랜에 꽂으면 우리 dnsmasq 가 그 랜의 PC에게 10.42.0.x 를 나눠 준다(권한 DHCP). 직결 케이블 용도 말고는 꽂지 않는다.
- 이 포트의 DNS 는 연결 확인 이름 11개만 답하고 나머지는 거부(REFUSED)한다. google.com 같은 이름은 건드리지 않는다. 다만 PC가 Wi-Fi 와 이 케이블을 같이 쓸 때 Windows 가 이 포트의 DNS 도 묻는데, 거부 답은 무시되고 Wi-Fi 쪽 답을 쓴다.
- 보드는 로그인 창에 "정상" 을 10분 동안 주므로 Windows 가 이 선을 "인터넷 있음" 으로 표시할 수 있다(라우터를 안 주므로 실제 인터넷 길은 아님). 10분 뒤 확인이 다시 우리를 거쳐 알림이 한 번 더 뜰 수 있다.
- 스위치(재시작 없이): `touch ~/.lwir_portal_off` → 모든 확인에 정상 답, 운영체제 창이 안 뜬다(DHCP, 고정 주소, 이름, PC 연결 프로그램은 그대로). 지우면 다시 켜진다. 완전 제거: `sudo bash …/install_direct_link_root.sh --uninstall`, NetworkManager 프로필은 `bash scripts/net/nm_direct_profile.sh restore`.
- 보안: 80번 전달과 DHCP 는 이 포트에만 걸린다. 웹 화면은 0.0.0.0 에 열지 않는다: Tailscale 주소(그대로)와 아래 주소들에만 열고, 주소마다 **자기 네트워크의 PC만** 받는다(10.42.0.1 → 10.42.0.0/24 등, 밖에서 온 패킷은 끊는다). 팀 GUI 가 쓰는 5002 같은 포트는 바꾸지 않았다.

## 서버가 열어 두는 주소 (`--live-extra-binds`, 기본 `LWIR_LIVE_EXTRA_BINDS`)

기본: `10.42.0.1/24, 169.254.42.1/16, 192.168.50.2/24, 192.168.77.2/24` (직결 + 링크로컬 대체 + 팀 링크 + 옛 직결). 이 보드(live.env, 저장소에는 없음)는 여기에 첫째 포트 자기 주소 /24 를 더했다(윈도 PC의 연결 공유(ICS)로 이어진 PC가 Tailscale 없이도 화면을 열 수 있게; 포트 설정은 건드리지 않음). 마지막 Tailscale 주소는 `~/.lwir_tailscale_ip` 에 기억해 두었다가(lwir_netd 가 적음) 서버가 Tailscale 없이 시작돼도 그 주소를 미리 열어 둔다(돌아오면 재시작 없이 응답).
`/netinfo`(누구나): `{mode, direct_urls, client_local, pc_download}`. `/network`(기존 LAN 자동 전환용)은 그대로다.

## 인터넷·Tailscale 이 없으면 무엇이 달라지나 (서버가 인터넷을 쓰는 곳)

- **서버 코드는 인터넷을 쓰지 않는다**: 모델 내려받기·업데이트 확인·외부 호출 없음(`urlopen`, `requests`, 외부 URL 없음), 화면 페이지에도 외부 글꼴·스크립트가 없다. 녹화 재생·라벨은 보드의 디스크에서 읽는다.
- **시간**: 보드에는 RTC 가 없고 시계는 chrony(NTP)가 맞춘다. 인터넷 없이 켜면 시계가 틀리고(녹화·로그 이름의 시각), 인터넷이 돌아오면 chrony 가 한 번에 맞춘다. 서버·keepalive 의 판단은 `/proc/uptime` 을 쓰므로 영향 없다.
- **Tailscale**: 원격 접속만. 끊기면 직결·LAN 주소만 남는다.
- **고친 것 1 (시작)**: 예전에는 Tailscale 주소가 없으면 `run_real_gui.sh` 가 "no page address" 로 서버를 켜지 않았다(인터넷 없이 부팅하면 서버가 안 뜸). 이제 주소가 없으면 루프백(127.0.0.1)으로 시작하고 직결 주소는 서버가 직접 연다. Tailscale 이 있으면 예전과 똑같다.
- **고친 것 2 (keepalive)**: 위 주소 선택이 비어 있지 않으므로 Tailscale 이 끊겨도 keepalive 가 루프백으로 /healthz 를 읽는다(전에는 주소가 비면 읽지 못했다).
- **고친 것 3 (이름 풀이)**: 서버가 귀를 열 때 `getfqdn()` 역방향 이름 풀이가 DNS 응답(시간 초과)을 기다려 시작이 수 초~수십 초 늦어질 수 있다(인터넷이 막혀 DNS 가 답하지 않을 때). 건너뛴다. 늦어지는 시간은 재 보지 않았다.
- 같은 이유로 Tailscale 주소에만 열려 있는 **라벨 서버(8770)** 는 직결 케이블에서 보이지 않는다(추적 화면의 녹화 재생은 된다). 인터넷 없이 `git pull`·`apt`·`gh` 는 안 된다: 배포는 scp 로.

## 장치 적용 상태 (2026-11-22)

- 적용함(일반 사용자 권한으로 되는 것 전부): 코드(main) + 서버 재시작 1번 + keepalive 재시작 + `live.env` 의 `LWIR_LIVE_EXTRA_BINDS` + NetworkManager 프로필 `lwir-direct-link`(둘째 포트 enP4p65s0 한정, 첫째 포트·Tailscale·옛 프로필 그대로).
- **아직 안 함 = 사용자 sudo 필요:** `install_direct_link_root.sh` (DHCP, 로그인 창 DNS, 80번 전달). 그 전에는 케이블을 꽂으면 보드는 10.42.0.1 로 응답하지만 PC가 주소를 자동으로 받지 못하고 로그인 창도 뜨지 않는다.
- 장치에서 직접 확인한 것: NM 이 주소 4개(manual)를 주면(임시 더미 장치 시험) 실행 중인 서버가 재시작 없이 10.42.0.1:8090, 169.254.42.1:8090, 192.168.50.2:8090 에 응답하고 `/netinfo` 의 `direct_urls` 에 나타난다. 프로필을 지우면 다시 닫힌다. 로그인 창 응답기(8091)는 302 를 준다. Tailscale 쪽 접속에는 `pc_download` 가 false 이고 `/pc/…` 는 404 다.

## 파일

`src/net/` (netd, supervisor, portal, discovery, netutil, test_net), `src/web/direct_listeners.py`, `pc_connect/` (lwir_connect.bat/.ps1/.sh, check_find_pi.py), `scripts/net/` (dnsmasq 설정, nft, systemd 서비스, root 설치, NM 프로필, 시험), `page/js/17_net_info.js`(+html·css 조각).

## 시험 (2026-11-22)

- 단위 시험 `python3 -m unittest src.net.test_net src.web.test_direct_listeners src.web.test_lan_listener` + `bash scripts/net/test_live_bind.sh`, 웹 전체 113개, 기존 keepalive 시험 2개.
- **네임스페이스 시험** `bash scripts/net/netns/netns_test.sh` (일반 사용자, 보드의 네트워크는 건드리지 않음): 사용자·네트워크 네임스페이스 안에 이름이 같은 포트 `enP4p65s0` (veth) 를 만들고 실제 dnsmasq(실제 설정), 실제 nft 규칙, lwir_netd, 실제 `direct_listeners` 를 돌린다. 반대쪽 네임스페이스가 PC: dhclient 로 주소를 받고, 80번 확인 요청, UDP 신호, mDNS, 링크로컬, 케이블 뽑았다 꽂기, 팀 링크 고정 주소를 확인한다. 57개 모두 통과. NetworkManager 만 `ip addr add` 로 대신했다.
- PowerShell 은 이 저장소의 PC(Windows PowerShell 5.1)에서 구문 검사, 보드 Tailscale 주소로 찾기·못 찾기, 전체 순서, 설치/상태/제거(임시 폴더), 감시(보드 감지 → 화면 열기), 하나만 도는지(뮤텍스)를 시험했고, 같은 논리를 `pc_connect/check_find_pi.py` 로 옮겨 네임스페이스에서 돌렸다.
- **실제 PC와 케이블이 있어야만 증명되는 것**: Windows/macOS 가 유선에서 로그인 창·브라우저를 실제로 띄우는지(위 표), 실제 NIC 의 DHCP 와 Windows 방화벽, SmartScreen 표시, `.local` 이름 풀이 시간, 둘째 포트의 하드웨어. NetworkManager 가 실제 케이블에서 프로필(manual + 주소 4개)을 적용하는지.

## 되돌리기

- NM 프로필: `bash scripts/net/nm_direct_profile.sh restore` (옛 프로필 `lwir-pc-link` 등이 이 포트를 다시 맡는다). 적용할 때는 `apply --rollback-after 180` 로 타이머가 자동으로 지운다(`confirm` 으로 확정).
- 서버: 이 변경 이전 커밋으로 `git reset --hard` 후 같은 재시작(배포 안전 스크립트가 3분 안에 /healthz 가 안 돌아오면 자동으로 한다). 보드 백업: `~/claude_backup_20261122_net/` (NM 프로필 덤프, live.env, keepalive.sh, run_real_gui.sh, 서버 실행줄, 주소·경로).
- root 부분: `--uninstall`. live.env 의 `LWIR_NETD=0` 은 lwir_netd 를 끈다.
