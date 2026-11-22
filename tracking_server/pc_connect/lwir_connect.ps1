<#
  LWIR 연결 (PC 프로그램) - 주소를 치지 않고 오렌지파이 추적 화면을 찾아 엽니다.
  설치 없음, exe 없음 (SmartScreen 이 막는 서명 없는 exe 를 쓰지 않으려고 스크립트로 만들었습니다).

  찾는 순서 (먼저 찾은 것을 씁니다):
    1) 10.42.0.1:8090          보드의 직결 포트 고정 주소 (케이블을 PC에 직접 꽂았을 때)
    2) orangepi.local          mDNS 이름 (Windows 10/11, macOS 가 기본으로 풀어 줍니다)
    3) UDP 신호 (47890)        보드가 2초마다 방송하는 신호를 5초 듣고, 같은 시간에 'LWIR_DISCOVER' 질문도 보냅니다
    4) 링크로컬 이웃           PC가 169.254.x.x 자동 주소를 쓸 때: 169.254.42.1:8090 과 신호
  찾으면 기본 브라우저로 화면을 엽니다.

  쓰는 법 (보통은 lwir_connect.bat 을 두 번 누르면 됩니다):
    lwir_connect.ps1                  찾아서 화면을 연다
    lwir_connect.ps1 -NoOpen          찾기만 하고 주소를 출력한다 (브라우저를 열지 않음)
    lwir_connect.ps1 -Server <주소>   찾지 않고 이 주소를 확인해서 연다 (시험용)
    lwir_connect.ps1 -Install         Windows 를 켤 때마다 뒤에서 지켜보다가 보드가 연결되면 화면을 열게 한다 (묻고 나서)
    lwir_connect.ps1 -Uninstall       위 설정을 지우고 지켜보는 프로그램을 끈다
    lwir_connect.ps1 -Status          설정과 지켜보는 프로그램 상태를 본다
    lwir_connect.ps1 -Watch           (-Install 이 쓰는 모드) 계속 지켜본다

  팀 PC GUI: 이 저장소는 그 프로그램의 설치 위치나 실행 명령을 문서로 갖고 있지 않아 경로를 추측하지 않습니다.
  프로그램 경로를 한 줄로 적은 gui.txt (이 파일 옆 또는 %LOCALAPPDATA%\LWIRConnect\gui.txt) 가 있으면 화면을 연 뒤 시작할지 묻습니다.
  (환경 변수 LWIR_GUI 도 같은 뜻입니다.)

  이 파일은 UTF-8 (BOM 있음) 입니다: Windows PowerShell 5.1 이 한글을 바르게 읽도록.
#>
[CmdletBinding()]
param(
    [string]$Server = '',
    [int]$Port = 8090,
    [switch]$NoOpen,
    [switch]$Watch,
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Status,
    [switch]$Yes,
    [int]$BeaconSeconds = 5,
    [string]$StartupDir = '',     # tests only: put the start-up shortcut here instead of the Windows Startup folder
    [string]$InstallRoot = ''     # tests only: copy the program here instead of %LOCALAPPDATA%\LWIRConnect
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$FixedAddress = '10.42.0.1'
$LinkLocalAddress = '169.254.42.1'
$MdnsName = 'orangepi.local'
$BeaconPort = 47890
$PortalStatusPort = 8091
$InstallDir = if ($InstallRoot) { $InstallRoot } else { Join-Path $env:LOCALAPPDATA 'LWIRConnect' }
$ShortcutName = 'LWIR Connect.lnk'

function Say([string]$Text) { Write-Host $Text }

function Test-Tcp([string]$HostName, [int]$TcpPort, [int]$TimeoutMs = 1500) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $pending = $client.BeginConnect($HostName, $TcpPort, $null, $null)
        if (-not $pending.AsyncWaitHandle.WaitOne($TimeoutMs)) { return $false }
        $client.EndConnect($pending)
        return $true
    } catch { return $false } finally { $client.Close() }
}

# HttpWebRequest, not Invoke-WebRequest: IWR hangs in Windows PowerShell 5.1 when its output is redirected / the window is hidden (watch mode).
# Proxy = $null: the board is on the local cable, never ask the PC's proxy.
function Get-Text([string]$Url, [int]$TimeoutSec = 3) {
    try {
        $request = [System.Net.HttpWebRequest]::Create($Url)
        $request.Timeout = $TimeoutSec * 1000
        $request.ReadWriteTimeout = $TimeoutSec * 1000
        $request.Proxy = $null
        $response = $request.GetResponse()
        try {
            if ([int]$response.StatusCode -eq 200) {
                $reader = New-Object System.IO.StreamReader($response.GetResponseStream(), [System.Text.Encoding]::UTF8)
                return $reader.ReadToEnd()
            }
        } finally { $response.Close() }
    } catch { }
    return $null
}

# 이 주소의 8090 이 정말 우리 추적 서버인지 (다른 웹 서버로 착각하지 않게)
function Test-Lwir([string]$HostName, [int]$WebPort = $Port) {
    if (-not (Test-Tcp $HostName $WebPort)) { return $false }
    $info = Get-Text ("http://{0}:{1}/netinfo" -f $HostName, $WebPort)
    if ($info -and $info -match '"client_local"') { return $true }
    $health = Get-Text ("http://{0}:{1}/healthz" -f $HostName, $WebPort)   # netinfo 가 없는 옛 서버
    return [bool]($health -and $health -match '"up"')
}

function Get-LocalIPv4 {
    $found = @()
    foreach ($nic in [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces()) {
        if ($nic.OperationalStatus -ne 'Up' -or $nic.NetworkInterfaceType -eq 'Loopback') { continue }
        foreach ($u in $nic.GetIPProperties().UnicastAddresses) {
            if ($u.Address.AddressFamily -eq 'InterNetwork') { $found += $u.Address.ToString() }
        }
    }
    return $found
}

function Resolve-Mdns([string]$Name, [int]$TimeoutMs = 3000) {
    try {
        $pending = [System.Net.Dns]::BeginGetHostAddresses($Name, $null, $null)
        if (-not $pending.AsyncWaitHandle.WaitOne($TimeoutMs)) { return @() }
        return @([System.Net.Dns]::EndGetHostAddresses($pending) | Where-Object { $_.AddressFamily -eq 'InterNetwork' } | ForEach-Object { $_.ToString() })
    } catch { return @() }
}

# UDP 신호: 듣기 + 질문. 질문은 PC의 모든 IPv4 주소에서 보낸다 (어댑터가 여러 개여도 직결 쪽으로 나가게).
function Find-ByBeacon([int]$Seconds) {
    $listener = $null
    try {
        $listener = New-Object System.Net.Sockets.UdpClient
        $listener.Client.SetSocketOption([System.Net.Sockets.SocketOptionLevel]::Socket, [System.Net.Sockets.SocketOptionName]::ReuseAddress, $true)
        $listener.Client.Bind((New-Object System.Net.IPEndPoint ([System.Net.IPAddress]::Any), $BeaconPort))
        $listener.Client.ReceiveTimeout = 200
    } catch { $listener = $null }   # 포트가 쓰이는 중이거나 방화벽: 아래 질문 방식은 그래도 동작한다
    $askers = @()
    foreach ($ip in (Get-LocalIPv4)) {
        try {
            $u = New-Object System.Net.Sockets.UdpClient (New-Object System.Net.IPEndPoint ([System.Net.IPAddress]::Parse($ip)), 0)
            $u.EnableBroadcast = $true
            $u.Client.ReceiveTimeout = 200
            $askers += $u
        } catch { }
    }
    $question = [System.Text.Encoding]::ASCII.GetBytes('LWIR_DISCOVER')
    $deadline = (Get-Date).AddSeconds($Seconds)
    $result = $null
    try {
        while (-not $result -and (Get-Date) -lt $deadline) {
            foreach ($u in $askers) { try { [void]$u.Send($question, $question.Length, '255.255.255.255', $BeaconPort) } catch { } }
            $sockets = @($askers)
            if ($listener) { $sockets += $listener }
            foreach ($s in $sockets) {
                try {
                    $from = New-Object System.Net.IPEndPoint ([System.Net.IPAddress]::Any), 0
                    $data = $s.Receive([ref]$from)
                    $message = [System.Text.Encoding]::UTF8.GetString($data) | ConvertFrom-Json
                    if ($message.lwir -eq 1 -and $message.addr) { $result = $message; break }
                } catch { }
            }
        }
    } finally {
        foreach ($u in $askers) { $u.Close() }
        if ($listener) { $listener.Close() }
    }
    return $result
}

# 찾는 순서대로. 찾으면 주소 문자열, 못 찾으면 $null.  $Quiet 이면 말하지 않는다.
function Find-Pi([switch]$Quiet) {
    if ($Server) {
        if (-not $Quiet) { Say "지정한 주소 $Server 확인 중..." }
        if (Test-Lwir $Server) { return $Server }
        return $null
    }
    if (-not $Quiet) { Say "1) 직결 고정 주소 $FixedAddress ..." }
    if (Test-Lwir $FixedAddress) { return $FixedAddress }
    if (-not $Quiet) { Say "2) 이름 $MdnsName ..." }
    foreach ($ip in (Resolve-Mdns $MdnsName)) { if (Test-Lwir $ip) { return $ip } }
    if (-not $Quiet) { Say "3) UDP 신호 ($BeaconSeconds 초) ..." }
    $beacon = Find-ByBeacon $BeaconSeconds
    if ($beacon) {
        $beaconPort = if ($beacon.port) { [int]$beacon.port } else { $Port }
        if (Test-Lwir $beacon.addr $beaconPort) { return $beacon.addr }
    }
    if (-not $Quiet) { Say "4) 링크로컬 이웃 $LinkLocalAddress ..." }
    if ((Get-LocalIPv4) -match '^169\.254\.') {
        if (Test-Lwir $LinkLocalAddress) { return $LinkLocalAddress }
    }
    return $null
}

function Open-Page([string]$Address) {
    $url = "http://{0}:{1}/" -f $Address, $Port
    Say "화면: $url"
    if (-not $NoOpen) { Start-Process $url }
    return $url
}

function Offer-Gui {
    if ($NoOpen -or $Watch) { return }
    $path = $env:LWIR_GUI
    foreach ($file in @((Join-Path $PSScriptRoot 'gui.txt'), (Join-Path $InstallDir 'gui.txt'))) {
        if (-not $path -and (Test-Path $file)) { $path = (Get-Content $file -TotalCount 1).Trim() }
    }
    if (-not $path) { return }
    if (-not (Test-Path $path)) { Say "팀 PC GUI 경로를 찾지 못했습니다: $path"; return }
    $answer = if ($Yes) { 'y' } else { Read-Host '팀 PC GUI 도 시작할까요? [y/N]' }
    if ($answer -match '^[yY]') { Start-Process $path }
}

# ---------------------------------------------------------------- 보드가 나타나면 화면을 여는 감시 (Windows 시작 때 뒤에서)
function Invoke-Watch {
    $mutex = New-Object System.Threading.Mutex($false, 'Local\LWIRConnectWatch')
    if (-not $mutex.WaitOne(0)) { return }   # 이미 하나 돌고 있다
    $present = $false
    $lostSince = $null
    $round = 0
    while ($true) {
        $round++
        $hit = $null
        if ($Server) { if (Test-Lwir $Server) { $hit = $Server } }   # -Server: test hook, watch this address only
        elseif (Test-Lwir $FixedAddress) { $hit = $FixedAddress }
        elseif ((Get-LocalIPv4) -match '^169\.254\.' -and (Test-Lwir $LinkLocalAddress)) { $hit = $LinkLocalAddress }
        elseif ($round % 6 -eq 1) { $hit = Find-Pi -Quiet }   # 30초마다 이름과 신호까지
        if ($hit) {
            $lostSince = $null
            if (-not $present) {
                $present = $true
                # 운영체제의 '네트워크 로그인' 창이 이미 화면을 열었으면 (보드가 알려 준다) 또 열지 않는다. 12초 안에 안 열리면 우리가 연다.
                $shown = $false
                for ($i = 0; $i -lt 6 -and -not $shown; $i++) {
                    Start-Sleep -Seconds 2
                    $status = Get-Text ("http://{0}:{1}/lwir-status" -f $hit, $PortalStatusPort) 2
                    if ($status -and $status -match '"shown":\s*true') { $shown = $true }
                    if ($hit -eq $LinkLocalAddress -and -not $status) { break }   # 링크로컬에는 로그인 창이 없다
                }
                if (-not $shown) { [void](Open-Page $hit) }
            }
        } elseif ($present) {
            if (-not $lostSince) { $lostSince = Get-Date }
            elseif (((Get-Date) - $lostSince).TotalSeconds -gt 30) { $present = $false; $lostSince = $null }
        }
        Start-Sleep -Seconds 5
    }
}

function Get-ShortcutPath {
    $folder = if ($StartupDir) { $StartupDir } else { [Environment]::GetFolderPath('Startup') }
    Join-Path $folder $ShortcutName
}

function Get-WatcherProcesses {
    @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'lwir_connect\.ps1' -and $_.CommandLine -match '-Watch' -and $_.ProcessId -ne $PID })
}

function Install-Startup {
    Say 'Windows 를 켤 때마다 뒤에서 LWIR 연결이 보드가 연결되기를 기다리다가, 연결되면 화면을 엽니다.'
    Say "복사 위치: $InstallDir   시작 프로그램 바로가기: $(Get-ShortcutPath)   (언제든 -Uninstall 로 지울 수 있습니다)"
    $answer = if ($Yes) { 'y' } else { Read-Host '설정할까요? [y/N]' }
    if ($answer -notmatch '^[yY]') { Say '설정하지 않았습니다.'; return }
    New-Item -ItemType Directory -Force $InstallDir | Out-Null
    Copy-Item $PSCommandPath (Join-Path $InstallDir 'lwir_connect.ps1') -Force
    $bat = Join-Path $PSScriptRoot 'lwir_connect.bat'
    if (Test-Path $bat) { Copy-Item $bat $InstallDir -Force }
    New-Item -ItemType Directory -Force (Split-Path (Get-ShortcutPath)) | Out-Null
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut((Get-ShortcutPath))
    $link.TargetPath = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $link.Arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}" -Watch' -f (Join-Path $InstallDir 'lwir_connect.ps1')
    $link.WorkingDirectory = $InstallDir
    $link.Description = 'LWIR connect: opens the tracking page when the board is plugged in'
    $link.Save()
    if (-not $NoOpen -and (Get-WatcherProcesses).Count -eq 0) {   # -NoOpen: test run, do not start the watcher
        Start-Process (Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe') -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden', '-File', (Join-Path $InstallDir 'lwir_connect.ps1'), '-Watch')
    }
    Say '설정했습니다. 지금부터 보드가 연결되면 화면이 열립니다.'
}

function Uninstall-Startup {
    $link = Get-ShortcutPath
    if (Test-Path $link) { Remove-Item $link -Force; Say "바로가기를 지웠습니다: $link" }
    foreach ($p in (Get-WatcherProcesses)) { try { Stop-Process -Id $p.ProcessId -Force; Say "지켜보던 프로그램을 껐습니다 (PID $($p.ProcessId))" } catch { } }
    if (Test-Path $InstallDir) { try { Remove-Item $InstallDir -Recurse -Force; Say "복사본을 지웠습니다: $InstallDir" } catch { Say "복사본을 지우지 못했습니다: $InstallDir" } }
    Say '지웠습니다.'
}

function Show-Status {
    Say ('시작 프로그램 바로가기: ' + $(if (Test-Path (Get-ShortcutPath)) { '있음' } else { '없음' }))
    Say ('지켜보는 프로그램: ' + $(if ((Get-WatcherProcesses).Count -gt 0) { '돌고 있음' } else { '꺼져 있음' }))
}

# ---------------------------------------------------------------- 시작
if ($Status) { Show-Status; return }
if ($Uninstall) { Uninstall-Startup; return }
if ($Install) { Install-Startup; return }
if ($Watch) { Invoke-Watch; return }

$address = Find-Pi
if (-not $address) {
    Say ''
    Say '보드를 찾지 못했습니다. 확인하세요:'
    Say '  - 보드의 두 번째 랜 포트(직결용)에 PC를 랜 케이블로 직접 꽂았나요?'
    Say '  - 몇 초 기다린 뒤 다시 실행해 보세요 (PC가 주소를 받는 데 10~30초 걸릴 수 있습니다).'
    exit 2
}
Say "찾았습니다: $address"
[void](Open-Page $address)
Offer-Gui
exit 0
