"""Start the model-bank direct mode on the cabled Orange Pi over SSH.

No password or device-specific repository path is kept in this module.
"""
from __future__ import annotations

import json
import shlex
import time
from pathlib import Path

from link import Link, LinkError


class StartupError(RuntimeError):
    pass


class PasswordRequired(StartupError):
    pass


class SwitchRequired(StartupError):
    """The existing model-bank web run needs the user's consent to stop."""


def ready(board, port, timeout=0.8):
    try:
        link = Link(board, port, timeout=timeout)
        header, body = link.request({'op': 'route', 'method': 'GET', 'path': '/time'}, timeout=timeout)
        result = json.loads(body)
        return header.get('status') == 200 and isinstance(result, dict) and 't' in result
    except (OSError, ValueError, KeyError, RuntimeError, LinkError):
        return False


def connect(board, user, password=None):
    try:
        import paramiko
    except ImportError as exc:
        raise StartupError('PC에 python3-paramiko 패키지가 필요합니다') from exc
    client = paramiko.SSHClient()
    hosts = Path.home() / '.ssh' / 'known_hosts'
    hosts.parent.mkdir(mode=0o700, exist_ok=True)
    hosts.touch(mode=0o600, exist_ok=True)
    client.load_host_keys(str(hosts))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(board, username=user, password=password, port=22,
                       timeout=4, auth_timeout=6, banner_timeout=6,
                       allow_agent=password is None, look_for_keys=password is None)
    except paramiko.AuthenticationException as exc:
        client.close()
        if password is None:
            raise PasswordRequired('Orange Pi SSH 비밀번호가 필요합니다') from exc
        raise StartupError('Orange Pi SSH 인증에 실패했습니다') from exc
    except (paramiko.SSHException, OSError) as exc:
        client.close()
        raise StartupError('유선 SSH 연결 실패: ' + str(exc)) from exc
    return client


def remote_command(client, command, timeout=12):
    try:
        _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        stdout.channel.settimeout(timeout)
        code = stdout.channel.recv_exit_status()
        return code, stdout.read().decode('utf-8', 'replace').strip(), stderr.read().decode('utf-8', 'replace').strip()
    except (OSError, EOFError) as exc:
        raise StartupError('Orange Pi 명령 연결이 끊겼습니다: ' + str(exc)) from exc


def find_root(client, preferred=''):
    if preferred:
        root = preferred.rstrip('/')
        path = shlex.quote(root + '/tracking_server/scripts/run_direct_lan.sh')
        code, _, _ = remote_command(client, 'test -f ' + path)
        if code:
            raise StartupError('지정한 Orange Pi 폴더에 직결 실행 스크립트가 없습니다: ' + root)
        return root
    command = ('find "$HOME" -maxdepth 7 -type f '
               "-path '*/tracking_server/scripts/run_direct_lan.sh' -print 2>/dev/null")
    code, out, _ = remote_command(client, command, timeout=20)
    paths = sorted(set(line for line in out.splitlines() if line.endswith('/tracking_server/scripts/run_direct_lan.sh')))
    if not paths:
        raise StartupError('Orange Pi에서 직결 실행 스크립트를 찾지 못했습니다. 이 브랜치를 장치의 model-bank에 먼저 설치하세요.')
    if len(paths) > 1:
        raise StartupError('Orange Pi에 model-bank 복사본이 여러 개 있습니다. 설정에서 remote_root를 지정하세요:\n' + '\n'.join(paths[:6]))
    return paths[0][:-len('/tracking_server/scripts/run_direct_lan.sh')]


def web_process(client):
    """Identify only a model-bank web launcher and its live Python child."""
    inspect = '''from pathlib import Path
found=[]
for item in Path('/proc').iterdir():
    if not item.name.isdigit(): continue
    try:
        argv=(item/'cmdline').read_bytes().split(b'\\0')
        if b'start_tracking_server.py' not in argv or b'--live-port' not in argv or b'--direct-lan-bind' in argv: continue
        parent=int(next(s.split(':',1)[1].strip() for s in (item/'status').read_text().splitlines() if s.startswith('PPid:')))
        par=Path('/proc')/str(parent)
        pargv=(par/'cmdline').read_bytes().split(b'\\0')
        cwd=par/'cwd'
        if not any(x.endswith(b'run_real_gui.sh') for x in pargv): continue
        if not (cwd/'scripts/run_real_gui.sh').is_file(): continue
        found.append((int(item.name),parent))
    except (OSError, StopIteration, ValueError): continue
print(*('%s %s'%pair for pair in found), sep='\\n')'''
    code, out, err = remote_command(client, 'python3 -c ' + shlex.quote(inspect))
    if code:
        raise StartupError('기존 model-bank 실행을 확인하지 못했습니다: ' + err)
    rows = [line.split() for line in out.splitlines() if line.strip()]
    if len(rows) > 1:
        raise StartupError('Orange Pi에 기존 웹 실행이 여러 개 있어 자동 전환할 수 없습니다')
    if not rows:
        return None
    if len(rows[0]) != 2 or not all(part.isdigit() for part in rows[0]):
        raise StartupError('기존 model-bank 실행 정보가 올바르지 않습니다')
    return tuple(map(int, rows[0]))


def pause_keepalive(client):
    code, marker, error = remote_command(client,
        'if test -e "$HOME/lwir_keepalive.pause"; then echo existing; '
        'else touch "$HOME/lwir_keepalive.pause" && echo created; fi')
    if code:
        raise StartupError('장치의 keepalive 일시정지를 설정하지 못했습니다: ' + error)
    return marker == 'created'


def stop_web(client, child, parent):
    if web_process(client) != (child, parent):
        raise StartupError('기존 model-bank 실행이 바뀌어 자동 전환을 중단했습니다')
    code, _, error = remote_command(client, 'kill -TERM ' + str(parent))
    if code:
        raise StartupError('기존 model-bank 종료 요청 실패: ' + error)
    for _ in range(30):
        code, _, _ = remote_command(client, 'kill -0 ' + str(child) + ' 2>/dev/null')
        if code:
            return
        time.sleep(1)
    raise StartupError('기존 model-bank가 30초 안에 종료되지 않았습니다. 장치 상태를 확인하세요')


def launch(client, root):
    root_q = shlex.quote(root)
    script_q = shlex.quote(root + '/tracking_server/scripts/run_direct_lan.sh')
    log_q = shlex.quote(root + '/tracking_server/direct_lan_launch.out')
    # A paused keepalive cannot race us by relaunching the web mode. Restore
    # the marker on startup failure only when this launch created it.
    created = pause_keepalive(client)
    detached = (
        'import subprocess,sys; '
        'root,script,log=sys.argv[1:]; '
        'output=open(log,"ab",buffering=0); '
        'child=subprocess.Popen(["bash",script],cwd=root,stdin=subprocess.DEVNULL,'
        'stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True); '
        'print(child.pid)'
    )
    command = f'python3 -c {shlex.quote(detached)} {root_q} {script_q} {log_q}'
    code, pid, error = remote_command(client, command)
    if code or not pid.isdigit():
        if created:
            remote_command(client, 'rm "$HOME/lwir_keepalive.pause"')
        raise StartupError('직결 실행 요청 실패: ' + (error or pid))
    return int(pid), created


def launch_failure(client, root, pid):
    code, _, _ = remote_command(client, 'kill -0 ' + str(pid) + ' 2>/dev/null')
    if code == 0:
        return ''
    log_q = shlex.quote(root + '/tracking_server/direct_lan_launch.out')
    _, out, _ = remote_command(client, 'tail -n 12 ' + log_q + ' 2>/dev/null')
    return out or 'Orange Pi 직결 실행이 바로 종료되었습니다'


def restore_marker(client):
    remote_command(client, 'rm "$HOME/lwir_keepalive.pause"')


def start(board, port, user, password=None, remote_root='', progress=lambda _text: None,
          allow_web_switch=False):
    if ready(board, port):
        progress('직결 모드가 이미 실행 중입니다')
        return False
    progress('Orange Pi 유선 SSH에 연결 중…')
    client = connect(board, user, password)
    created = False
    try:
        if ready(board, port):
            progress('직결 모드가 이미 실행 중입니다')
            return False
        root = find_root(client, remote_root)
        old = web_process(client)
        if old:
            if not allow_web_switch:
                raise SwitchRequired('기존 model-bank 8090 실행을 종료하고 직결 모드로 전환할까요?')
            progress('기존 model-bank 웹 실행을 정상 종료 중…')
            created = pause_keepalive(client)
            stop_web(client, *old)
        progress('Orange Pi에서 model-bank 직결 모드를 시작 중…')
        try:
            pid, launch_created = launch(client, root)
            created = created or launch_created
        except StartupError:
            if created:
                restore_marker(client)
            raise
        for index in range(120):
            if ready(board, port):
                progress('Orange Pi 직결 모드 준비 완료')
                return True
            if index == 1 or (index and index % 5 == 0):
                progress('Orange Pi 모델과 카메라 준비 중… %d초' % index)
                failure = launch_failure(client, root, pid)
                if failure:
                    if created:
                        restore_marker(client)
                    raise StartupError('Orange Pi 실행 실패:\n' + failure)
            time.sleep(1)
        raise StartupError('Orange Pi는 실행 중이지만 직결 통신 준비가 120초 안에 끝나지 않았습니다. 장치 로그를 확인하세요. keepalive 일시정지는 유지했습니다.')
    finally:
        client.close()
