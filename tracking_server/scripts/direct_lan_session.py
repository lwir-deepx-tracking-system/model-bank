#!/usr/bin/env python3
"""Remember one desktop-started direct session and restore its prior web run.

This script runs on the Orange Pi. It never kills a process without verifying
its command, working directory and start time first.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener


STATE = Path.home() / '.modelbank_direct_session.json'
PAUSE = Path.home() / 'lwir_keepalive.pause'


def process(pid):
    try:
        proc = Path('/proc') / str(pid)
        argv = [part.decode('utf-8', 'replace') for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(argv=argv, cwd=str((proc / 'cwd').resolve()),
                    parent=int(fields[1]), started=fields[19])
    except (OSError, ValueError, IndexError):
        return None


def launcher(pid, root):
    info = process(pid)
    return bool(info and info['cwd'] == str(Path(root) / 'tracking_server')
                and any(arg.endswith('run_real_gui.sh') for arg in info['argv']))


def child(pid, mode):
    info = process(pid)
    flag = '--direct-lan-bind' if mode == 'direct' else '--live-port'
    return bool(info and 'start_tracking_server.py' in info['argv'] and flag in info['argv'])


def descendants(parent):
    result = {}
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        info = process(int(item.name))
        if info:
            result[int(item.name)] = info
    owned = set()
    frontier = {parent}
    while frontier:
        level = {pid for pid, info in result.items() if info['parent'] in frontier and pid not in owned}
        owned.update(level)
        frontier = level
    return {pid: result[pid]['started'] for pid in owned}


def same_process(pid, started):
    info = process(pid)
    return bool(info and info['started'] == started)


def stop_owned(pid, root, mode):
    info = process(pid)
    if not launcher(pid, root):
        raise RuntimeError('종료 대상이 지정한 model-bank 실행과 일치하지 않습니다')
    owned = descendants(pid)
    matching = [p for p in owned if child(p, mode)]
    if len(matching) != 1:
        raise RuntimeError('종료 대상의 추적 프로그램을 하나로 식별하지 못했습니다')
    stamp = info['started']
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        alive = {p: s for p, s in owned.items() if same_process(p, s)}
        if not same_process(pid, stamp) and not alive:
            return
        time.sleep(.5)
    if any(same_process(p, owned[p]) for p in matching):
        raise RuntimeError('추적 프로그램이 종료되지 않아 카메라·짐벌 정리를 중단했습니다')
    # Only the helpers recorded as descendants before shutdown may receive TERM.
    for helper, started in owned.items():
        if helper not in matching and same_process(helper, started):
            os.kill(helper, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if not same_process(pid, stamp) and not any(same_process(p, s) for p, s in owned.items()):
            return
        time.sleep(.5)
    raise RuntimeError('추적·카메라·짐벌 프로세스가 종료되지 않아 전환을 중단했습니다')


def web_running():
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        info = process(int(item.name))
        if info and 'start_tracking_server.py' in info['argv'] and '--live-port' in info['argv']:
            return True
    return False


def web_healthy():
    opener = build_opener(ProxyHandler({}))
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        info = process(int(item.name))
        if not info or 'start_tracking_server.py' not in info['argv']:
            continue
        args = info['argv']
        try:
            port = int(args[args.index('--live-port') + 1])
            bind = args[args.index('--live-bind') + 1]
        except (ValueError, IndexError):
            continue
        if bind == '0.0.0.0':
            bind = '127.0.0.1'
        try:
            with opener.open('http://%s:%d/healthz' % (bind, port), timeout=1) as response:
                if response.status == 200:
                    return True
        except (OSError, ValueError):
            continue
    return False


def direct_running():
    for item in Path('/proc').iterdir():
        if item.name.isdigit() and child(int(item.name), 'direct'):
            return True
    return False


def record(root, pid, web_root, marker_created):
    root = str(Path(root).resolve())
    if not launcher(pid, root):
        raise RuntimeError('직결 실행 PID가 지정한 작업 폴더와 일치하지 않습니다')
    if not any(child(int(item.name), 'direct') and process(int(item.name))['parent'] == pid
               for item in Path('/proc').iterdir() if item.name.isdigit()):
        raise RuntimeError('직결 추적 프로그램이 아직 실행되지 않았습니다')
    if web_root and not (Path(web_root) / 'tracking_server/scripts/run_real_gui.sh').is_file():
        raise RuntimeError('복구할 기존 서버 실행 파일이 없습니다')
    data = dict(root=root, pid=pid, started=process(pid)['started'],
                web_root=web_root, marker_created=marker_created)
    temp = STATE.with_name(STATE.name + '.new')
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as output:
        json.dump(data, output)
    os.replace(temp, STATE)


def wait_npu_idle(timeout=60):
    """A stopped tracker may leave DXRT allocated briefly after its PID exits."""
    import dx_engine

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = dx_engine.DeviceStatus.get_current_status(0)
        if status.get_memory_used() < 1024**2 and all(
                status.get_core_utilization(core) < 1 for core in range(3)):
            return
        time.sleep(.5)
    raise RuntimeError('기존 서버 복구 전 NPU가 60초 안에 비워지지 않았습니다')


def restore_web(web_root, marker_created):
    if direct_running():
        raise RuntimeError('직결 추적 프로그램이 아직 실행 중이라 기존 서버를 시작하지 않습니다')
    if web_root and not web_running():
        base = Path(web_root) / 'tracking_server'
        if not (base / 'scripts/run_real_gui.sh').is_file():
            raise RuntimeError('기존 서버 실행 파일이 없어 복구하지 못했습니다')
        wait_npu_idle()
        with open(base / 'direct_lan_restore.out', 'ab', buffering=0) as output:
            subprocess.Popen(['bash', 'scripts/run_real_gui.sh'], cwd=base,
                             stdin=subprocess.DEVNULL, stdout=output,
                             stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    if web_root:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not web_healthy():
            time.sleep(1)
        if not web_healthy():
            raise RuntimeError('기존 서버가 90초 안에 /healthz 응답을 주지 않았습니다')
    if marker_created and PAUSE.exists():
        PAUSE.unlink()


def finish():
    if not STATE.exists():
        print('no desktop-owned direct session')
        return
    data = json.loads(STATE.read_text())
    pid = int(data['pid'])
    if same_process(pid, data['started']):
        stop_owned(pid, data['root'], 'direct')
    restore_web(data['web_root'], data['marker_created'])
    STATE.unlink()
    print('previous model-bank server restored' if data['web_root'] else 'direct session stopped')


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='action', required=True)
    save = subs.add_parser('record')
    save.add_argument('--root', required=True)
    save.add_argument('--pid', type=int, required=True)
    save.add_argument('--web-root', default='')
    save.add_argument('--marker-created', action='store_true')
    subs.add_parser('finish')
    quiesce = subs.add_parser('quiesce')
    quiesce.add_argument('--pid', type=int, required=True)
    quiesce.add_argument('--root', required=True)
    restore = subs.add_parser('restore')
    restore.add_argument('--web-root', required=True)
    restore.add_argument('--marker-created', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'record':
            record(args.root, args.pid, args.web_root, args.marker_created)
        elif args.action == 'quiesce':
            stop_owned(args.pid, args.root, 'web')
        elif args.action == 'restore':
            restore_web(args.web_root, args.marker_created)
        else:
            finish()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
