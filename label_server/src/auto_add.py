# -*- coding: utf-8 -*-
"""30초마다 auto_add_measure.py 를 따로 띄우는 배경 작업.

auto_add_measure.py 는 numpy 와 cv2 가 필요해서 서버 안에서 부르지 않고 별도 프로그램으로 돌린다.
새 시퀀스가 추가되었거나 실패했을 때만 auto_add.log 에 한 줄 남긴다.
프로그램 파일은 이 파일과 같은 src 폴더에 있다. 이미 처리한 측정 목록(auto_added.json)과 기록(auto_add.log)은
전과 같이 서버 폴더(HERE)에 둔다. 그래서 목록 파일의 자리를 --state 로 알려 준다.
"""
import os, sys, threading, time

from .state import HERE

PROGRAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auto_add_measure.py')


def start_auto_add(root):
    """root(라벨 폴더)를 --dst 로 넘겨 auto_add_measure.py 를 되풀이해 돌리는 배경 작업을 시작한다."""
    def auto_add():   # finished live-page measurements appear on the list by themselves (auto_add_measure.py, own process: needs numpy/cv2)
        import subprocess
        while True:
            try:
                r = subprocess.run([sys.executable, PROGRAM, '--dst', root, '--state', os.path.join(HERE, 'auto_added.json')], capture_output=True, text=True, timeout=1800)
                if 'ADDED' in r.stdout or r.returncode:
                    open(os.path.join(HERE, 'auto_add.log'), 'a', encoding='utf-8').write(time.strftime('%Y%m%d_%H%M%S ') + (r.stdout + r.stderr[-600:]).strip() + chr(10))
            except Exception as e:
                open(os.path.join(HERE, 'auto_add.log'), 'a', encoding='utf-8').write(time.strftime('%Y%m%d_%H%M%S ') + 'ERR %s' % e + chr(10))
            time.sleep(30)
    threading.Thread(target=auto_add, daemon=True).start()
