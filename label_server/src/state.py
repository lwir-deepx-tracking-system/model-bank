# -*- coding: utf-8 -*-
"""여러 부품이 함께 쓰는 고정값과, 서버가 켜져 있는 동안의 기억.

HERE 는 start_label_server.py 가 있는 폴더(서버 폴더)다.
접속 키(access_key.txt), 백업(backups/), 휴지통(trash/), 자동 추가 기록(auto_add.log, auto_added.json)이 HERE 에 생긴다.
start_label_server.py 가 켜질 때 자기 위치로 HERE 를 정해 준다. 아래 줄은 이 패키지만 따로 불러 쓸 때의 값이고, 보통은 같은 폴더다.
PRESENCE, SEQS, RESCAN 은 모든 부품이 같은 사전을 나눠 쓴다. 내용만 고치고, 새 사전으로 바꿔 넣지 않는다.
"""
import os, re, threading

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # 이 src 폴더의 바로 위
SAVE_NAMES = ('tracks_edited_mot.txt', 'review_state.json')
READ_NAMES = ('tracks_mot.txt',) + SAVE_NAMES
IMG_RE = re.compile(r'^images/\d{6}\.png$')
MOT_LINE = re.compile(r'^\d+,-?\d+(,-?[\d.]+){5},-?\d+,-?\d+,-?\d+$')
LOCK = threading.Lock()
PRESENCE = {}          # rel -> {user: last_seen}
SEQS = {}              # rel -> {'dir':..., 'frames': n}
RESCAN = {'t': 0.0, 'only': ''}
