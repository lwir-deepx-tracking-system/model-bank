# -*- coding: utf-8 -*-
"""LWIR web labeling server (stdlib only).

Serves the existing label_editor.html over http so a teammate only opens an address.
Saves go into the same OneDrive sequence folder (tracks_edited_mot.txt, review_state.json),
so OneDrive keeps syncing them. Every overwritten file is backed up first.

  python start_label_server.py --root <labels folder> --bind <address> --port 8770
"""
import argparse, os, secrets, time

HERE = os.path.dirname(os.path.abspath(__file__))
from src import state
state.HERE = HERE      # 접속 키, 백업, 휴지통, 자동 추가 기록이 생기는 폴더. 아래 부품들을 불러오기 전에 정해 준다
from src.state import RESCAN, SEQS
from src.sequences import scan
from src.editor_page import build_editor
from src.handler import H, Srv
from src.auto_add import start_auto_add


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('LWIR_LABEL_ROOT', 'LWIR_labels'))
    ap.add_argument('--bind', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--only', default='', help='only sequences whose path starts with this (e.g. opi_sequences)')
    ap.add_argument('--no-token', action='store_true')
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    RESCAN['only'], RESCAN['t'] = a.only, time.time()
    for rel, s in scan(root).items():
        if rel.startswith(a.only):
            SEQS[rel] = s
    build_editor(root)
    tf = os.path.join(HERE, 'access_key.txt')
    token = ''
    if not a.no_token:
        if not os.path.exists(tf):
            open(tf, 'w').write(secrets.token_urlsafe(9))
        token = open(tf).read().strip()
    srv = Srv((a.bind, a.port), H)
    srv.root, srv.token = root, token
    srv.backup = os.path.join(HERE, 'backups')
    os.makedirs(srv.backup, exist_ok=True)
    print('sequences: %d  frames: %d' % (len(SEQS), sum(s['frames'] for s in SEQS.values())), flush=True)
    print('address: http://%s:%d/%s' % (a.bind, a.port, ('?k=' + token) if token else ''), flush=True)
    start_auto_add(root)
    srv.serve_forever()


if __name__ == '__main__':
    main()
