# -*- coding: utf-8 -*-
"""파일을 바꾸는 일.

purge      삭제 표시한 프레임의 영상 파일을 trash/ 로 옮긴다. 남은 프레임 번호는 그대로다.
seq_admin  목록 화면의 버튼: 시퀀스 폴더를 trash/ 로 보내거나 폴더 이름을 바꾼다. 다른 사람이 열어 두었으면 하지 않는다.
save       편집 화면의 저장: tracks_edited_mot.txt, review_state.json 을 쓴다. 덮어쓰기 전 파일은 backups/ 에 남긴다.
한 일은 backups/save_log.txt 에 한 줄씩 남는다. 지우는 것은 없고 옮기기만 한다.
"""
import json, os, re, shutil, time

from .state import HERE, LOCK, MOT_LINE, PRESENCE, RESCAN, SAVE_NAMES, SEQS
from .sequences import scan, version


class FileOps:
    """handler.H 에 섞어 쓰는 부분. self.send, self.seq_of, self.headers, self.rfile, self.server 는 H 의 것이다."""

    def purge(self, q):
        # move the image files of soft-deleted frames to the server trash folder; frame numbers stay as they are
        rel = self.seq_of(q)
        if not rel:
            return self.send(404, {'error': 'unknown sequence'})
        user = re.sub(r'[^\w가-힣 .-]', '', q.get('user', [''])[0])[:20] or 'unknown'
        n = int(self.headers.get('Content-Length', '0') or 0)
        try:
            body = json.loads(self.rfile.read(n).decode('utf-8')) if 0 < n < 4096 else {}
        except Exception:
            return self.send(400, {'error': 'bad body'})
        d = SEQS[rel]['dir']
        frames = SEQS[rel]['frames']
        with LOCK:
            cur = version(d)
            if body.get('base') != cur:
                return self.send(409, {'error': 'conflict', 'version': cur})
            rp = os.path.join(d, 'review_state.json')
            try:
                rv = json.load(open(rp, encoding='utf-8'))
                todo = sorted({int(x) for x in rv.get('removed', []) if 1 <= int(x) <= frames})
            except Exception:
                return self.send(400, {'error': 'no saved review state'})
            stamp = time.strftime('%Y%m%d_%H%M%S')
            bdir = os.path.join(self.server.backup, rel.replace('/', os.sep))
            tdir = os.path.join(HERE, 'trash', rel.replace('/', os.sep), stamp + '_' + user)
            os.makedirs(bdir, exist_ok=True)
            for nm in SAVE_NAMES:
                fp = os.path.join(d, nm)
                if os.path.exists(fp):
                    shutil.copy2(fp, os.path.join(bdir, '%s_before_purge_%s_%s' % (stamp, user, nm)))
            moved = []
            for f in todo:
                fp = os.path.join(d, 'images', '%06d.png' % f)
                if os.path.isfile(fp):
                    os.makedirs(tdir, exist_ok=True)
                    shutil.move(fp, os.path.join(tdir, '%06d.png' % f))
                    moved.append(f)
            gone = set(todo)
            rv['purged'] = sorted(set(rv.get('purged', [])) | gone)
            rv['removed'] = sorted(gone | set(rv['purged']))
            ep = os.path.join(d, SAVE_NAMES[0])
            if os.path.exists(ep):
                keep = [l for l in open(ep, encoding='utf-8').read().replace('\r', '').split('\n') if l.strip() and int(l.split(',', 1)[0]) not in gone]
                with open(ep + '.tmp_srv', 'w', encoding='utf-8', newline='\n') as f:
                    f.write('\n'.join(keep) + ('\n' if keep else ''))
                os.replace(ep + '.tmp_srv', ep)
            with open(rp + '.tmp_srv', 'w', encoding='utf-8', newline='\n') as f:
                f.write(json.dumps(rv, ensure_ascii=False, indent=1))
            os.replace(rp + '.tmp_srv', rp)
            with open(os.path.join(self.server.backup, 'save_log.txt'), 'a', encoding='utf-8') as f:
                f.write('%s\t%s\t%s\tPURGE %d image files -> %s\t%s\n' % (stamp, rel, user, len(moved), tdir, self.client_address[0]))
            return self.send(200, {'ok': True, 'moved': len(moved), 'version': version(d)})

    def seq_admin(self, path, q):
        # list-page buttons: move a whole sequence folder to the server trash, or rename its folder. Nothing is erased.
        n = int(self.headers.get('Content-Length', '0') or 0)
        if n > 4096:
            return self.send(413, {'error': 'bad size'})
        try:
            body = json.loads(self.rfile.read(n).decode('utf-8')) if n else {}
        except Exception:
            body = {}
        user = re.sub(r'[^\w가-힣 .-]', '', q.get('user', [''])[0])[:20].strip()
        if not user:
            return self.send(400, {'error': '이름을 먼저 적어 주세요.'})
        rel = q.get('seq', [''])[0]
        stamp = time.strftime('%Y%m%d_%H%M%S')
        root = self.server.root
        with LOCK:
            if rel not in SEQS:
                return self.send(404, {'error': '없는 시퀀스입니다. 목록을 새로 고쳐 주세요.'})
            d = os.path.abspath(SEQS[rel]['dir'])
            if d == root or not d.startswith(root + os.sep):
                return self.send(400, {'error': 'bad path'})
            now = time.time()
            others = sorted(x for x, t in PRESENCE.get(rel, {}).items() if now - t < 90 and x != user)
            if others:
                return self.send(409, {'error': '지금 %s 님이 열어 두었습니다. 닫은 뒤에 다시 해 주세요.' % ', '.join(others)})
            if path == '/api/seq_trash':
                dst = os.path.join(HERE, 'trash', '_sequences', stamp + '__' + rel.replace('/', '__'))
                what = 'TRASH sequence -> %s' % dst
            else:
                name = re.sub(r'[^\w가-힣 .-]', '', str(body.get('name', ''))).strip(' .')[:60]
                if not name:
                    return self.send(400, {'error': '새 이름이 비었습니다. 한글, 영문, 숫자, 빈칸, - _ . 만 쓸 수 있습니다.'})
                dst = os.path.join(os.path.dirname(d), name)
                what = 'RENAME sequence -> %s' % os.path.relpath(dst, root).replace(os.sep, '/')
            if os.path.exists(dst):
                return self.send(409, {'error': '같은 이름이 이미 있습니다.'})
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.rename(d, dst)
            except Exception as e:
                return self.send(500, {'error': '옮기지 못했습니다: %s' % e})
            PRESENCE.pop(rel, None)
            try:
                new = {r: v for r, v in scan(root).items() if r.startswith(RESCAN['only'])}
                SEQS.clear(); SEQS.update(new); RESCAN['t'] = time.time()
            except Exception:
                SEQS.pop(rel, None)
            with open(os.path.join(self.server.backup, 'save_log.txt'), 'a', encoding='utf-8') as f:
                f.write('%s\t%s\t%s\t%s\t%s\n' % (stamp, rel, user, what, self.client_address[0]))
        return self.send(200, {'ok': True})

    def save(self, q):
        # 편집 화면의 저장 (/api/save). 주소와 접속 키 확인은 handler.H.do_POST 가 먼저 한다.
        rel = self.seq_of(q)
        if not rel:
            return self.send(404, {'error': 'unknown sequence'})
        user = re.sub(r'[^\w가-힣 .-]', '', q.get('user', [''])[0])[:20] or 'unknown'
        n = int(self.headers.get('Content-Length', '0') or 0)
        if n <= 0 or n > 30 * 1024 * 1024:
            return self.send(413, {'error': 'bad size'})
        try:
            body = json.loads(self.rfile.read(n).decode('utf-8'))
            mot = body['mot']
            review = body['review']
            assert isinstance(mot, str) and isinstance(review, dict)
        except Exception:
            return self.send(400, {'error': 'bad body'})
        lines = [l for l in mot.replace('\r', '').split('\n') if l.strip()]
        frames = SEQS[rel]['frames']
        for l in lines:
            if not MOT_LINE.match(l) or not (1 <= int(l.split(',', 1)[0]) <= frames):
                return self.send(400, {'error': '형식이 맞지 않는 줄: ' + l[:60]})
        d = SEQS[rel]['dir']
        with LOCK:
            cur = version(d)
            if not body.get('force') and body.get('base') not in (None, cur):
                by = ''
                try:
                    by = json.load(open(os.path.join(d, 'review_state.json'), encoding='utf-8')).get('saved_by', '')
                except Exception:
                    pass
                return self.send(409, {'error': 'conflict', 'version': cur, 'by': by})
            stamp = time.strftime('%Y%m%d_%H%M%S')
            bdir = os.path.join(self.server.backup, rel.replace('/', os.sep))
            os.makedirs(bdir, exist_ok=True)
            for nm in SAVE_NAMES:
                fp = os.path.join(d, nm)
                if os.path.exists(fp):
                    shutil.copy2(fp, os.path.join(bdir, '%s_before_%s_%s' % (stamp, user, nm)))
            review['saved_by'] = user
            old0 = {}
            try:
                old0 = json.load(open(os.path.join(d, 'review_state.json'), encoding='utf-8'))
            except Exception:
                pass
            pg = sorted({int(x) for x in old0.get('purged', [])})      # frames whose image file is gone stay removed
            review.pop('purged', None)
            if pg:
                review['purged'] = pg
                review['removed'] = list(review.get('removed', [])) + pg
                lines = [l for l in lines if int(l.split(',', 1)[0]) not in set(pg)]
            review.pop('completed_by', None); review.pop('completed', None)
            try:
                review['removed'] = sorted({int(x) for x in review.get('removed', []) if 1 <= int(x) <= frames})
            except Exception:
                return self.send(400, {'error': 'bad removed list'})
            if len(set(review.get('reviewed', [])) | set(review['removed'])) >= frames:      # whole sequence reviewed: keep who finished it
                old = {}
                try:
                    old = json.load(open(os.path.join(d, 'review_state.json'), encoding='utf-8'))
                except Exception:
                    pass
                was = len(set(old.get('reviewed', [])) | set(old.get('removed', []))) >= frames and old.get('completed_by')
                review['completed_by'] = old['completed_by'] if was else user
                review['completed'] = old.get('completed', '') if was else time.strftime('%Y-%m-%d %H:%M')
            for nm, txt in ((SAVE_NAMES[0], '\n'.join(lines) + ('\n' if lines else '')),
                            (SAVE_NAMES[1], json.dumps(review, ensure_ascii=False, indent=1))):
                tmp = os.path.join(d, nm + '.tmp_srv')
                with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(txt)
                os.replace(tmp, os.path.join(d, nm))
            with open(os.path.join(self.server.backup, 'save_log.txt'), 'a', encoding='utf-8') as f:
                f.write('%s\t%s\t%s\t%d boxes\t%s\n' % (stamp, rel, user, len(lines), self.client_address[0]))
            return self.send(200, {'ok': True, 'version': version(d), 'boxes': len(lines)})
