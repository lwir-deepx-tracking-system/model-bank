# -*- coding: utf-8 -*-
"""파일을 바꾸는 일.

purge      삭제 표시한 프레임의 영상 파일을 trash/ 로 옮긴다. 남은 프레임 번호는 그대로다.
seq_admin  목록 화면의 버튼: 시퀀스 폴더를 trash/ 로 보내거나 폴더 이름을 바꾼다. 다른 사람이 열어 두었으면 하지 않는다.
save       편집 화면의 MOT, 검수 상태, 신체부위 데이터를 함께 저장한다. 기존 파일은 backups/ 에 남긴다.
한 일은 backups/save_log.txt 에 한 줄씩 남는다. 지우는 것은 없고 옮기기만 한다.
"""
import json, os, re, shutil, sys, tempfile, time
from uuid import uuid4

from .body_parts import validate_parts
from .state import EDITED_MOT, HERE, LOCK, MOT_LINE, PARTS_FILE, PRESENCE, RESCAN, REVIEW_FILE, SAVE_NAMES, SEQS
from .sequences import mot_name, read_json, scan, version
from .transactions import finish_transaction, publish_journal, purge_paths, recover_transaction, transfer_image, write_durable


def frame_numbers(values):
    if not isinstance(values, list):
        raise ValueError('expected a list of frame numbers')
    if any(type(value) not in (int, str) for value in values):
        raise ValueError('expected integer frame numbers')
    return {int(value) for value in values}


def mot_lines(text):
    return [line for line in text.replace('\r', '').split('\n') if line.strip()]


def mot_bytes(lines):
    return ('\n'.join(lines) + ('\n' if lines else '')).encode('utf-8')


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False).encode('utf-8')


def transaction_stamp():
    return f'{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()}'


def commit_files(d, rel, backup_dir, backup_tag, updates, operation=None, images=()):
    """Stage all writes, save prior files, then replace them under the caller's LOCK."""
    images = list(images)
    image_moves = purge_paths(d, rel, operation, images)
    with tempfile.TemporaryDirectory(prefix='.label-save-', dir=d) as staging:
        previous = {}
        os.makedirs(backup_dir, exist_ok=True)
        for name in SAVE_NAMES:
            source = os.path.join(d, name)
            if os.path.exists(source):
                previous[name] = os.path.join(staging, 'previous_' + name)
                shutil.copy2(source, previous[name])
                shutil.copy2(previous[name], os.path.join(backup_dir, backup_tag + '_' + name))
        for name, content in updates.items():
            write_durable(os.path.join(staging, name), content)

        try:
            publish_journal(d, staging, previous, operation, images)
            for source, destination in image_moves:
                transfer_image(source, destination)
            for name in updates:
                os.replace(os.path.join(staging, name), os.path.join(d, name))
            finish_transaction(d)
        except Exception:
            try:
                recover_transaction(d, rel)
            except OSError as error:
                raise RuntimeError(f'rollback failed: {error!r}; backup: {backup_tag}') from error
            raise


class FileOps:
    """handler.H 에 섞어 쓰는 부분. self.send, self.seq_of, self.headers, self.rfile, self.server 는 H 의 것이다."""

    def purge(self, q):
        # move the image files of soft-deleted frames to the server trash folder; frame numbers stay as they are
        user = re.sub(r'[^\w가-힣 .-]', '', q.get('user', [''])[0])[:20] or 'unknown'
        try:
            n = int(self.headers.get('Content-Length', '0') or 0)
            body = json.loads(self.rfile.read(n).decode('utf-8')) if 0 < n < 4096 else {}
            if not isinstance(body, dict):
                raise ValueError('expected a JSON object')
        except (OSError, ValueError):
            return self.send(400, {'error': 'bad body'})
        with LOCK:
            rel = self.seq_of(q)
            if not rel:
                return self.send(404, {'error': 'unknown sequence'})
            d, frames = SEQS[rel]['dir'], SEQS[rel]['frames']
            try:
                cur = version(d, rel)
            except OSError:
                return self.send(500, {'error': '저장된 라벨 파일을 읽지 못했습니다.'})
            if body.get('base') != cur:
                return self.send(409, {'error': 'conflict', 'version': cur})
            try:
                rv = read_json(d, REVIEW_FILE, None)
                if rv is None:
                    raise ValueError('missing review state')
                purged = frame_numbers(rv.get('purged', []))
                gone = {f for f in frame_numbers(rv.get('removed', [])) if 1 <= f <= frames} - purged
            except (OSError, ValueError):
                return self.send(400, {'error': 'no saved review state'})
            if not gone:
                return self.send(200, {'ok': True, 'moved': 0, 'version': cur})

            try:
                with open(os.path.join(d, mot_name(d)), encoding='utf-8') as source:
                    lines = mot_lines(source.read())
                kept = [line for line in lines if int(line.split(',', 1)[0]) not in gone]
                rv['purged'] = rv['removed'] = sorted(purged | gone)
                updates = {EDITED_MOT: mot_bytes(kept), REVIEW_FILE: json_bytes(rv)}
                parts = read_json(d, PARTS_FILE, None)
                if parts is not None:
                    parts = validate_parts(parts, lines, frames)
                    parts['annotations'] = [row for row in parts['annotations'] if row['frame'] not in gone]
                    updates[PARTS_FILE] = json_bytes(parts)
            except (IndexError, OSError, ValueError):
                return self.send(500, {'error': '영상 파일을 옮기기 전에 저장된 라벨을 확인해 주세요.'})

            stamp = transaction_stamp()
            bdir = os.path.join(self.server.backup, rel.replace('/', os.sep))
            operation = stamp + '_' + uuid4().hex
            images = []
            for frame in sorted(gone):
                name = f'{frame:06d}.png'
                source = os.path.join(d, 'images', name)
                if os.path.isfile(source):
                    images.append(name)
            try:
                commit_files(d, rel, bdir, stamp + '_before_purge_' + user, updates, operation, images)
            except (OSError, RuntimeError, ValueError) as error:
                return self.write_failed(error)
            tdir = os.path.join(HERE, 'trash', rel.replace('/', os.sep), operation)
            self.save_log(stamp, rel, user, f'PURGE {len(images)} image files -> {tdir}')
            return self.send(200, {'ok': True, 'moved': len(images), 'version': version(d, rel)})

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
            try:
                recover_transaction(d, rel)
            except OSError:
                return self.send(500, {'error': '이전 저장 작업을 복구하지 못했습니다.'})
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
        user = re.sub(r'[^\w가-힣 .-]', '', q.get('user', [''])[0])[:20] or 'unknown'
        try:
            n = int(self.headers.get('Content-Length', '0') or 0)
        except ValueError:
            return self.send(400, {'error': 'bad size'})
        if n <= 0 or n > 30 * 1024 * 1024:
            return self.send(413, {'error': 'bad size'})
        try:
            body = json.loads(self.rfile.read(n).decode('utf-8'))
            mot, review = body['mot'], body['review']
            if not isinstance(mot, str) or not isinstance(review, dict):
                raise ValueError('expected MOT text and review object')
        except (KeyError, OSError, TypeError, ValueError):
            return self.send(400, {'error': 'bad body'})
        lines = mot_lines(mot)
        with LOCK:
            rel = self.seq_of(q)
            if not rel:
                return self.send(404, {'error': 'unknown sequence'})
            d, frames = SEQS[rel]['dir'], SEQS[rel]['frames']
            try:
                cur = version(d, rel)
            except OSError:
                return self.send(500, {'error': '저장된 라벨 파일을 읽지 못했습니다.'})
            has_parts = 'parts' in body
            base = body.get('base')
            if not has_parts and os.path.exists(os.path.join(d, PARTS_FILE)):
                return self.send(428, {'error': '신체부위 라벨이 있습니다. 이 페이지를 새로고침한 뒤 저장해 주세요.', 'version': cur})
            if has_parts and (not isinstance(base, str) or not base):
                return self.send(428, {'error': '라벨 버전이 필요합니다. 이 페이지를 새로고침해 주세요.', 'version': cur})

            try:
                old = read_json(d, REVIEW_FILE, {})
                purged = frame_numbers(old.get('purged', []))
                old_done = frame_numbers(old.get('reviewed', [])) | frame_numbers(old.get('removed', []))
            except (OSError, ValueError):
                return self.send(500, {'error': '저장된 검수 상태를 확인해 주세요.'})

            if (has_parts and base != cur) or (not has_parts and not body.get('force') and base not in (None, cur)):
                return self.send(409, {'error': 'conflict', 'version': cur, 'by': old.get('saved_by', '')})

            try:
                kept = []
                for line in lines:
                    if not MOT_LINE.fullmatch(line):
                        raise ValueError('형식이 맞지 않는 줄: ' + line[:60])
                    frame = int(line.split(',', 1)[0])
                    if not 1 <= frame <= frames:
                        raise ValueError('영상 범위를 벗어난 줄: ' + line[:60])
                    if frame not in purged:
                        kept.append(line)
                removed = {f for f in frame_numbers(review.get('removed', [])) | purged if 1 <= f <= frames}
                reviewed = {f for f in frame_numbers(review.get('reviewed', [])) if 1 <= f <= frames}
                review.update(removed=sorted(removed), reviewed=sorted(reviewed), saved_by=user)
                review.pop('purged', None)
                if purged:
                    review['purged'] = sorted(purged)
                review.pop('completed_by', None)
                review.pop('completed', None)
                if len(reviewed | removed) >= frames:
                    was_complete = sum(1 for f in old_done if 1 <= f <= frames) >= frames and old.get('completed_by')
                    review['completed_by'] = old['completed_by'] if was_complete else user
                    review['completed'] = old.get('completed', '') if was_complete else time.strftime('%Y-%m-%d %H:%M')
                updates = {EDITED_MOT: mot_bytes(kept), REVIEW_FILE: json_bytes(review)}
                if has_parts:
                    updates[PARTS_FILE] = json_bytes(validate_parts(body['parts'], kept, frames))
            except ValueError as error:
                return self.send(400, {'error': str(error)})

            stamp = transaction_stamp()
            bdir = os.path.join(self.server.backup, rel.replace('/', os.sep))
            try:
                commit_files(d, rel, bdir, stamp + '_before_' + user, updates)
            except (OSError, RuntimeError) as error:
                return self.write_failed(error)
            self.save_log(stamp, rel, user, f'{len(kept)} boxes')
            return self.send(200, {'ok': True, 'version': version(d, rel), 'boxes': len(kept)})

    def write_failed(self, error):
        print(f'label save failed: {error!r}', file=sys.stderr, flush=True)
        if isinstance(error, RuntimeError):
            return self.send(500, {'error': '저장 실패 후 자동 복구를 완료하지 못했습니다. 서버 백업을 확인해 주세요.'})
        return self.send(500, {'error': '파일을 저장하지 못했습니다. 다시 시도해 주세요.'})

    def save_log(self, stamp, rel, user, message):
        try:
            with open(os.path.join(self.server.backup, 'save_log.txt'), 'a', encoding='utf-8') as f:
                f.write(f'{stamp}\t{rel}\t{user}\t{message}\t{self.client_address[0]}\n')
        except OSError as error:
            print(f'label save log failed: {error!r}', file=sys.stderr, flush=True)
