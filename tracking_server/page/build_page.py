#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""조각 파일을 이어 붙여 웹 화면 파일(live_page.html)을 만든다.

서버는 이 폴더(page/)의 live_page.html 한 파일만 읽는다. 그 파일은 손으로 고치지 않는다.
이 폴더의 조각(html/, css/, js/)을 고친 뒤 이 도구로 다시 만든다.

쓰는 법 (이 폴더에서):
    python3 build_page.py            조각을 이어 붙여 이 폴더의 live_page.html 을 다시 만든다
    python3 build_page.py --check    만들지 않고, 지금 있는 live_page.html 이 조각을 이어 붙인 것과
                                     바이트까지 같은지만 본다 (같으면 0, 다르면 1 로 끝남)
    python3 build_page.py --list     조각 목록과 크기만 보여 준다

이어 붙이는 순서는 parts.json 에 적힌 순서다. 조각 사이에 아무것도 넣지 않는다.
줄바꿈, 빈칸, 머리말을 더하거나 빼지 않고 파일에 있는 바이트를 그대로 잇는다.

끝나는 값: 0 = 됨 (또는 같음), 1 = --check 에서 다름, 2 = 조각이나 목록이 잘못됨.
표준 라이브러리만 쓴다. Python 3.10 에서 돌아간다.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LIST = HERE / "parts.json"
DEFAULT_OUTPUT = "live_page.html"


class PartsError(Exception):
    """조각이나 목록이 잘못됐을 때."""


def md5_hex(data):
    try:
        return hashlib.md5(data, usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(data).hexdigest()


def load_list(list_path):
    """parts.json 을 읽어 (만들 파일 경로, [조각 이름, ...]) 을 돌려준다."""
    try:
        spec = json.loads(list_path.read_bytes().decode("utf-8-sig"))
    except OSError as e:
        raise PartsError("조각 목록(%s)을 읽지 못했습니다 (%s)" % (list_path.name, e.strerror or e))
    except ValueError as e:
        raise PartsError("조각 목록(%s)이 올바른 JSON 이 아닙니다: %s" % (list_path.name, e))
    if not isinstance(spec, dict) or not isinstance(spec.get("parts"), list) or not spec["parts"]:
        raise PartsError('조각 목록에 "parts" 가 없거나 비어 있습니다')
    names = []
    for item in spec["parts"]:
        name = item.get("file") if isinstance(item, dict) else item
        if not isinstance(name, str) or not name:
            raise PartsError('조각 목록의 항목에 "file" 이 없습니다: %r' % (item,))
        p = Path(name)
        if p.is_absolute() or name.startswith("/") or ".." in p.parts or "\\" in name or ":" in name:
            raise PartsError("조각 이름은 이 폴더 안의 상대 경로(/ 로 구분)여야 합니다: %s" % name)
        if name in names:
            raise PartsError("같은 조각이 목록에 두 번 있습니다: %s" % name)
        names.append(name)
    out = spec.get("output", DEFAULT_OUTPUT)
    if not isinstance(out, str) or not out:
        raise PartsError('조각 목록의 "output" 이 잘못됐습니다')
    return (list_path.parent / out), names


def read_parts(base, names):
    """조각을 바이트 그대로 읽는다. 없거나 빈 조각이 있으면 멈춘다."""
    parts = []
    for name in names:
        f = base / name
        try:
            data = f.read_bytes()
        except OSError as e:
            raise PartsError("조각을 읽지 못했습니다: %s (%s)" % (name, e.strerror or e))
        if not data:
            raise PartsError("조각이 비어 있습니다: %s" % name)
        parts.append((name, data))
    return parts


def warnings_for(parts):
    """화면이 깨질 수 있는 흔한 실수를 알려 준다 (만들기는 그대로 한다)."""
    notes = []
    for name, data in parts:
        if data.startswith(b"\xef\xbb\xbf"):
            notes.append("%s: 파일 맨 앞에 BOM 이 있습니다. 편집기에서 'BOM 없는 UTF-8' 로 저장하세요." % name)
        if b"\r" in data:
            notes.append("%s: 줄 끝에 CR(윈도우식 줄바꿈)이 %d개 있습니다. 줄바꿈을 LF 로 저장하세요." % (name, data.count(b"\r")))
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            notes.append("%s: UTF-8 이 아닌 글자가 있습니다. UTF-8 로 저장하세요." % name)
    return notes


def first_difference(parts, built, existing):
    """처음 다른 자리를 사람이 찾기 쉬운 말로 돌려준다."""
    n = min(len(built), len(existing))
    pos = next((i for i in range(n) if built[i] != existing[i]), n)
    line_out = existing.count(b"\n", 0, pos) + 1
    start = 0
    where = "조각 끝을 지난 자리"
    for name, data in parts:
        if pos < start + len(data):
            where = "조각 %s 의 %d번째 줄" % (name, data.count(b"\n", 0, pos - start) + 1)
            break
        start += len(data)
    return "live_page.html %d번째 줄 · %s (앞에서 %d 바이트째)" % (line_out, where, pos)


def write_atomic(out, data):
    """옆에 임시 파일을 쓰고 한 번에 바꿔치기한다 (서버가 반만 쓰인 파일을 읽지 않게)."""
    tmp = out.with_name(out.name + ".tmp-build")
    try:
        tmp.write_bytes(data)
        if out.exists():
            try:
                shutil.copymode(str(out), str(tmp))
            except OSError:
                pass
        os.replace(str(tmp), str(out))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="조각 파일을 이어 붙여 live_page.html 을 만든다")
    ap.add_argument("--check", action="store_true", help="만들지 않고, 있는 파일이 조각을 이은 것과 같은지만 본다")
    ap.add_argument("--list", action="store_true", help="조각 목록과 크기만 보여 준다")
    ap.add_argument("--parts", default=str(DEFAULT_LIST), help="조각 목록 파일 (기본: 이 폴더의 parts.json)")
    ap.add_argument("--out", default=None, help="만들 파일 (기본: 조각 목록의 output = 이 폴더의 live_page.html)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:
            pass

    list_path = Path(args.parts).resolve()
    try:
        out, names = load_list(list_path)
        if args.out:
            out = Path(args.out).resolve()
        parts = read_parts(list_path.parent, names)
    except PartsError as e:
        print("멈춤: %s" % e, file=sys.stderr)
        return 2
    built = b"".join(data for _, data in parts)
    for note in warnings_for(parts):
        print("주의: %s" % note, file=sys.stderr)

    if args.list:
        for name, data in parts:
            print("%7d 바이트  %s" % (len(data), name))
        print("%7d 바이트  합계 (조각 %d개, md5 %s)" % (len(built), len(parts), md5_hex(built)))
        return 0

    try:
        existing = out.read_bytes() if out.is_file() else None
    except OSError as e:
        print("멈춤: %s 을 읽지 못했습니다 (%s)" % (out.name, e.strerror or e), file=sys.stderr)
        return 2

    if args.check:
        if existing is None:
            print("다릅니다: %s 이 없습니다. python3 build_page.py 로 만드세요." % out.name)
            return 1
        if existing == built:
            print("같습니다: %s 은 조각 %d개를 이어 붙인 것과 바이트까지 같습니다 (%d 바이트, md5 %s)" % (out.name, len(parts), len(built), md5_hex(built)))
            return 0
        print("다릅니다: %s (%d 바이트) 이 조각을 이어 붙인 것 (%d 바이트) 과 다릅니다." % (out.name, len(existing), len(built)))
        print("  처음 다른 곳: %s" % first_difference(parts, built, existing))
        print("  조각을 고친 뒤라면: python3 build_page.py 를 돌려 다시 만드세요.")
        print("  %s 을 직접 고쳤다면: 고친 내용을 조각 파일로 옮긴 뒤 다시 만드세요 (지금 다시 만들면 직접 고친 내용이 사라집니다)." % out.name)
        return 1

    if existing == built:
        print("바뀐 것 없음: %s 은 이미 조각 %d개를 이어 붙인 것과 같습니다 (%d 바이트, md5 %s)" % (out.name, len(parts), len(built), md5_hex(built)))
        return 0
    try:
        write_atomic(out, built)
    except OSError as e:
        print("멈춤: %s 을 쓰지 못했습니다 (%s)" % (out.name, e.strerror or e), file=sys.stderr)
        return 2
    if existing is None:
        print("만들었습니다: %s (조각 %d개, %d 바이트, md5 %s)" % (out.name, len(parts), len(built), md5_hex(built)))
    else:
        print("다시 만들었습니다: %s (조각 %d개, %d → %d 바이트, md5 %s)" % (out.name, len(parts), len(existing), len(built), md5_hex(built)))
        print("  이전 파일과 처음 다른 곳: %s" % first_difference(parts, built, existing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
