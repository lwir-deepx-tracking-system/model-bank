"""Recover an interrupted annotation save before a server reader can see it."""

import base64
import binascii
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile

from . import state


JOURNAL = '.label_transaction.json'
PURGE_NAME = re.compile(r'\d{8}_\d{6}_\d{19,}_[0-9a-f]{32}', re.ASCII)
FRAME_NAME = re.compile(r'\d{6}\.png', re.ASCII)


class TransactionError(OSError):
    """A journal is invalid or recovery could not be completed."""


def sync_directory(directory):
    # Windows cannot open a directory using os.open. File data is still flushed there.
    if os.name != 'nt':
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_durable(path, content):
    with open(path, 'wb') as target:
        target.write(content)
        target.flush()
        os.fsync(target.fileno())


def _purge_paths(directory, rel, operation, names):
    if not isinstance(operation, str) or not PURGE_NAME.fullmatch(operation):
        raise ValueError('invalid purge operation name')
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not FRAME_NAME.fullmatch(name) for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('invalid purge frame names')
    rel_path = PurePosixPath(rel)
    if rel_path.is_absolute() or '..' in rel_path.parts or '\\' in rel or ':' in rel:
        raise ValueError('invalid sequence path')
    image_dir = Path(directory).resolve() / 'images'
    trash_dir = (Path(state.HERE) / 'trash').resolve().joinpath(*rel_path.parts, operation)
    if image_dir.resolve() != image_dir or trash_dir.resolve() != trash_dir:
        raise ValueError('unexpected linked purge directory')
    paths = [(image_dir / name, trash_dir / name) for name in names]
    if any(source.is_symlink() or trash.is_symlink() for source, trash in paths):
        raise ValueError('unexpected linked purge file')
    return paths


def purge_paths(directory, rel, operation, names):
    """Validate a new purge plan before the journal or any files are written."""
    return _purge_paths(directory, rel, operation, names) if names else []


def publish_journal(directory, staging, previous, operation, names):
    payload = {'schema_version': 1, 'files': {
        name: base64.b64encode(Path(previous[name]).read_bytes()).decode('ascii') if name in previous else None
        for name in state.SAVE_NAMES
    }, 'purge': {'operation': operation, 'images': names} if names else None}
    temp = Path(staging) / JOURNAL
    write_durable(temp, json.dumps(payload, separators=(',', ':')).encode('ascii'))
    os.replace(temp, Path(directory) / JOURNAL)
    sync_directory(directory)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate transaction key')
        result[key] = value
    return result


def _read_journal(path):
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError('journal is not a regular file')
    payload = json.loads(path.read_bytes(), object_pairs_hook=_unique_keys)
    if (not isinstance(payload, dict) or payload.keys() != {'schema_version', 'files', 'purge'}
            or type(payload['schema_version']) is not int or payload['schema_version'] != 1
            or not isinstance(payload['files'], dict) or payload['files'].keys() != set(state.SAVE_NAMES)):
        raise ValueError('invalid journal schema')
    previous = {}
    for name in state.SAVE_NAMES:
        data = payload['files'][name]
        if data is not None and not isinstance(data, str):
            raise ValueError('invalid rollback file contents')
        previous[name] = base64.b64decode(data, validate=True) if data is not None else None
    purge = payload['purge']
    if purge is not None and (not isinstance(purge, dict) or purge.keys() != {'operation', 'images'}):
        raise ValueError('invalid rollback purge plan')
    return previous, purge


def transfer_image(source, destination):
    """Copy and flush the new file before unlinking the source, even across drives."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.label-image-', dir=destination.parent) as staging:
        temp = Path(staging) / destination.name
        with open(source, 'rb') as reader, open(temp, 'wb') as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temp, destination)
        sync_directory(destination.parent)
    source.unlink()
    sync_directory(source.parent)


def finish_transaction(directory):
    sync_directory(directory)
    (Path(directory) / JOURNAL).unlink()
    sync_directory(directory)


def recover_transaction(directory, rel):
    """Rollback to the prior bundle, or block access if the journal cannot be used."""
    directory = Path(directory)
    with state.LOCK:
        try:
            record = _read_journal(directory / JOURNAL)
            if record is None:
                return
            previous, purge = record
            images = _purge_paths(directory, rel, purge['operation'], purge['images']) if purge else []
            for source, trash in images:
                if (source.exists() and not source.is_file()) or (trash.exists() and not trash.is_file()):
                    raise ValueError('unexpected purge file type')
                if not source.exists() and not trash.exists():
                    raise ValueError('both copies of a purge image are missing')

            for source, trash in reversed(images):
                if not source.exists():
                    transfer_image(trash, source)
                elif trash.exists():
                    trash.unlink()
                    sync_directory(trash.parent)

            with tempfile.TemporaryDirectory(prefix='.label-rollback-', dir=directory) as staging:
                for name, content in previous.items():
                    target = directory / name
                    if content is None:
                        target.unlink(missing_ok=True)
                    else:
                        temp = Path(staging) / ('previous_' + name)
                        write_durable(temp, content)
                        os.replace(temp, target)
            finish_transaction(directory)
        except (binascii.Error, OSError, ValueError) as error:
            raise TransactionError(f'label transaction recovery failed for {rel!r}: {error}') from error
