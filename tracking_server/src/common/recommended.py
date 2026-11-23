"""Recommended-model mark (models/RECOMMENDED.json): one place that says which detector and which appearance model are
recommended now, with the date and the evidence. The page shows ' · <label> (<date>)' after that model's name. Change the file,
restart the server, and the mark moves; the per-model name files never carry it."""
import json
from pathlib import Path

from .shared_values import HERE

FILE = HERE / 'models' / 'RECOMMENDED.json'


def read(path=None):
    """-> {'detector': {...}, 'reid': {...}} or {} when the file is missing or broken (no mark is shown then)."""
    try:
        data = json.loads(Path(path or FILE).read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def mark(kind, key, path=None):
    """' · 추천 (2026-11-20)' when `key` is the recommended model of `kind` ('detector' or 'reid'), else ''."""
    entry = read(path).get(kind)
    if not isinstance(entry, dict) or entry.get('key') != key:
        return ''
    label = str(entry.get('label') or '추천').strip()[:20]
    date = str(entry.get('date') or '').strip()[:10]
    return ' · %s%s' % (label, ' (%s)' % date if date else '')
