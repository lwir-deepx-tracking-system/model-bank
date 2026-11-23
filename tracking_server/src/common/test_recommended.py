"""models/RECOMMENDED.json mark on model names (2026-11-20)."""
import json
import tempfile
import unittest
from pathlib import Path

from . import recommended


class RecommendedTest(unittest.TestCase):
    def test_shipped_file_names_existing_models(self):
        data = recommended.read()
        self.assertIn('detector', data)
        self.assertIn('reid', data)
        self.assertTrue(recommended.mark('detector', data['detector']['key']).startswith(' · 추천'))

    def test_mark_only_for_the_recommended_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'r.json'
            p.write_text(json.dumps({'detector': {'key': 'a', 'label': '추천', 'date': '2026-11-20'}}), encoding='utf-8')
            self.assertEqual(recommended.mark('detector', 'a', p), ' · 추천 (2026-11-20)')
            self.assertEqual(recommended.mark('detector', 'b', p), '')
            self.assertEqual(recommended.mark('reid', 'a', p), '')

    def test_missing_or_broken_file_gives_no_mark(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(recommended.mark('detector', 'a', Path(d) / 'none.json'), '')
            p = Path(d) / 'bad.json'; p.write_text('{', encoding='utf-8')
            self.assertEqual(recommended.mark('detector', 'a', p), '')


if __name__ == '__main__':
    unittest.main()
