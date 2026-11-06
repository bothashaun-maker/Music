"""Opening-section export regression checks; no audio/model execution."""
import csv
import importlib.util
import io
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RegionExportTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault('STS_AUTOLAUNCH', '0')
        spec = importlib.util.spec_from_file_location('region_export_engine', ROOT / 'timestamp_studio/engine.py')
        self.engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.engine)

    def rows(self, entries, duration=60):
        return list(csv.DictReader(io.StringIO(self.engine.fmt_reaper(entries, duration))))

    def entries(self, first=10.66):
        return [{'kind':'tag', 'text':'Intro', 'start':first},
                {'kind':'sing', 'text':'First lyric', 'start':first},
                {'kind':'tag', 'text':'Verse 1', 'start':29.812},
                {'kind':'sing', 'text':'Verse lyric', 'start':29.812}]

    def test_intro_starts_at_zero_while_lyric_markers_remain_at_vocal_onsets(self):
        rows = self.rows(self.entries())
        markers = [r for r in rows if r['#'].startswith('M')]
        regions = [r for r in rows if r['#'].startswith('R')]
        self.assertEqual(markers[0]['Start'], '00:00:10.660')
        self.assertEqual(markers[1]['Start'], '00:00:29.812')
        self.assertEqual(regions[0]['Start'], '00:00:00.000')
        self.assertEqual(regions[0]['End'], '00:00:29.812')
        self.assertEqual(regions[0]['Length'], '00:00:29.812')
        self.assertEqual(regions[1]['Start'], regions[0]['End'])

    def test_short_opening_intro_also_starts_at_exact_zero(self):
        rows = self.rows(self.entries(.25))
        self.assertEqual(rows[0]['Start'], '00:00:00.250')
        self.assertEqual(next(r for r in rows if r['#']=='R1')['Start'], '00:00:00.000')

    def test_region_boundaries_match_chapters_and_do_not_mutate_entries(self):
        entries = self.entries()
        rows = self.rows(entries)
        chapters, _ = self.engine.build_chapters(entries, 60)
        regions = [r for r in rows if r['#'].startswith('R')]
        self.assertEqual([r['Start'] for r in regions],
            [self.engine.fmt_clock(c['start'], force_hours=True) for c in chapters])
        self.assertEqual(entries[0]['start'], 10.66)

    def test_first_verse_with_a_lead_in_gets_the_same_intro_as_chapters(self):
        entries = [{'kind':'tag','text':'Verse 1','start':10},
                   {'kind':'sing','text':'Opening words','start':10}]
        regions = [r for r in self.rows(entries) if r['#'].startswith('R')]
        self.assertEqual([(r['Name'],r['Start']) for r in regions],
                         [('Intro','00:00:00.000'),('Verse 1','00:00:10.000')])

    def test_unsectioned_lyrics_keep_existing_marker_only_export(self):
        rows = self.rows([{'kind':'sing','text':'Opening words','start':10}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['#'], 'M1')
        self.assertEqual(rows[0]['Start'], '00:00:10.000')


if __name__ == '__main__':
    unittest.main()
