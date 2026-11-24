import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Input, OptionList, RichLog, Static

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import metrology
from metrology_tui import Fields, MetrologyApp


class ConsoleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.setup = Path(tmp.name) / 'uc60'
        metrology.save_setup(self.setup, metrology.DEFAULT_OPTICS, metrology.DEFAULTS)
        self.app = MetrologyApp(self.setup)

    async def test_search_and_launch_preflight_to_live_log(self):
        args = [sys.executable, '-u', '-c', 'print("offline ok")']
        with patch.object(metrology, 'command', return_value=args) as cmd:
            async with self.app.run_test(size=(100, 30)) as pilot:
                await pilot.press('ctrl+k')
                self.app.query_one('#search', Input).value = 'проверить'
                await pilot.pause()
                self.assertEqual(self.app.query_one(OptionList).option_count, 1)
                await pilot.press('enter')
                await asyncio.wait_for(self.app.workers.wait_for_complete(), timeout=10)
                cmd.assert_called_once_with('check', self.setup, None)
                self.assertTrue(self.app.in_journal)
                self.assertTrue(any('offline ok' in line.text for line in self.app.query_one(RichLog).lines))
                await pilot.press('escape')
                self.assertFalse(self.app.in_journal)

    async def test_named_setup_form_validates_and_persists(self):
        with patch.object(metrology, 'SETUPS', self.setup.parent):
            async with self.app.run_test(size=(65, 24)) as pilot:
                await pilot.press('ctrl+k')
                self.app.query_one('#search', Input).value = 'создать'
                await pilot.pause()
                await pilot.press('enter')
                form = self.app.screen
                self.assertIsInstance(form, Fields)
                form.query_one('#name', Input).value = '../bad'
                form.apply()
                self.assertTrue(str(form.query_one('#error', Static).content))
                form.query_one('#name', Input).value = 'test-table'
                form.query_one('#area', Input).value = '68,5 47'
                form.apply()
                await pilot.pause()
                self.assertEqual(self.app.setup.name, 'test-table')
                self.assertEqual(metrology.read_config(self.app.setup)['area_mm'], [68.5, 47])
                self.assertEqual((self.setup.parent / 'active.txt').read_text(), 'test-table')
                self.assertFalse((self.app.setup / 'table_frame.json').exists())

    async def test_archive_selection_and_cancel_dont_launch_camera(self):
        path = ROOT / 'mosaics' / 'offline example'
        with patch.object(metrology, 'result_paths', return_value=[path]), patch.object(self.app, 'start_job') as job:
            async with self.app.run_test() as pilot:
                await pilot.press('down', 'down', 'down', 'enter')
                self.assertEqual(self.app.page, 'mosaic')
                await pilot.press('enter')
                self.assertEqual(job.call_args.args[1][-2:], ['--view', str(path)])
                await pilot.press('escape')
                self.assertEqual(self.app.page, 'home')
                self.assertEqual(job.call_count, 1)
