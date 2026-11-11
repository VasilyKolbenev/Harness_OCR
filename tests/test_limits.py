from pathlib import Path
from dataclasses import replace
import unittest
from harness_ocr.settings import Settings

class LimitTests(unittest.TestCase):
    def test_nonfinite_timeout_cannot_disable_watchdog(self):
        base=Settings(Path('data'),Path('models'))
        for value in [float('nan'),float('inf'),float('-inf'),0,-1,True,'60']:
            for attr in ['call_timeout_seconds','document_timeout_seconds']:
                with self.subTest(value=value,attr=attr),self.assertRaises(ValueError):
                    replace(base,**{attr:value}).validate()
        base.validate()

if __name__=='__main__':unittest.main()
