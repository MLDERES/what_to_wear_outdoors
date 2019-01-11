import os
import tempfile
from pathlib import Path
from unittest import TestCase

import utility


class Test_utility(TestCase):
    def test_is_file_age(self):
        fp = tempfile.NamedTemporaryFile()
        fname = fp.name
        fp.write(b'Hello world!')
        assert not utility.file_age(fname, hours=1)
        assert utility.file_age(fname, hours=1, compare='younger')
        assert not utility.file_age('c:/temp/md.md')
        assert not utility.file_age(Path('c:/temp/md.md'))
