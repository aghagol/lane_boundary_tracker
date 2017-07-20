import os
import sys
import unittest
import line_connector
from filecmp import dircmp
import difflib

curr_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.dirname(curr_dir)
sys.path.append(src_dir)
data_dir = curr_dir + "/data"


class TestSurfaceStreetLineConnector(unittest.TestCase):
    images = data_dir + "/images"
    poses = data_dir + "/poses"
    chunks = data_dir + "/chunks"
    drives = data_dir + "/drives.txt"
    config = src_dir + "/conf.json"
    cache = "/tmp/sslc"
    fuses = cache + "/output/fuse"
    tagged = cache + "/fuse_tagged"

    def print_diff_files(self, diff):
        for name in diff.diff_files:
            expected = diff.left + "/" + name
            actual = diff.right + "/" + name

            with open(expected, 'r') as f:
                expected_lines = f.readlines()
            with open(actual) as f:
                actual_lines = f.readlines()

            udiff = difflib.unified_diff(expected_lines, actual_lines, expected, actual, n=True)
            sys.stdout.writelines(udiff)

            self.fail("Unexpected difference in %s found in %s and %s" % (name, diff.left, diff.right))

        for sub_diff in diff.subdirs.values():
            self.print_diff_files(sub_diff)

    def test_SingleImage(self):
        line_connector.run(self.images, self.fuses, self.tagged, self.cache, self.drives, self.poses,
                           self.chunks, self.config, 0, 1)

        diff = dircmp(data_dir + "/expected", self.cache)
        self.assertTrue(len(diff.right_only) == 0)
        self.assertTrue(len(diff.left_only) == 0)
        self.print_diff_files(diff)

if __name__ == "__main__":
    unittest.main()
