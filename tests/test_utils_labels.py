import unittest
from pathlib import Path

from utils import extract_folder_label, extract_label_from_filename


class UtilsLabelExtractionTests(unittest.TestCase):
    def test_extract_label_from_filename_numeric_positive(self):
        self.assertEqual(extract_label_from_filename('atelectasis_1'), {'Atelectasis': 1})

    def test_extract_label_from_filename_numeric_negative(self):
        self.assertEqual(extract_label_from_filename('atelectasis-0'), {'Atelectasis': 0})

    def test_extract_label_from_filename_keyword_positive(self):
        self.assertEqual(extract_label_from_filename('positive_pneumonia_case_22'), {'Pneumonia': 1})

    def test_extract_label_from_filename_keyword_negative(self):
        self.assertEqual(extract_label_from_filename('nodule_negative_case5'), {'Nodule': 0})

    def test_extract_folder_label_uses_filename_first(self):
        path = Path('xrays/Pneumonia_positive/atelectasis_1.jpeg')
        self.assertEqual(extract_folder_label(path), {'Atelectasis': 1})

    def test_extract_folder_label_no_false_positive_when_unknown(self):
        path = Path('xrays/unknown_folder/scan_007.jpeg')
        self.assertIsNone(extract_folder_label(path))


if __name__ == '__main__':
    unittest.main()
