import unittest
from src.preprocessing import MyanmarTextPreprocessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.processor = MyanmarTextPreprocessor()
        
    def test_clean_text(self):
        test_text = "ဒီစာကြောင်း တစ်ခုပါ! 123"
        result = self.processor.clean_text(test_text)
        self.assertEqual(result, "ဒီစာကြောင်း တစ်ခုပါ")
        
    def test_syllable_segmentation(self):
        test_text = "မင်္ဂလာပါ"
        result = self.processor.segment_syllables(test_text)
        self.assertEqual(result, [['မ', 'င်္ဂ', 'လာ', 'ပါ']])
