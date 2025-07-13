import re
import unicodedata
from myanmar import phonetics
from config.settings import Config

class MyanmarTextPreprocessor:
    def __init__(self):
        self.config = Config()
        
    def normalize_unicode(self, text):
        """Zawgyi to Unicode normalization"""
        return unicodedata.normalize('NFKC', text)
    
    def clean_text(self, text):
        """Basic Myanmar text cleaning"""
        text = self.normalize_unicode(text)
        text = re.sub(r'[^\u1000-\u109F\uAA60-\uAA7F\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def segment_syllables(self, text):
        """Myanmar syllable segmentation"""
        return [phonetics.segment(word) for word in text.split()]
