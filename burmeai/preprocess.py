import re
from myanmar import phonetics

class BurmeseTextProcessor:
    def __init__(self):
        self.stopwords = ["များ", "တယ်", "သည်"]  # Myanmar stopwords
        
    def clean_text(self, text):
        # Remove non-Myanmar characters
        text = re.sub(r'[^\u1000-\u109F\uAA60-\uAA7F]+', ' ', text)
        return text.strip()
    
    def syllable_segment(self, text):
        return phonetics.segment(text)
