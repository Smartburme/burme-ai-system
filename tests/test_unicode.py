# tests/test_unicode.py တွင် ထည့်သွင်းစမ်းသပ်ပါ
def test_unicode_normalization():
    test_text = "ေကာင္းပါတယ္"  # Zawgyi text
    processor = BurmeseTextProcessor()
    normalized = processor.normalize_unicode(test_text)
    assert "္" not in normalized  # Zawgyi marker should be removed
