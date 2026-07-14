from swathi_ai.language import clean_text, detect_language

def test_clean_text(): assert clean_text("  Hello   WORLD ") == "hello world"
def test_tamil(): assert detect_language("வணக்கம்") == "tamil"
def test_tanglish(): assert detect_language("epdi irukinga") == "tanglish"
def test_english(): assert detect_language("How are things today?") == "english"
