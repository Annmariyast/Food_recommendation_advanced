# utils/extract_foods_nlp.py
import spacy

# Load English NLP model lazily with fallback to simple split if model missing
_nlp = None
_nlp_error = None
try:
    _nlp = spacy.load("en_core_web_sm")
except Exception as e:
    _nlp_error = str(e)

def extract_food_items(text):
    if not text:
        return []
    if _nlp is None:
        # Fallback: naive tokenization on commas and spaces
        raw = [t.strip() for part in text.split(',') for t in part.split()]
        tokens = [t for t in raw if len(t) > 2 and t.isalpha()]
        return list(sorted(set(tokens)))

    doc = _nlp(text.lower())
    food_list = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
            food_list.append(token.text)
    return list(sorted(set(food_list)))
