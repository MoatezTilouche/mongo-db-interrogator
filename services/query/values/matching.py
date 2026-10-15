"""Language-independent spelling keys; stored values are never rewritten."""
import re
import unicodedata


def normalize_for_match(text):
    text = unicodedata.normalize("NFKC", text)
    text = "".join(char for char in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(char))
    split = []
    for index, char in enumerate(text):
        if index:
            previous = text[index - 1]
            following = text[index + 1] if index + 1 < len(text) else ""
            if ((previous.islower() and char.isupper())
                    or (previous.isupper() and char.isupper() and following.islower())
                    or (previous.isalpha() and char.isdigit())
                    or (previous.isdigit() and char.isalpha())):
                split.append(" ")
        split.append(char)
    normalized = re.sub(r"[_\-./\s]+", " ", "".join(split).casefold()).strip()
    return normalized, normalized.split()
