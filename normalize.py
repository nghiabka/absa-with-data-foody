import string

import re

import gensim
from pyvi import ViTokenizer


def remove_punctuation(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(i for i in text.split())
    return text


def remove_emoji(string):
    # Emojis pattern
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u'\U00010000-\U0010ffff'
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def normalize_text(raw_text):

    text = " ".join(i for i in raw_text.split())

    filtered_text = remove_punctuation(text)
    clean_text = remove_emoji(filtered_text)

    filtered_text = gensim.utils.simple_preprocess(filtered_text)
    filtered_text = ' '.join(filtered_text)
    clean_text = ViTokenizer.tokenize(filtered_text)

    return clean_text.lower()



if __name__ == '__main__':
    print(normalize_text("🙂 cmt cho đỡ mất huy hiệu thôi 🙂 chứ t với ny t cũng như đang trong phim hành động rồi.."))