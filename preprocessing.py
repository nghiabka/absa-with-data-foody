from os.path import dirname, join
from random import shuffle

import pandas as pd
import re


def read(path):
    with open(path, encoding='utf-8') as f:
        content = f.read().split("\n\n")
    return content


def transform(s):
    sentence = {}
    sentence["text"] = s.split("\n")[1]
    sentiments = s.split("\n")[2]
    sentiments_ = re.split("}, +{", sentiments)
    sentiments__ = [re.sub(r"[{}]", "", item) for item in sentiments_]
    labels = [item.upper().replace(", ", "#") for item in sentiments__]
    sentence["labels"] = labels
    return sentence


def convert_to_corpus(sentences, file_path):
    data = []
    labels = sorted(list(set(sum([s["labels"] for s in sentences], []))))
    for s in sentences:
        item = {}
        item["text"] = s["text"]
        for label in labels:
            if label in s["labels"]:
                item[label] = 1
            else:
                item[label] = 0
        data.append(item)
    shuffle(data)
    df = pd.DataFrame(data)
    columns = ["text"] + labels
    df.to_excel(file_path, index=False, columns=columns)


if __name__ == '__main__':
    # path = join(dirname(dirname(dirname(__file__))), "data", "vlsp2018", "restaurant")
    corpus = join(dirname(__file__),"data")
    print(corpus)
    train_data = read(join("../data/1-VLSP2018-SA-Restaurant-train (7-3-2018).txt"))
    train_data = [transform(sent) for sent in train_data]
    convert_to_corpus(train_data, join(corpus, "train.xlsx"))
    #
    dev_data = read(join("../data/2-VLSP2018-SA-Restaurant-dev (7-3-2018).txt"))
    dev_data = [transform(sent) for sent in dev_data]
    convert_to_corpus(dev_data, join(corpus, "dev.xlsx"))

    test_data = read(join("../data/3-VLSP2018-SA-Restaurant-test-eval-gold-data (8-3-2018).txt"))
    test_data = [transform(sent) for sent in test_data]
    convert_to_corpus(test_data, join(corpus, "test.xlsx"))