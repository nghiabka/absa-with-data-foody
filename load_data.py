import pandas as pd

from src.normalize import normalize_text


def load_dataset(path):
    df = pd.read_excel(path)
    y = df.drop("text", 1)

    print(y.shape)
    if "RESTAURANT#MISCELLANEOUS#NEGATIVE" in df:
        y = y.drop("RESTAURANT#MISCELLANEOUS#NEGATIVE",1)
    if "RESTAURANT#MISCELLANEOUS#POSITIVE" in df:
        y = y.drop("RESTAURANT#MISCELLANEOUS#POSITIVE",1)
    if "RESTAURANT#MISCELLANEOUS#NEUTRAL" in df:
        y = y.drop("RESTAURANT#MISCELLANEOUS#NEUTRAL",1)
    print(y.shape)
    X = list(df["text"])
    # for x in X:
    #     if len(x) < 10:
    #         print(x)
    X = [normalize_text(x) for x in X]

    columns = y.columns
    temp = y.apply(lambda item: item > 0)
    y = list(temp.apply(lambda item: list(columns[item.values]), axis=1))
    return X, y



if __name__ == '__main__':
    X,y = load_dataset("./data/train.xlsx")
    #
    print(len(X))
    print(len(y))
    labels = list(set(sum([s for s in y],[])))
    for i in sorted(labels):
        print(i)
    print(len(labels))
    # df = pd.read_excel("./data/train.xlsx")
    # print(df.columns)



