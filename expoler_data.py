import pickle
from os.path import join, dirname

from sklearn.model_selection import train_test_split

from src.load_data import load_dataset

if __name__ == '__main__':
    train_path = join(dirname(__file__), "data", "train.xlsx")
    dev_path = join(dirname(__file__), "data", "dev.xlsx")
    test_path = join(dirname(__file__), "data", "test.xlsx")
    # data tu gan nhan
    data2_path = join(dirname(__file__), "data", "data.xlsx")

    serialization_dir = join(dirname(__file__), "snapshots")
    print("Load data...")
    X_train, y_train = load_dataset(train_path)
    X_dev, y_dev = load_dataset(dev_path)
    X_test, y_test = load_dataset(test_path)
    X_train2, y_train2 = load_dataset(data2_path)


    print("train: ",len(X_train))
    print("dev: ",len(X_dev))
    print("test: ",len(X_test))
    print("datae: ",len(X_train2))


    X= X_train+X_dev+ X_train2+X_test
    y = y_train+y_dev+ y_train2 +y_test
    print(len(X))
    print(len(y))

    X_train,X_test, y_train,y_test =  train_test_split(X,y , train_size=0.8,shuffle=True)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train,y_train,train_size=0.8,shuffle=True)
    print(len(y_train))

    print(len(y_dev))
    print(len(y_test))

    # pickle.dump(X_train,open("./data/X_train.pkl","wb"))
    # pickle.dump(y_train,open("./data/y_train.pkl","wb"))
    #
    #
    #
    # pickle.dump(X_dev,open("./data/X_dev.pkl","wb"))
    # pickle.dump(y_dev,open("./data/y_dev.pkl","wb"))
    #
    #
    # pickle.dump(X_test,open("./data/X_test.pkl","wb"))
    # pickle.dump(y_test,open("./data/y_test.pkl","wb"))
