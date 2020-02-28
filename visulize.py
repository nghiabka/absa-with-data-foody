import pickle

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import pickle

# objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
# y_pos = np.arange(len(objects))
# performance = [10, 8, 6, 4, 2, 1]
#
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Usage')
# plt.title('Programming language usage')
#
# plt.show()


def thongke(y,target_names):
    y_val = []
    for i  in range(len(target_names)):
        count = 0
        for labs in y:
            for l  in labs:
                if l == target_names[i]:
                    count +=1
        y_val.append(count)
    return  y_val

def draw_graph(y_val, target_names):
    objects = np.arange(len(target_names))

    y_pos = np.arange(len(objects))
    performance = y_val
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('number')
    plt.title('number of aspect_polary')

    plt.show()

if __name__ == '__main__':
    X_train = pickle.load(open("./data/X_train.pkl", "rb"))
    y_train = pickle.load(open("./data/y_train.pkl", "rb"))

    X_dev = pickle.load(open("./data/X_dev.pkl", "rb"))
    y_dev = pickle.load(open("./data/y_dev.pkl", "rb"))
    #
    X_test = pickle.load(open("./data/X_test.pkl", "rb"))
    y_test = pickle.load(open("./data/y_test.pkl", "rb"))
    target_names = sorted(list(set(sum([s for s in y_train], []))))
    print(target_names[17])
    # train
    y_val = thongke(y_train,target_names)
    draw_graph(y_val,target_names)

    # dev
    y_val = thongke(y_dev, target_names)
    draw_graph(y_val, target_names)
# test
    y_val = thongke(y_test, target_names)
    draw_graph(y_val, target_names)






