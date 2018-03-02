# author: Wang Yongjie
# Email:  wangyongjie@ict.ac.cn

"""
generate train batch for deep fusion module
"""

import scipy.io as sio
import sys
import random

class Data(object):

    def __init__(self, filename, batch_size, class_num):
        """
        filename: feature name 
        batch_size: train_batch
        class_num:  class number
        """
        self.filename = filename
        self.batch_size = batch_size
        self.class_num = class_num
    
    def load_feature(self):
        self.features = []
        self.labels = []
        dataset = sio.loadmat(self.filename)
        flag = 0
        f = open("dataset.txt", 'w')
        not_include = ["__version__", "__globals__", "__header__"] # scipy.io.savemat save these unrelevant information
        for k, v in dataset.iteritems():
            if k not in not_include:
                label = [0] * self.class_num
                #print flag
                label[flag] = 1
                flag = flag + 1
                sub_feature = []
                for i in range(len(v)):
                    sub_feature.append(v[i])
                self.labels.append(label)
                self.features.append(sub_feature)

        #pairs = list(zip(self.features, self.labels))
        #random.shuffle(pairs)
        #self.features, self.labels = zip(*pairs)
        f.close()

    def next_batch(self, group_num):
        """
        frame numbers of each group
        """
        train_feature, train_label = [], []
        start = random.randint(0, self.class_num)
        for i in range(start, start + self.batch_size):
            train_group = []
            seed = random.randint(0, len(self.features[i % self.class_num]) - group_num)
            for j in range(seed, seed + group_num):
                #print i, j
                train_group.append(self.features[i % self.class_num][j])

            train_feature.append(train_group)
            train_label.append(self.labels[i % self.class_num])

        return train_feature, train_label


if __name__ == "__main__":
    filename = "./YoutubeFaces.mat"
    dataset = Data(filename, 3, 1595) 
    dataset.load_feature()
    train_features, train_label = dataset.next_batch(5)
    print train_features, train_label

