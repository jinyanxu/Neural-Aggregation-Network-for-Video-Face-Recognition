#!/usr/bin/python

# Author: Wang Yongjie
# Email:  wangyongjie@ict.ac.cn

import os
import sys
import caffe
import scipy.io as sio
import argparse
import numpy as np
import copy
# from sklearn.decomposition import PCA


class cnn_feature(object):
    """
    extract facial feature from cnn neural network.

    """
    def __init__(self, prototxt, weights, layer, gpu = True):

        """
        default construct function

        - prototxt:  string, cnn structure(caffe prototxt)
        - weights:   string, network weights file name
        - gpu:       boolean,gpu or cpu mode
        - layer:     extract layer's feature

        """
        self.prototxt = prototxt
        self.weights = weights
        self.gpu = gpu
        self.layer = layer

    def load_network(self):
        """
        load network from the prototxt and weights

        """
        if self.gpu:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Net(self.prototxt, self.weights, caffe.TEST)
        self.height = self.net.blobs["data"].data.shape[2]
        self.width = self.net.blobs["data"].data.shape[3]
        self.channels = self.net.blobs["data"].data.shape[1]


    def extract_feature(self, image_dir, feature_name):
        """
        extract feature from specified directory and save in feature_dir
        image_dir:  string, face image directory
        feature_name:    string, feature name ended with .mat
        """

        assert type(image_dir) == str and type(feature_name) == str
        assert feature_name.split(".")[-1] == "mat"

        self.transformer = caffe.io.Transformer({'data':self.net.blobs['data'].data.shape})
         
        # [height, width, channels] -> [channels, height, width]
        self.transformer.set_transpose('data', (2, 0, 1))
        # RGB2BGR
        self.transformer.set_channel_swap('data', (2, 1, 0))
        # 0 - 255
        self.transformer.set_raw_scale('data', 255.0)

        self.net.blobs['data'].reshape(1, 3, 112, 96)
        feature_set = {}

        f = open("feature.txt", "w")

        for term in os.listdir(image_dir):
            sub_img_dir = os.path.join(image_dir, term)
            sub_feature_list = []
            f.write(term + "\n")
            for subitem in os.listdir(sub_img_dir):
                sub_sub_img_dir = os.path.join(sub_img_dir, subitem)
                for iterm in os.listdir(sub_sub_img_dir):
                    filename = os.path.join(sub_sub_img_dir, iterm)
                    #print filename, iterm
                    # featurename = os.path.join(sub_fea_dir, iterm)
                    img = caffe.io.load_image(filename)
                    if len(img) == 0:
                        print "open " + filename + " error!"
                        continue

                    self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img)
                    self.net.forward()
                    # extract feature
                    feature = copy.copy(self.net.blobs[self.layer].data[0])
                    sub_feature_list.append(feature)

            feature_set[term] = sub_feature_list

        sio.savemat(feature_name, feature_set)
        f.close()


def parser_args():
    """
    parser argument

    """
    parser = argparse.ArgumentParser(description = "extract cnn feature")
    parser.add_argument("-p", "--prototxt", type = str, default = "/home/wyj/experiment/sphereface/train/code/sphereface_deploy.prototxt")
    parser.add_argument("-m", "--model", type = str, default = "/home/wyj/experiment/sphereface/train/code/sphereface_model.caffemodel")
    parser.add_argument("-l", "--layer", type = str, default = "fc5")
    parser.add_argument("-g", "--gpu", type = bool, default = True)

    parser.add_argument("-d", "--directory", type = str, default = "/media/hysia/wyj/dataset/face_recog/YoutubeFaces-crop-align/")
    parser.add_argument("-n", "--name", type = str, default = "YoutubeFaces.mat")
    args = parser.parse_args()

    return args.prototxt, args.model, args.layer, args.gpu, args.directory, args.name


if __name__ == "__main__":

    prototxt, model, layer, gpu, directory, name = parser_args()

    Extracter = cnn_feature(prototxt, model, layer, layer)
    Extracter.load_network()
    Extracter.extract_feature(directory, name)

