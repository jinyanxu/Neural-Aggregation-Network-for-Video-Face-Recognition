# Introduction

This is a repository of reproducing this paper: [CVPR2017: Neural Aggregation Network for Video Face Recognition on](https://arxiv.org/abs/1603.05474)Tensorflow platform.

# Content

src/extract_feature.py this python file extract feature using sphereface model and save it into .mat file.

src/data.py load the feature and generate a batch for network training.

src/crop-alignment.py detect the face and align it and save the face on disk.

src/network.py  the aggregation network which consists of two attention modules.

src/log this is a directory that save the training loss and training accuracy by tf.summary

src/model this is the model file of network. 



# Test 

updating.





