# Image Caption Generation

The dataset can be downloaded from https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k. The image caption can be downloaded from https://github.com/ysbecca/flickr8k-custom/tree/main/captions.

The basic principle of our image-to-text model is as pictured in the diagram below, where an Encoder network encodes the input image as a feature vector by providing the output of fully-connected layer of a pre-trained CNN. This pre-trained network has been trained on the complete ImageNet dataset and is thus  
