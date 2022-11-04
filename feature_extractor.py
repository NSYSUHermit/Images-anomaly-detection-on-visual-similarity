# for loading/processing the images  
from keras.applications.vgg16 import preprocess_input 

# models 
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import Xception
#import keras_efficientnet_v2

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering

# for everything else
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

class FeatureExtractor:
    def __init__(self):
        #self.model = keras_efficientnet_v2.EfficientNetV2S(pretrained="imagenet", include_preprocessing=False)
        self.model = ResNet50()
        self.model = Model(inputs = self.model.inputs, outputs = self.model.layers[-2].output)
        self.config = self.model.get_config()
        self.input_size = self.config["layers"][0]["config"]["batch_input_shape"][1]

    def extract_cnn_features(self, img, model):
        reshaped_img = img.reshape(1, self.input_size, self.input_size, 3)
        features = model.predict(reshaped_img, use_multiprocessing=True, verbose=0)
        return features

    def extract_imgs_features(self, imgs, model):
        features_list = []
        num1 = 0
        total = len(imgs)
        for img in imgs:
            features = self.extract_cnn_features(img, model)
            features_list.append(features)
            # time line
            num1 += 1
            print('\r' + '[CNN Features Progress]:[%s%s]%.2f%%;' % ('█' * int(num1*20/total), ' ' * (20-int(num1*20/total)),float(num1/total*100)), end='')

        # get a list of just the features
        features_list = np.array(features_list)
        print('\n')
        return features_list

    def dimension_reduction_fit(self, feat, n_clusters=50):
        #pca = PCA(n_components=10, random_state=22)
        #pca.fit(feat)
        #redu_feat = pca.transform(feat)
        model_dr = cluster.FeatureAgglomeration(n_clusters = n_clusters)
        model_dr.fit(feat)
        return model_dr

    def dimension_reduction_pred(self, model_dr, feat):
        redu_feat = model_dr.transform(feat)
        return redu_feat

    def cluster_feature_vectors(self, feat, n_clusters):
        clustering = KMeans(n_clusters, random_state=22)
        clustering.fit(feat)
        cluster_list = clustering.predict(feat)

        #clustering = AgglomerativeClustering(n_clusters).fit(feat)
        #cluster_list = clustering.labels_
        return cluster_list

    def get_cluster_file(self, nlist, cluster_list):
        for file, cluster in zip(nlist, cluster_list):
            if cluster not in groups.keys():
                groups[cluster] = []
                groups[cluster].append(file)
            else:
                groups[cluster].append(file)
        return groups

    def autoencoder_fit(self, redu_feat, epochs=200):
        redu_feat_cast = tf.cast(redu_feat, tf.float32)
        input_dim = redu_feat_cast.shape[1]
        BATCH_SIZE = 512

        # https://keras.io/layers/core/
        autoencoder = tf.keras.models.Sequential([
            # deconstruct / encode
            tf.keras.layers.Dense(input_dim, activation='relu', input_shape=(input_dim, )), 
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            # reconstruction / decode
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='relu')
        ])

        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        from tensorflow import keras
        opt=tf.keras.optimizers.Adam()
        autoencoder.compile(optimizer=opt, loss='mse')

        history = autoencoder.fit(redu_feat_cast, redu_feat_cast, 
                  epochs=epochs, 
                  batch_size=BATCH_SIZE,
                  validation_data=(redu_feat_cast, redu_feat_cast),
                  callbacks=[tensorboard_callback],
                  shuffle=True)
        return autoencoder

    def autoencoder_pred(self, autoencoder, train, test, limits=1.1):
        re_train = autoencoder.predict(train) 
        re_test = autoencoder.predict(test)      
        pred = np.mean(((re_test - test) ** 2), axis=1) > np.mean(((re_train - train) ** 2), axis=1).max() * limits
        return pred
