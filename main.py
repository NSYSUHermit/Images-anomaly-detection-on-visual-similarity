import numpy as np
import pandas as pd
import sklearn as sk
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from sklearn.metrics import precision_score, recall_score

dl = DataLoader()
fe = FeatureExtractor()

def Load(folder_path):
    print("###############Load###############")
    dl.folder_path = "D:\\dataset\\Label_harry\\classification\\Circuit"
   
    imgs, label_list = dl.load_data(dl.folder_path)
    print("Shape of traing set: ",np.array(imgs).shape)
    print("Label lens: ",len(label_list))
    return imgs, label_list

def Extract(imgs):
    print("###############Extract###############")
    features_list = fe.extract_imgs_features(imgs, fe.model)
    features_list = features_list.reshape(-1, features_list.shape[2])
    print(np.array(features_list).shape)
    return features_list

def train_test_split():
    print("###############train_test_split###############")
    pass_all = [i for i in range(100,200)]
    train_ok_list = np.random.choice(a=pass_all, size=10, replace=False)
    test_ok_list = [i for i in pass_all if i not in train_ok_list]
    return train_ok_list, test_ok_list

def dimension_reduction(features_list, train_ok_list, test_ok_list):
    print("###############dimension_reduction###############")
    model_dr = fe.dimension_reduction_fit(features_list[train_ok_list])
    redu_feat = fe.dimension_reduction_pred(model_dr, features_list)
    return redu_feat

def autoencoder(redu_feat, train_ok_list, test_ok_list):
    print("###############autoencoder###############")
    train = redu_feat[train_ok_list]
    test_ok = redu_feat[test_ok_list]
    test_ng = redu_feat[0:100]
    test = np.concatenate((test_ok,test_ng), axis = 0)

    ae = fe.autoencoder_fit(train, epochs=100)
    pred_ok = fe.autoencoder_pred(ae, train, test_ok, limits=1)
    pred_ng = fe.autoencoder_pred(ae, train, test_ng, limits=1)

    y_pred = np.concatenate((pred_ok,pred_ng), axis = 0)
    y_pred = np.multiply(y_pred, 1)
    y_true = np.concatenate((np.zeros(len(pred_ok),dtype=int),np.ones(len(pred_ng),dtype=int)), axis = 0)

    prec = sk.metrics.precision_score(y_true, y_pred)
    recall = sk.metrics.recall_score(y_true, y_pred)
    acc = sk.metrics.accuracy_score(y_true, y_pred)

    print(f'{acc:.3f}',"/",f'{prec:.3f}',"/",f'{recall:.3f}')
    return train, test, y_true

def IsolationForest(train, test, y_true):
    print("###############IsolationForest###############")
    from sklearn.ensemble import IsolationForest
    oc_clf = IsolationForest(random_state=0).fit(train)
    oc_pred = oc_clf.predict(test)

    oc_pred = [0 if e == 1 else e for e in oc_pred]
    oc_pred = [1 if e == -1 else e for e in oc_pred]

    prec = sk.metrics.precision_score(y_true, oc_pred)
    recall = sk.metrics.recall_score(y_true, oc_pred)
    acc = sk.metrics.accuracy_score(y_true, oc_pred)

    print(f'{acc:.3f}',"/",f'{prec:.3f}',"/",f'{recall:.3f}')

def OCSVM(train, test, y_true):
    print("###############OCSVM###############")
    from sklearn.svm import OneClassSVM
    oc_clf = OneClassSVM(gamma='auto').fit(train)
    oc_max = oc_clf.score_samples(train).max()
    oc_pred = oc_clf.score_samples(test) < oc_max
    oc_pred = np.multiply(oc_pred, 1)

    prec = sk.metrics.precision_score(y_true, oc_pred)
    recall = sk.metrics.recall_score(y_true, oc_pred)
    acc = sk.metrics.accuracy_score(y_true, oc_pred)

    print(f'{acc:.3f}',"/",f'{prec:.3f}',"/",f'{recall:.3f}')

def Manifolds(train, test, y_true):
    print("###############Manifolds###############")
    tsne_x =  np.concatenate((train,test_ok), axis = 0)
    tsne_x = np.concatenate((tsne_x,test_ng), axis = 0)
    tsne_y = ["train"] * len(train) + ["test_ok"] * len(test_ok) + ["test_ng"] * len(test_ng)
    train_y = ["train"] * len(train) + ["test_ok"] * len(test_ok)
    label_tsne = train_y + label_list[750:]

    from sklearn.manifold import TSNE
    import seaborn as sns
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(tsne_x) 
    y = label_tsne
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    sns.scatterplot(x="comp-1", y="comp-2", hue="y",
                    palette=sns.color_palette("Set2"),
                    style="y",
                    data=df).set(title="LIDL data T-SNE projection") 
     
    import matplotlib.pyplot as plt
    plt.show()

def Clustering(train):    
    print("###############Clustering###############")
    cluster = fe.cluster_feature_vectors(train, 2)
    key = max(cluster.tolist(), key = cluster.tolist().count)
    cluster_train = train[cluster == key]
    print(cluster_train.shape)
    train = cluster_train

    df = pd.DataFrame(label)
    df[0] = df[0].map({0: False, 1: True})
    int_label_list = df[0].values.tolist()

    print("Cluster list accuracy：")
    print(sk.metrics.accuracy_score(cluster_list.tolist(), int_label_list))

if __name__ == '__main__':
    print("Input size:",dl.target_size)
    folder_path = "D:\\dataset\\Label_harry\\classification\\Circuit"
    #folder_path = "D:\\dataset\\Label_harry\\object_detection\\LIDL_dataset"

    imgs, label_list = Load(folder_path)
    features_list = Extract(imgs)
    train_ok_list, test_ok_list = train_test_split()
    redu_feat = dimension_reduction(features_list, train_ok_list, test_ok_list)
    train, test, y_true = autoencoder(redu_feat, train_ok_list, test_ok_list)





