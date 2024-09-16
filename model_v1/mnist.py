import numpy as np
from tensorflow import keras
from skimage.util import view_as_windows
from pixelhop import Pixelhop
from skimage.measure import block_reduce
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings, gc
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


np.random.seed(1)

# Preprocess
N_Train_Reduced = 10000    # 10000
N_Train_Full = 60000     # 50000
N_Test = 10000            # 10000

BS = 2000 # batch size


def shuffle_data(X, y):
    shuffle_idx = np.random.permutation(y.size)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    return X, y


def select_balanced_subset(images, labels, use_num_images):
    '''
    select equal number of images from each classes
    '''
    num_total, H, W, C = images.shape
    num_class = np.unique(labels).size
    num_per_class = int(use_num_images / num_class)

    # Shuffle
    images, labels = shuffle_data(images, labels)

    selected_images = np.zeros((use_num_images, H, W, C))
    selected_labels = np.zeros(use_num_images)

    for i in range(num_class):
        selected_images[i * num_per_class:(i + 1) * num_per_class] = images[labels == i][:num_per_class]
        selected_labels[i * num_per_class:(i + 1) * num_per_class] = np.ones((num_per_class)) * i

    # Shuffle again
    selected_images, selected_labels = shuffle_data(selected_images, selected_labels)

    return selected_images, selected_labels

def Shrink(X, shrinkArg):
    #---- max pooling----
    pool = shrinkArg['pool']

    out = X
    # Make two different things (if pool is True or if pool is False)
    # if False do nothing, if True, do 2x2 max-pooling
    if pool is False:
        pass
    elif pool is True:
        N, H, W, C = X.shape
        pool_height, pool_width = 2, 2
        stride = 2
        # reshape
        x_reshaped = X.reshape(N, H // pool_height, pool_height,
                            W // pool_width, pool_width, C)
        # pool in axis=2 first and then axis=3
        out = x_reshaped.max(axis=2).max(axis=3)

    #---- neighborhood construction
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    pad = shrinkArg['pad']

    ch = X.shape[-1]
    # pad
    if pad > 0:
    # pad at axis=1 and axis=2
        out = np.pad(out,((0,0), (pad,pad), (pad,pad), (0,0)), 'reflect')

    # neighborhood construction
    out = view_as_windows(out, (1,win,win,ch), (1,stride,stride,ch))

    # return array
    return out.reshape(out.shape[0], out.shape[1], out.shape[2], -1)

# example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X

def get_feat(X, p2,num_layers=3):
    output = p2.transform_singleHop(X,layer=0)
    if num_layers>1:
        for i in range(num_layers-1):
            output = p2.transform_singleHop(output, layer=i+1)
    return output

def find_model_size(p2, plus):
    num_param = 0
    
    # run get_feat to get K1, K2, and K3
    # get K1
    print('Getting the number of K1 features...')
    feats = get_feat(x_train_reduced, p2, num_layers=0)
    _, _, _, K1 = feats.shape
    print('Done.')

    # get K2
    print('Getting the number of K2 features...')
    feats = get_feat(x_train_reduced, p2, num_layers=2)
    _, _, _, K2 = feats.shape
    print('Done.')

    # get k3
    print('Getting the number of K3 features...')
    feats = get_feat(x_train_reduced, p2, num_layers=3)
    _, _, _, K3 = feats.shape
    print('Done.')
    
    # check if we're finding model size for PixelHop++
    if plus is True:
        num_param = (K1+K2+K3)*25
    else:
        num_param = (5*5*K1 + K1*5*5*K2 + K2*5*5*K3)
    
    return num_param

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # ---------- Load CIFAR-10 data and split ----------
    (x_train, y_train), (x_test,y_test) = keras.datasets.mnist.load_data()


    # -----------Data Preprocessing-----------
    """ x_train = np.asarray(x_train,dtype='float32')
    x_test = np.asarray(x_test,dtype='float32')
    y_train = np.asarray(y_train,dtype='int')
    y_test = np.asarray(y_test,dtype='int')
    y_train = np.squeeze(y_train, axis=-1)
    y_test = np.squeeze(y_test, axis=-1) """
    
    x_train = np.asarray(x_train,dtype='float32')[:,:,:,np.newaxis]
    x_test = np.asarray(x_test,dtype='float32')[:,:,:,np.newaxis]
    y_train = np.asarray(y_train,dtype='int')
    y_test = np.asarray(y_test,dtype='int')

    
    # if use only 10000 images train pixelhop
    x_train_reduced, y_train_reduced = select_balanced_subset(x_train, y_train, use_num_images=N_Train_Reduced)
    # x_train_reduced, y_train_reduced = x_train, y_train
    
    x_train_reduced /= 255.0
    x_train /= 255.0
    x_test /= 255.0


    # -----------Module 1: set PixelHop parameters-----------
    # Setup the Shrink Arguments
    shrinkArgs = [{'func':Shrink, 'win':5, 'stride':1, 'pad':2, 'pool':False},
                {'func':Shrink, 'win':5, 'stride':1, 'pad':0, 'pool':True},
                {'func':Shrink, 'win':5, 'stride':1, 'pad':0, 'pool':True}]
    # Setup the Saab Arguments for PixelHop++
    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'cw':True},
                {'num_AC_kernels':-1, 'needBias':True, 'cw':True},
                {'num_AC_kernels':-1, 'needBias':True, 'cw':True}]

    concatArg = {'func':Concat}

    thresh2 = 0.001
    thresh1 = 0.005
    # start_time = time.time()
    # -----------Module 1: Train PixelHop -----------
    # Construct PixelHop++ model
    p2_plus = Pixelhop(depth=3, TH1=thresh1, TH2=thresh2,
                SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)

    print('Training PixelHop++ model\n')
    p2_plus.fit(x_train_reduced)
    #pixel_hop_output = p2_plus.transform(x_train_reduced)
    #print(pixel_hop_output[0].shape, pixel_hop_output[1].shape, pixel_hop_output[2].shape)
    # model size
    # model_size_plus = find_model_size(p2=p2_plus, plus=True)
    # print(f'The model size of PixelHop++ is: {model_size_plus}')
    # model_size_base = find_model_size(p2=p2_base, plus=False)
    # print(f'The model size of PixelHop is: {model_size_base }')

    # --------- Module 2: get only Hop 3 feature for both training set and testing set -----------
    # you can get feature "batch wise" and concatenate them if your memory is restricted
    print('Running Module 2...')

    print('Getting the hop3 features for PixelHop++')
    train_hop3_feats_plus = get_feat(x_train_reduced, p2_plus)
    test_hop3_feats_plus = get_feat(x_test, p2_plus)

    # --------- Module 2: standardization

    STD_plus = np.std(train_hop3_feats_plus, axis=0, keepdims=1)
    train_hop3_feats_plus = train_hop3_feats_plus/STD_plus
    test_hop3_feats_plus = test_hop3_feats_plus/STD_plus

    print(train_hop3_feats_plus.shape)
    print(test_hop3_feats_plus.shape)

    # Reshape the features
    N_train_plus, _, _, X_train_plus = train_hop3_feats_plus.shape
    N_test_plus, _, _, X_test_plus = test_hop3_feats_plus.shape
    train_plus_reshaped = np.reshape(train_hop3_feats_plus, (N_train_plus, -1))
    test_plus_reshaped = np.reshape(test_hop3_feats_plus, (N_test_plus, -1))
    print(train_plus_reshaped.shape)
    print(test_plus_reshaped.shape)

    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("training time:", execution_time, "seconds")

    #---------- Module 3: Train XGBoost classifier on hop3 feature ---------
    # print('Running Module 3...')

    clf_plus = xgb.XGBClassifier(n_jobs=-1,
                        objective='multi:softprob',
                        # tree_method='gpu_hist', gpu_id=None,
                        max_depth=6,n_estimators=100,
                        min_child_weight=5,gamma=5,
                        subsample=1,learning_rate=0.2,
                        nthread=8,colsample_bytree=1.0)

    # # fit the classifier to the reshaped training data
    print('Fitting xgboost on PixelHop++\n')
    clf_plus.fit(train_plus_reshaped, y_train_reduced)
    
    # # use the model to make a prediction on the test set
    # print('Getting accuracy of PixelHop++\n')
    pred_test_plus = clf_plus.predict(test_plus_reshaped)

    # # get the accuracy score
    # acc_train_plus = accuracy_score(y_train_reduced, pred_train_plus)
    acc_test_plus = accuracy_score(y_test, pred_test_plus)
    print(f'The testing accuracy is : {acc_test_plus}')
    # print(f'The training accuracy is : {acc_train_plus}')


    # # print('Getting accuracy of PixelHop\n')
    # # pred_train_base = clf_base.predict(train_base_reshaped)
    # acc_train_base = accuracy_score(y_train_reduced, pred_train_base)
    # acc_test_base = accuracy_score(y_test, pred_test_base)
    # print(f'The testing accuracy of using is : {acc_test_base}')
    # print(f'The training accuracy is : {acc_train_base}')


    # Compare the performance of PixelHop and PixelHop++
    # TH1_list = np.array([0.001, 0.002, 0.005, 0.01])
    # plus_mnist_train_list = np.array([0.9768, 0.9651, 0.9158, 0.8626])
    # plus_mnist_test_list = np.array([0.9352, 0.9244, 0.8701, 0.8052])
    # plt.plot(TH1_list, plus_mnist_train_list)
    # plt.plot(TH1_list, plus_mnist_test_list)
    # basic_mnist_train_list = np.array([0.978, 0.9723, 0.9637, 0.9538])
    # basic_mnist_test_list = np.array([0.9275, 0.9248, 0.9132, 0.9032])
    # plt.plot(TH1_list, basic_mnist_train_list)
    # plt.plot(TH1_list, basic_mnist_test_list)
    # plt.legend(['PH++ Train', 'PH++ Test', 'PH Train', 'PH Test'])
    # plt.ylabel('Accuracy')
    # plt.xlabel('Threshold 2')
    # plt.title('TH2 vs MNIST Accuracy')
    # plt.show()

    # basic_fmnist_train_list = np.array([0.8957, 0.8726, 0.841, 0.7787])
    # basic_fmnist_test_list = np.array([0.8147, 0.8056, 0.7728, 0.7299])
    # basic_fmnist_train_list = np.array([0.9024, 0.8871, 0.84, 0.8374])
    # basic_fmnist_test_list = np.array([0.8216, 0.8152, 0.7802, 0.7818])
    # plt.plot(TH1_list, basic_fmnist_train_list)
    # plt.plot(TH1_list, basic_fmnist_test_list)
    # plt.plot(TH1_list, basic_fmnist_train_list)
    # plt.plot(TH1_list, basic_fmnist_test_list)
    # plt.legend(['PH++ Train', 'PH++ Test', 'PH Train', 'PH Test'])
    # plt.ylabel('Accuracy')
    # plt.xlabel('Threshold 1')
    # plt.title('TH1 vs Fashion-MNIST Accuracy')
    # plt.show()

"""     plt.rcParams["figure.figsize"] = (20,8)

    # labels for fashion-mnist
    fashion_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    # labels for mnist
    # mnist_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    cm_plus = confusion_matrix(y_test, pred_test_plus)
    disp_plus = ConfusionMatrixDisplay(confusion_matrix=cm_plus, display_labels=fashion_labels)
    disp_plus.plot()
    plt.show() """
    
    
    
    # h1 [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    # mnist train [0.9786, 0.9768, 0.9733, 0.9635, 0.9427, 0.9252]
    # mnist test [0.9398, 0.9352, 0.9309, 0.9158, 0.8893, 0.8689]
    # fm train [0.9019, 0.8957, 0.8837, 0.8673, 0.8498, 0.8232]
    # fm test [0.8179, 0.8147, 0.8015, 0.788, 0.7722, 0.7568]

    # h2 [0.001, 0.002, 0.005, 0.01]

    
    
    
    
