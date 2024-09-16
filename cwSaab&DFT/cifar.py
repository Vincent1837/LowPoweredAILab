from cwSaab import cwSaab
from feat_utils import feature_selection
from tensorflow import keras
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from skimage import transform
from skimage.util import view_as_windows

# Load MNIST and CIFAR-10 datasets
#(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = keras.datasets.mnist.load_data()
(cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = keras.datasets.cifar10.load_data()

def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs

# Resize MNIST images
#mnist_train_images = resize_batch(mnist_train_images)
#mnist_test_images = resize_batch(mnist_test_images)

# Callback functions for cwSaab
def Shrink(X, shrinkArg):
    win = shrinkArg['win']
    X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

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

def invShrink(X, invshrinkArg):
    win = invshrinkArg['win']
    S = X.shape
    X = X.reshape(S[0], S[1], S[2], -1, 1, win, win, 1)
    X = np.moveaxis(X, 5, 2)
    X = np.moveaxis(X, 6, 4)
    return X.reshape(S[0], win*S[1], win*S[2], -1)

def Concat(X, concatArg):
    return X

if __name__ == "__main__":
    # Test example using cwSaab
    X = cifar_train_images
    #X = mnist_train_images
    print("Input feature shape: %s" % str(X.shape))

    # Set arguments for cwSaab
    SaabArgs = [{'num_AC_kernels': -1, 'needBias': True, 'useDC': False, 'batch': None}]
    shrinkArgs = [{'func': Shrink, 'win': 2}]
    inv_shrinkArgs = [{'func': invShrink, 'win': 2}]
    concatArg = {'func': Concat}
    inv_concatArg = {'func': Concat}
    kernelRetainArg = {'Layer0': -1, 'Layer1': -1, 'Layer2': -1}
    
    shrinkArgs = [{'func':Shrink, 'win':5, 'stride':1, 'pad':2, 'pool':False},
                {'func':Shrink, 'win':5, 'stride':1, 'pad':0, 'pool':True},
                {'func':Shrink, 'win':5, 'stride':1, 'pad':0, 'pool':True}]
    # Setup the Saab Arguments for PixelHop++
    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'cw':False},
                {'num_AC_kernels':-1, 'needBias':True, 'cw':True},
                {'num_AC_kernels':-1, 'needBias':True, 'cw':True}]

    # Initialize and fit cwSaab
    print("Testing cwSaab with depth=1")
    cwsaab = cwSaab(depth=1, energyTH=0.5, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg, splitMode=0, cwHop1=False)
    output = cwsaab.fit(X)
    output = cwsaab.transform(X)

    # Perform inverse transform and check for errors
    Y = cwsaab.inverse_transform(output, inv_concatArg=inv_concatArg, inv_shrinkArgs=inv_shrinkArgs)
    Y = np.round(Y)
    assert (np.mean(np.abs(X - Y)) < 1), "invcwSaab error!"

    # Feature selection
    features = output[0].reshape(len(X), -1)
    labels = cifar_train_labels
    #labels = mnist_train_labels
    selected, dft_loss = feature_selection(features, labels, FStype='DFT_entropy', thrs=0.7, B=16)
    print("Selected features:", selected)

    # Prepare data for XGBoost
    X_train = cifar_train_images.reshape(len(cifar_train_images), -1)[:, selected]
    y_train = cifar_train_labels
    X_test = cifar_test_images.reshape(len(cifar_test_images), -1)[:, selected]
    y_test = cifar_test_labels
    print(f"X_train.shape: {X_train.shape}")

    # Create and train the XGBClassifier
    model = XGBClassifier(
        booster='gbtree',
        objective='multi:softprob',  # multi-class classification
        n_estimators=100, # number of estimators
        num_class=10,  # number of classes
        eta=0.1,  # learning rate
        max_depth=6,  # maximum depth of the trees
        eval_metric='mlogloss',  # evaluation metric
        use_label_encoder=False  # to suppress a warning
    )
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f"MNIST Accuracy: {accuracy * 100:.2f}%")