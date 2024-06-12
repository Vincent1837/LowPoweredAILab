from cwSaab import cwSaab
from feat_utils import feature_selection
from tensorflow import keras
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from skimage import transform
from skimage.util import view_as_windows

# Load MNIST and CIFAR-10 datasets
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = keras.datasets.mnist.load_data()
(cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = keras.datasets.cifar10.load_data()

def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs

# Resize MNIST images
mnist_train_images = resize_batch(mnist_train_images)
mnist_test_images = resize_batch(mnist_test_images)

# Callback functions for cwSaab
def Shrink(X, shrinkArg):
    win = shrinkArg['win']
    X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

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
    X = mnist_train_images
    print("Input feature shape: %s" % str(X.shape))

    # Set arguments for cwSaab
    SaabArgs = [{'num_AC_kernels': -1, 'needBias': False, 'useDC': False, 'batch': None}, 
                {'num_AC_kernels': 2, 'needBias': True, 'useDC': False, 'batch': None}]
    shrinkArgs = [{'func': Shrink, 'win': 2}, 
                  {'func': Shrink, 'win': 2},
                  {'func': Shrink, 'win': 2}]
    inv_shrinkArgs = [{'func': invShrink, 'win': 2}, 
                      {'func': invShrink, 'win': 2},
                      {'func': invShrink, 'win': 2}]
    concatArg = {'func': Concat}
    inv_concatArg = {'func': Concat}
    kernelRetainArg = {'Layer0': -1, 'Layer1': -1, 'Layer2': -1}

    # Initialize and fit cwSaab
    print("Testing cwSaab with depth=2")
    cwsaab = cwSaab(depth=2, energyTH=0.5, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg, splitMode=0, cwHop1=True)
    output = cwsaab.fit(X)
    output = cwsaab.transform(X)

    # Perform inverse transform and check for errors
    Y = cwsaab.inverse_transform(output, inv_concatArg=inv_concatArg, inv_shrinkArgs=inv_shrinkArgs)
    Y = np.round(Y)
    assert (np.mean(np.abs(X - Y)) < 1), "invcwSaab error!"

    # Feature selection
    features = output[1].reshape(len(X), -1)
    labels = mnist_train_labels
    selected, dft_loss = feature_selection(features, labels, FStype='DFT_entropy', thrs=0.8, B=16)
    print("Selected features:", selected)

    # Prepare data for XGBoost
    X_train = mnist_train_images.reshape(len(mnist_train_images), -1)[:, selected]
    y_train = mnist_train_labels
    X_test = mnist_test_images.reshape(len(mnist_test_images), -1)[:, selected]
    y_test = mnist_test_labels
    print(f"X_train.shape: {X_train.shape}")

    # Create and train the XGBClassifier
    model = xgb.XGBClassifier(
        booster='gbtree',
        objective='multi:softprob',  # multi-class classification
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
