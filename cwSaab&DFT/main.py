from cwSaab import cwSaab
from feat_utils import feature_selection
from tensorflow import keras
import numpy as np

(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = keras.datasets.mnist.load_data()
(cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = keras.datasets.cifar10.load_data()

def resize_batch(imgs):
    from skimage import transform
    # A function to resize a batch of MNIST images to (32, 32)
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs

mnist_train_images = resize_batch(mnist_train_images)
mnist_test_images = resize_batch(mnist_test_images)

if __name__ == "__main__":
    print("------- cwSaab -----\n")
    # example useage
    from sklearn import datasets
    from skimage.util import view_as_windows

    # example callback function for collecting patches and its inverse
    def Shrink(X, shrinkArg):
        win = shrinkArg['win']
        X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
        print("X = view_as_windows(X, (1,win,win,1), (1,win,win,1))")
        print(X.shape)
        print("##########################")
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    def invShrink(X, invshrinkArg):
        win = invshrinkArg['win']
        S = X.shape
        X = X.reshape(S[0], S[1], S[2], -1, 1, win, win, 1)
        X = np.moveaxis(X, 5, 2)
        X = np.moveaxis(X, 6, 4)
        return X.reshape(S[0], win*S[1], win*S[2], -1)

    # example callback function for how to concate features from different hops
    def Concat(X, concatArg):
        return X

    # read data
    import cv2
    print(" > This is a test example: ")
    #digits = datasets.load_digits()
    #X = digits.images.reshape((len(digits.images), 8, 8, 1))
    X = mnist_test_images
    print(" input feature shape: %s"%str(X.shape))

    # set args
    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':False, 'batch':None}, 
                {'num_AC_kernels':2, 'needBias':True, 'useDC':False, 'batch':None}]
    shrinkArgs = [{'func':Shrink, 'win':2}, 
                {'func': Shrink, 'win':2},
                {'func': Shrink, 'win':2}]
    inv_shrinkArgs = [{'func':invShrink, 'win':2}, 
                    {'func': invShrink, 'win':2},
                    {'func': invShrink, 'win':2}]
    concatArg = {'func':Concat}
    inv_concatArg = {'func':Concat}

    kernelRetainArg = {'Layer0':-1, 'Layer1':-1, 'Layer2':-1}

    print(" --> test inv")
    
    """ print(" -----> depth=1")
    cwsaab = cwSaab(depth=1, energyTH=0.1, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg, kernelRetainArg=kernelRetainArg)#depth=1
    output = cwsaab.fit(X)
    output = cwsaab.transform(X)
    Y = cwsaab.inverse_transform(output, inv_concatArg=inv_concatArg, inv_shrinkArgs=inv_shrinkArgs)
    Y = np.round(Y)
    assert (np.mean(np.abs(X-Y)) < 1e-5), "invcwSaab error!" """
    
    print(" -----> depth=2")
    cwsaab = cwSaab(depth=2, energyTH=0.5, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg, splitMode=0, cwHop1=True)
    output = cwsaab.fit(X)
    output = cwsaab.transform(X)
    Y = cwsaab.inverse_transform(output, inv_concatArg=inv_concatArg, inv_shrinkArgs=inv_shrinkArgs)
    Y = np.round(Y)
    assert (np.mean(np.abs(X-Y)) < 1), "invcwSaab error!"
    print(output[0].shape, output[1].shape) # (1797, 4, 4, 4) (1797, 2, 2, 4)
    print("------- DONE -------\n")
    print("------- DFT  -------\n")
    
    features = output[1].reshape(len(X), -1)
    labels = mnist_test_labels
    
    selected, dft_loss = feature_selection(features, labels, FStype='DFT_entropy', thrs=0.5, B=16)

    print(selected)