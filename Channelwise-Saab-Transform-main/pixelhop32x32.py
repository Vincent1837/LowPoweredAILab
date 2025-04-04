from tensorflow import keras
import numpy as np
from skimage import transform

# loading mnist and cifar10 datasets

(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labelss) = keras.datasets.mnist.load_data()
(cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = keras.datasets.cifar10.load_data()

# padding mnist images from 28x28 to 32x32
def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs

print("---------------resizing-------------------")
mnist_train_images = resize_batch(mnist_train_images)
mnist_test_images = resize_batch(mnist_test_images)
print(f"Shape of training images: {mnist_train_images.shape}")
print(f"Number of training labels: {len(mnist_train_labels)}")
print(f"Shape of testing images: {mnist_test_images.shape}")
print(f"Number of testing labels: {len(mnist_test_labelss)}")
print("-----------resize completed---------------")

print(f"Shape of training images: {cifar_train_images.shape}")
print(f"Number of training labels: {len(cifar_train_labels)}")
print(f"Shape of test images: {cifar_test_images.shape}")
print(f"Number of test labels: {len(cifar_test_labels)}")

# v 2021.04.12
# PixelHop and PixelHop++ (Module 1)
# modified from https://github.com/ChengyaoWang/PixelHop-_c-wSaab/blob/master/pixelhop2.py

from cwSaab import cwSaab
import pickle

class Pixelhop(cwSaab):
    def __init__(self, depth=1, TH1=0.005, TH2=0.001, SaabArgs=None, shrinkArgs=None, concatArg=None, load=False):
        super().__init__(depth=depth, TH1=TH1, TH2=TH2, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, load=load)
        self.TH1 = TH1
        self.TH2 = TH2
        self.idx = []        
        self.concatArg = concatArg

    def fit(self, X):
        super().fit(X)
        return self

    def transform(self, X):
        X = super().transform(X)
        return self.concatArg['func'](X, self.concatArg)

    def transform_singleHop(self, X, layer=0):
        X = super().transform_singleHop(X, layer=layer)
        return X

    '''Methods for Saving & Loading'''
    def save(self, filename: str):
        assert (self.trained == True), "Need to Train First"
        pixelhop_model = {}
        pixelhop_model['par'] = self.par
        pixelhop_model['bias'] = self.bias
        pixelhop_model['depth'] = self.depth
        pixelhop_model['energy'] = self.Energy
        pixelhop_model['SaabArgs'] = self.SaabArgs
        pixelhop_model['shrinkArgs'] = self.shrinkArgs
        pixelhop_model['concatArgs'] = self.concatArg
        pixelhop_model['TH1'] = self.TH1
        pixelhop_model['TH2'] = self.TH2

        with open(filename + '.pkl','wb') as f:
            pickle.dump(pixelhop_model, f)
        return

    def load(self, filename: str):
        pixelhop_model = pickle.load(open(filename + '.pkl','rb'))
        self.par = pixelhop_model['par']
        self.bias = pixelhop_model['bias']
        self.depth = pixelhop_model['depth']
        self.Energy = pixelhop_model['energy']
        self.SaabArgs = pixelhop_model['SaabArgs']
        self.shrinkArgs = pixelhop_model['shrinkArgs']
        self.concatArg = pixelhop_model['concatArgs']
        self.trained = True
        self.TH1 = pixelhop_model['TH1']
        self.TH2 = pixelhop_model['TH2']
        
        return self

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn import datasets
    from skimage.util import view_as_windows

    # example callback function for collecting patches and its inverse
    def Shrink(X, shrinkArg):
        win = shrinkArg['win']
        X = view_as_windows(X, (1, win, win, 1), (1, win, win, 1))
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    # example callback function for how to concate features from different hops
    def Concat(X, concatArg):
        return X

    # read data
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), 8, 8, 1))
    #X = mnist_test_images
    print(" input feature shape: %s"%str(X.shape))

    # set args
    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'cw': False},
                {'num_AC_kernels':-1, 'needBias':True, 'cw':True},
                {'num_AC_kernels':-1, 'needBias':True, 'cw':True}] 
    shrinkArgs = [{'func':Shrink, 'win':2, 'stride': 2}, 
                {'func': Shrink, 'win':2, 'stride': 2},
                {'func': Shrink, 'win':2, 'stride': 2}]
    concatArg = {'func':Concat}

    print(" --> test inv")
    print(" -----> depth=2")
    p2 = Pixelhop(depth=2, TH1=0.005, TH2=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    p2.fit(X)
    output1 = p2.transform(X)
    output2 = p2.transform_singleHop(X)
    
    with open('output.pkl', "wb") as file:
        pickle.dump(output1, file)

    """ '''Test for Save / Load'''
    p2.save('./dummy')
    p2_new = Pixelhop(load=True).load('./dummy')
    output1_new = p2_new.transform(X)
    output2_new = p2_new.transform_singleHop(X) """

    print("------- DONE -------\n")
    