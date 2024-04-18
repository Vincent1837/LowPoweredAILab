from skimage.util import view_as_windows
from sklearn import datasets
from saab import Saab
import numpy as np

digits = datasets.load_digits()
X = digits.images.reshape((len(digits.images), 8, 8, 1))
X = X[0:1]


def Shrink(X, shrinkArg):
        win = shrinkArg['win']
        X = view_as_windows(X, (1, win, win, 1), (1, win, win, 1))
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
    
shrinkArgs = {'func':Shrink, 'win':2, 'stride': 2}

print(f'X.shape = {X.shape}')
print('X = ')
print(X.reshape(8, 8))

Shrunk = Shrink(X, shrinkArgs)
S = list(Shrunk.shape)
Shrunk = Shrunk.reshape(-1, S[-1])

print(f'Reshaped Shrunk.shape = {Shrunk.shape}')
print('Reshaped Shrunk X = ')
print(Shrunk)

print(f'Input feature shape:    {Shrunk.shape}')
saab = Saab()
saab.fit(X)

Xt = saab.transform(X)
print(Xt)