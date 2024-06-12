---
marp: true
theme: uncover
---

# cwSaab to DFT

---

## dataset
##### Mnist test dataset (10000, 28, 28, 1) -> (10000, 32, 32, 1)

```python
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
```

---

## cwSaab
###### depth=2, energyTH=0.5, splitMode=0, cwHop1=True
```python
# set args
SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':False,'batch':None}, 
            {'num_AC_kernels':2, 'needBias':True, 'useDC':False, 'batch':None}]
shrinkArgs = [{'func':Shrink, 'win':2}, 
              {'func': Shrink, 'win':2},
              {'func': Shrink, 'win':2}]

cwsaab = cwSaab(depth=2, energyTH=0.5, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg, splitMode=0, cwHop1=True)
output = cwsaab.fit(X)
output = cwsaab.transform(X)
```

```
output[0].shape, output[1].shape = (10000, 16, 16,4), (10000, 8, 8, 2)
```

---

## DFT
```python
selected, dft_loss = feature_selection(features, labels, FStype='DFT_entropy', thrs=0.5, B=16)
print(selected)
```

```
100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:02<00:00, 47.31it/s] 
[ 52  68  84  90  71  55 100  72  24  70  73  74  57  53  75  76  86  89
  37  22  56  58  83  54  38  42  59  77  93  60 102  92 118  82   8  36
  85  67  40 101 120  44  66  23  69   9  21  20 107 121 103   6  26  61
  45 104  98   7  91 119  25  47  87  51]
```

---

## the version of cwSaab

Last time we discussed about the issue of the parameter 'num_of_kernels' not being used. This time I used the other version of cwSaab to perform the task, but the structure of the code is slightly different from the previous version, as we can see from the parameter splitMode=0, cwHop1=true etc. 
