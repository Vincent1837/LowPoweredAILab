---
marp: true
---

# Cifar-10

---

## 20240815

### Input shape: (50000, 32, 32, 3)

```
Running Module 1...
Training PixelHop++ model

Input shape: (50000, 32, 32, 3)
=============================================>c/w Saab Train Hop 1
=============================================>c/w Saab Train Hop 1
=============================================>c/w Saab Train Hop 2
=============================================>c/w Saab Train Hop 3
Running Module 2...
Getting the hop3 features for PixelHop++
100%|████████████████████████████████████████████████| 848/848
Running Module 3...
Fitting xgboost on PixelHop++

The testing accuracy is : 0.1991
```

---

### Input shape: (10000, 32, 32, 3) balanced

```
Running Module 1...
Training PixelHop++ model

Input shape: (10000, 32, 32, 3)
=============================================>c/w Saab Train Hop 1
=============================================>c/w Saab Train Hop 2
=============================================>c/w Saab Train Hop 3
Running Module 2...
Getting the hop3 features for PixelHop++
Selecting Features from training set...
Training set shape: (10000, 848)
100%|████████████████████████████████████████████████| 848/848
Running Module 3...
Fitting xgboost on PixelHop++

The testing accuracy is : 0.4566
```