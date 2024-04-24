import pickle
from sklearn import datasets

digits = datasets.load_digits()
X = digits.images.reshape((len(digits.images), 8, 8, 1))

with open('output.pkl', "rb") as file:
    output1 = pickle.load(file)
    
hop1 = output1[0]
hop2 = output1[1]

print(f'X.shape = {X.shape}')
print(f'hop1.shape = {hop1.shape}')
print(f'hop2.shape = {hop2.shape}')
print(hop1[0])
