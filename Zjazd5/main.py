import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# import pandas as pd

# Wczytaj dane z pliku
data_array_seeds = np.loadtxt('seeds_dataset.txt')

X = data_array_seeds[:, :-1]
y = data_array_seeds[:, -1]

# Uzupełnij brakujące wartości
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Podziel dane na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standaryzuj dane (opcjonalne, ale zalecane dla sieci neuronowych)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Zbuduj model sieci neuronowej
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Trenuj model
model.fit(X_train, y_train)

# Ocen skuteczność modelu na zestawie testowym
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

p=unpickle("./cifar-10-batches-py/test_batch")
print(p[b'labels'])