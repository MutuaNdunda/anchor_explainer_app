# save_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X, y = data.data, data.target

clf = RandomForestClassifier().fit(X, y)

with open('models/model.pkl', 'wb') as f:
    pickle.dump((clf, data), f)
