import numpy as np 
import matplotlib as plt

from preprocessing_data import preprocess

model = np.load("model_parameters.npz", allow_pickle=True)

w_learned = model["weights"]
b_learned = model["bias"]

mean = model["mean"]
std_deviation = model["standard_deviation"]

headers = model["headers"]
continuous_idx = model["continuous_idx"]

# print("Weights shape:", w_learned.shape)
# print("Bias:", b_learned)
# print("Mean shape:", mean.shape)
# print("Standard deviation shape:", std_deviation.shape)
# print("Headers:", headers)
# print("Continuous feature indices:", continuous_idx)

features, targets, _ = preprocess()

X = np.array(features)
y = np.array(targets)

# print("Feature matrix shape:", X.shape)
# print("Target vector shape:", y.shape)
# print("Number of weights:", len(w_learned))

if X.shape[1] != len(w_learned):
    raise ValueError(
        f"Feature count mismatch: X has {X.shape[1]} columns, "
        f"but model has {len(w_learned)} weights."
    )

def predict(X, w, b):
    return X @ w + b

prediction = predict(X, w_learned, b_learned)
error = np.absolute(y - prediction)

print(error)