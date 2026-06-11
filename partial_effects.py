import numpy as np 
import matplotlib.pyplot as plt

from preprocessing_data import preprocess

model = np.load("model_parameters.npz")

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

feature_idx = 1
feature_name = headers[feature_idx]

print(f"Observing partial effect of : {feature_name}")

x_values = np.linspace(
    X[:, feature_idx].min(),
    X[:, feature_idx].max(),
    200
)

x_values_normied = (x_values - mean[feature_idx])/std_deviation[feature_idx]

X_partial = np.zeros((len(x_values), X.shape[1]))
X_partial[:, feature_idx] = x_values_normied

y_partial = predict(X_partial, w_learned, b_learned)

plt.scatter(X[:, feature_idx], y,alpha=0.5)
plt.plot(x_values, y_partial)

plt.xlabel(feature_name)
plt.ylabel("Salary")
plt.title(f"Patial Effect of {feature_name} on Salary")

plt.show()