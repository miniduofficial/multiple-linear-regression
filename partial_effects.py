import numpy as np 
import matplotlib.pyplot as plt

from preprocessing_data import preprocess

#Comment out the stylistic tweaks if it isn't to your liking
plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#181C14',
    'figure.facecolor': 'none',
    'savefig.facecolor': 'none',
    'axes.edgecolor': '#697565',
    'axes.labelcolor': '#CFC4B3',
    'xtick.color': '#CFC4B3',
    'ytick.color': '#CFC4B3',
    'text.color': '#ECDFCC',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'font.family': 'serif'
})

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
        f