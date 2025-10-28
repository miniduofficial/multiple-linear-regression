import numpy as np
import matplotlib.pyplot as plt
import random
from preprocessing_data import preprocess
from utilities import *

#Load and unpack features, targets, and headers
features,targets,headers = preprocess()

#Shuffle the dataset to randomize sample order
data = list(zip(features, targets))
random.shuffle(data)
features_shuffled, targets_shuffled = zip(*data)

features_array = np.array(features_shuffled)
targets_array = np.array(targets_shuffled)
headers_array = np.array(headers)

# Visualize each feature against the target (optional)

show_plot = input("Show feature-target scatter plots? (Y/n) :\n").strip().lower()
if show_plot in ("y", ""):
    for i in range(features_array.shape[1]):
        plt.figure(figsize=(6, 4))
        plt.scatter(features_array[:, i], targets_array, alpha=0.6)
        plt.xlabel(headers_array[i])
        plt.ylabel("Salary")
        plt.title(f"{headers_array[i]} vs Salary")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

#Split the dataset into training (80%) and testing (20%)
X_train, y_train, X_test, y_test = split_data(features_array, targets_array)

# Normalize training features using z-score normalization  
# Normalize test features using the training mean and std
X_train_normalized, mean, standard_deviation = z_score_normalizer(X_train,[0,1,2])
X_test_normalized = normalize_test_data(X_test,mean,standard_deviation,[0,1,2])

#Train model and run evals pre training 
w = np.zeros((X_train.shape[1],))
b = 0
mse1 = mse_eval(X_test_normalized, w, b, y_test)
rmse1 = rmse_eval(X_test_normalized, w, b, y_test)
mae1 = mae_eval(X_test_normalized, w, b, y_test)
r_squared1 = r_2(X_test_normalized, w, b, y_test)

w_learned, b_learned, J_history = reg_grad_desc(X_train_normalized, y_train, w, b, 5e-6, 1000000, 10)

#Save weights and biases for future use
np.savez("model_parameters.npz", weights= w_learned, bias= b_learned)

#Run evals post training 
mse2 = mse_eval(X_test_normalized, w_learned, b_learned,y_test)
rmse2 =rmse_eval(X_test_normalized, w_learned, b_learned, y_test)
mae2 = mae_eval(X_test_normalized, w_learned, b_learned, y_test)
r_squared2 = r_2(X_test_normalized, w_learned, b_learned, y_test)

#Compare pre vs post training evals
print(f"MSE before training :\033[32m{mse1}\033[0m, MSE after training :\033[32m{mse2}\033[0m")
print(f"RMSE before training :\033[32m{rmse1}\033[0m, RMSE after training :\033[32m{rmse2}\033[0m")
print(f"MAE before training :\033[32m{mae1}\033[0m, MAE after training :\033[32m{mae2}\033[0m")
print(f"R**2 before training :\033[32m{r_squared1}\033[0m, R**2 after training :\033[32m{r_squared2}\033[0m")

#Plot the learning curve 