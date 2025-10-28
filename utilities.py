import copy
import math
import numpy as np
import time
import shutil

terminal_width = shutil.get_terminal_size().columns

#DIY csv_reader function
def csv_reader(file):
    with open(file,'r') as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        cleaned_lines = line.strip()
        fields = cleaned_lines.split(',')
        for i in range(len(fields)):
            fields[i] = convert_value(fields[i])
        rows.append(fields)
    return rows

#Defining a helper function for datatype conversion
def convert_value(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

#Defining a function to find the unique values(categories) of a feature and add them to a list
def unique_value_identifier(dataset,feature):
    the_set = set()
    for i in range(len(dataset)):
        the_set.add(dataset[i][feature])
    the_set = set(the_set)
    return the_set

#Defining a function for binary encoding (one-hot encoding)
def bin_encoding(dataset, features, feature_index, header, base, feature_name):
    features_modified = features.copy()
    features_modified.remove(base)
    modified_dataset = copy.deepcopy(dataset)
    for category in features_modified:
        header.append(f"{feature_name}_{category}")
        for i in range(len(dataset)):
            if modified_dataset[i][feature_index] == category:
                modified_dataset[i].append(1)
            else:
                modified_dataset[i].append(0)
    for j in range(len(dataset)):
        del modified_dataset[j][feature_index]
    if feature_name in header:
        header.remove(feature_name)
    return modified_dataset, header

#Defining a function for cyclic encoding (sin/cos encoding)
def cyclic_encoding (dataset, feature_index, feature, headers, parser, no_of_values):
    if feature in headers:
        headers.remove(feature)
    headers.append(f"{feature}_sin")
    headers.append(f"{feature}_cos")
    for i in range(len(dataset)):
        target = parser(dataset[i][feature_index])
        angle = 2*math.pi*(target-1)/(no_of_values)
        dataset[i].append(math.sin(angle))
        dataset[i].append(math.cos(angle))
        dataset[i].remove(dataset[i][feature_index])
    return dataset, headers

#Defining a helper function to parse the date
def get_month(date):
    cleaned_date = date.split('-')
    month = cleaned_date[1]
    month = convert_value(month)
    return month

#Defining a function to split the dataset into train/test (80/20)
def split_data(features, targets):
    n_samples = len(targets)
    n_train = round(0.8*n_samples)
    X_train = features[:n_train]
    y_train = targets[:n_train]
    X_test = features[n_train:]
    y_test = targets[n_train:]
    return X_train, y_train, X_test, y_test

#Defining the z-score normalizer
def z_score_normalizer (features, scaling_features_indices):
    scaled_features = copy.deepcopy(features)
    m,n = scaled_features.shape
    mean = np.zeros((n,))
    st_deviation = np.zeros((n,))
    for i in scaling_features_indices:
        mean[i] = np.mean(scaled_features[:,i])
        st_deviation[i] = np.sqrt(np.mean((scaled_features[:,i]-mean[i])**2))
        if st_deviation[i] != 0:
            scaled_features[:,i] = (scaled_features[:,i] - mean[i])/st_deviation[i]
    return scaled_features, mean, st_deviation

#Defining a function to normalize test data
def normalize_test_data(features, mean, st_deviation, scaling_feature_indices):
    scaled_features = np.copy(features)
    m,n = scaled_features.shape
    for i in scaling_feature_indices:
        if st_deviation[i] != 0:
            scaled_features[:,i] = (scaled_features[:,i] - mean[i])/st_deviation[i]
    return scaled_features

#Defining the vectorized cost function
def cost(w,b,features,targets):
    m,n = features.shape
    cost = ((np.dot(features,w) + b) - targets)**2
    cost = np.mean(cost)/2
    return cost

#Defining the Ridge regularized cost function
def reg_cost (w, b, features, targets, lambda_):
    raw_cost = cost(w, b, features, targets)
    m = len(targets)
    reg_term = (lambda_/(2*m))*(np.sum(w**2))
    regularized_cost = raw_cost + reg_term
    return regularized_cost

#Defining a vectorized helper function to calculate gradient
def compute_gradient (w, b, features, targets):
    m,n = features.shape
    error = (np.dot(features,w)+b)-targets
    dj_dw = np.dot(features.T, error)/m
    dj_db = np.mean(error)
    return dj_dw, dj_db


#Implementing the gradient descent algorithm with J_history and terminal outputs
def grad_desc (features, targets, w_in, b_in, alpha, iters):
    w = w_in.copy()
    b = b_in
    J_history = []
    for i in range(iters):
        dj_dw, dj_db = compute_gradient(w,b,features,targets)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        J_history.append(cost(w,b,features,targets))
        if i % (iters//10) == 0 or i == iters -1:
            print(f"Iteration : \033[32m{i}\033[0m    Cost: \033[32m{J_history[-1]}\033[0m")
        if i > 0 and abs(J_history[-1] - J_history[-2])<1e-8:
            print("\033[32mConvergence Achieved\033[0m")
            break
    return w,b, J_history

#Implementing gradient descent with L2 (Ridge) regularization
def reg_grad_desc (features, targets, w_in, b_in, alpha, iters, lambda_):
    w = w_in.copy()
    b = b_in
    m = len(targets)
    J_history = []
    width = len(str(iters))
    initial_cost = reg_cost(w_in,b_in,features,targets,lambda_)
    start_time = time.time()
    training_title = " Optimization Log "
    print(f"\n{training_title.center(terminal_width,'-')}\n")
    for i in range(iters):
        dj_dw, dj_db = compute_gradient(w,b,features,targets)
        #Apply Ridge regularization to weight updates
        w = w - alpha*(dj_dw + (lambda_/m)*w)
        b = b - alpha*dj_db
        J_history.append(reg_cost(w,b,features,targets, lambda_))
        if i % (iters//10) == 0 or i == iters -1:
            print(f"Iteration : \033[32m{i:{width}}\033[0m    Cost: \033[32m{J_history[-1]:.4f}\033[0m")
        if i > 0 and abs(J_history[-1] - J_history[-2])<1e-8:
            print("\033[32mConvergence Achieved\033[0m")
            break
    end_time = time.time()
    w = np.array(w)
    b = np.array(b)
    J_history = np.array(J_history)
    summary_title = " Training Summary "
    print(f"\n{summary_title.center(terminal_width, '-')}\n")
    print(f"Learned parameters\n \nw : \033[32m{np.array2string(w, precision=4, separator=', ')}\033[0m    \nb : \033[32m{np.array2string(b, precision=4, separator=', ')}\033[0m\n")
    print(f"Cost before Gradient Descent    : {initial_cost:.4f}\n")
    print(f"Cost after Gradient Descent     : {reg_cost(w,b,features,targets,lambda_):.4f}\n")
    print(f"Training completed in           : {(end_time - start_time):.4f} seconds.\n")
    return w,b, J_history

#Defining the MSE function to evaluate model predictions 
def mse_eval (test_features, w, b, test_targets):
    y_pred = np.dot(test_features, w) + b
    mse = np.mean((y_pred - test_targets)**2)
    return mse

#Defining the RMSE function to evaluate model predictions 
def rmse_eval (test_features, w, b, test_targets):
    mse = mse_eval(test_features,w,b,test_targets)
    rmse = mse**(0.5)
    return rmse

#Defining the MAE function to evaluate model predictions 
def mae_eval (test_features, w, b, test_targets):
    y_pred = np.dot(test_features,w) + b
    mae = np.mean(np.abs(y_pred - test_targets))
    return mae

#Defining the Coefficient of Determination function to evaluate model predictions 
def r_2 (test_features, w, b,test_targets):
    y_pred = np.dot(test_features,w) + b
    y_mean = np.mean(test_targets)
    rss = np.sum((y_pred - test_targets)**2)
    tss = np.sum((test_targets - y_mean)**2)
    r_squared = 1 - ((rss)/(tss))
    return r_squared