import csv
import matplotlib.pyplot as plt
import random
import numpy as np

with open('Factory_Salary.csv','r') as f: 
    #Opens file in read mode. 
    #The 'with' statement ensures the file is automatically closed when you're done
    #f is the file object (like a handle or a pointer to the open file)
    reader = csv.reader(f) 
    #Creates a CSV reader object from the file
    #This object lets you iterate over each row in the CSV file
    #Each row will be returned as a list of strings
    headers = next(reader)
    #Reads the first row from the CSV file (the header row)
    #Stores in a varable called headers
    #Advances the reades to the next line--so the next loop will read actual data
    data = [row for row in reader]
    #Uses a list comprehension to collect all the remaining rows
    #Each row is a list of strings
    #You now have a data list with each row of the CSV

#DIY csv_reader function

def csv_reader(file):
#Defining a function to read the dataset in the CSV file and convert it into a list of lists
    with open(file,'r') as f:
    #The 'with' statement ensures that the file is closed after reading and "open(file,'r') as f" opens the file in read mode with the f handle
        lines = f.readlines()
        #Creates a list of all all the training examples including the header row using the readlines method for the f object
    rows = []
    #Creates and empty list to which we will add the parsed training examples 
    for line in lines:
    #Loop over all the rows in the dataset. Loop over the extries in the lines list to be more specific
        cleaned_lines = lines.strip()
        #Using the strip method to remove blank spaces, tabs and new line characters
        fields = cleaned_lines.split(',')
        #Creates a list using each element of the lines list where the features of each training example are split at a ','
        rows.append(fields)
        #Appends the previously created empty list with each cleaned up and parsed training example
    return rows
    #Returns the rows list


#Defining a helper function for datatype conversions

def convert_value(value):
    try:
        return int(value)
    except ValueError:
    #We specify the error so that we catch the exact errors we expect
        try:
            return float(value)
        except ValueError:
            return value
        

#Plotting Salary vs features[i]
def salary_v_feature(features_array, targets,headers):
    for i in range(features_array.shape[1]): #features_array.shape[1] corresponds to the number of features in a training example
        plt.figure(figsize=(6,4)) #This sets the size of the canvas. It starts a new figure for each plot with a width of 6 inches and a height of 4 inches
        plt.scatter(features_array[:,i], targets, alpha=0.6) #This creates a scatter plot of--x-axis: the i-th feature and the y-axis: the target values. alpha=0.6 makes the points slightly transparent 
        plt.xlabel(headers[i])
        plt.ylabel("Salary")
        plt.title(f"{headers[i]} vs Salary")
        plt.grid(True) #Adds grid lines for easier visual analysis
        plt.tight_layout() #Prevents overlap of labels/titles
        plt.show()

#Randomizing the data
def randomizing(features,targets):
    data = list(zip(features, targets))
    #The list() constructor in python is used to create a list. 
    #Lists are ordered, mutable collections that can hold items of any type.
    #The zip() function combines multiple iterables (like lists or tuples) element-wise into tuples
    random.shuffle(data)
    #This is python's built in randomizing method
    features_shuffled, targets_shuffled = zip(*data)
    #The * operator is used for unpacking. 
    #You're simply asking python to unpack the list of tuples so that each tuple becomes a separate argument to zip()

    
#Previous implementation of the z-score normalizer. Less concise and mathematically incoherent    
def z_score_normalizer (features, scaling_feature_indices):
    features_modified = np.copy(features)
    feature_list = np.zeros((features.shape[0],len(scaling_feature_indices)))
    mean = np.zeros((len(scaling_feature_indices)))
    st_deviation = np.zeros((len(scaling_feature_indices)))
    for indx,i in enumerate(scaling_feature_indices):
        feature = features[:,indx]
        feature_list[:,i] = feature
        mean[i] = np.mean(feature)
        std = 0
        for j in range(feature.shape[0]):
            std += (feature[j]-mean[i])**2
        st_deviation[i] = std/len(feature)
        st_deviation[i] = st_deviation[i]**(1/2)
    for k in range(features_modified.shape[0]):
        for l in range(features_modified.shape[1]):
                if l in scaling_feature_indices:
                    features_modified[k,l] = (features_modified[k,l] - mean[l])/st_deviation[l]
    return features_modified, mean, st_deviation


#Gradient calculation without vectorized dj_dw
def grad_cal (w, b, features, targets):
    m,n = features.shape
    error = (np.dot(features,w)+b)-targets
    dj_dw = np.zeros((n,))
    for i in range(m):
        for j in range(n):
            dj_dw[j] += error[i]*features[i,j]
    dj_dw /= m
    dj_db = np.mean(error)
    return dj_dw, dj_db

#Implementing the gradient descent algorithm with J_history and terminal outputs
def grad_desc (features, targets, w_in, b_in, alpha, iters, compute_gradient, cost):
    w = w_in.copy()
    b = b_in
    J_history = []
    for i in range(iters):
        dj_dw, dj_db = compute_gradient(w,b,features,targets)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        J_history.append(cost(w,b,features,targets))
        #Keeps track of how the cost varies with the number of iterations
        if i % 100 == 0 or i == iters -1:
            print(f"Iteration : {i}    Cost: {J_history[-1]}")
        #If the number of iterations is a multiple of 100 or if the number of iterations is the last iteration (note that i goes from 0 to iters so iters is not the last value of i) print the iteration count and the associated cost
        if i > 0 and abs(J_history[-1] - J_history[-2])<1e-8:
        #If atleast one iteration has passed and the absolute value of the difference of the last two entries of J_history is less than 1e-8, print convergence achieved and break the loop
            print("Convergence Achieved")
            break
    return w,b, J_history