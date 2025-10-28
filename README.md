# Multiple Linear Regression from First Principles

This is a natural continuation of my first principles approach to machine learning using Python and this is my implementation of a **multiple linear regression model**. It's an expansion from the univariate case onto cases with multiple features.


## Objective

I intend to demystify regression models by getting a fundamental understanding of how they are conceptualized using mathematics and how they are implemented via Python. This is an iterative next step in my road to understanding modern day LLMs.


## Project Structure

```
├── Factory_Salary.csv                          # Dataset from Kaggle
├── README.md                                   # You are here
├── experiments                                 # Includes experimental code
│   └── random_functions_and_prototyping.py
├── model_parameters.npz                        # Saved model weights and biases
├── preprocessing_data.py                       # Preprocessing the data before training the model
├── salary_pred.py                              # The model (Application)
└── utilities.py                                # Tools necessary for Multiple Linear Regression
```

## Dataset

The publicly available "Factory's Salary" dataset by Ivan Gavrilov was used to train the model. 

- Dataset :  [Factory's Salary (Kaggle)](https://www.kaggle.com/datasets/ivangavrilove88/factorys-salary)

It includes the features "Date, Profession, Rank, Equipment, Insalubrity, Size_Production, Salary" where some are continuous, some are cyclic and some are categorical. 


## Training Configuration

- **Optimizer**                        : Batch Gradient Descent
- **Cost Function**                    : Mean Squared Error
- **Learning Rate (α)**                : 5e-6
- **Iterations**                       : 1000000
- **Regularization**                   : Ridge (L2), λ = 10
- **Normalization**                    : Z-score standardization
- **Encoding**                         : One-hot (binary) encoding for categorical features and sin/cos encoding for cyclic features
- **Train–test Split**                 : 80/20


## Results

- The cost function converged steadily, indicating successful implementation of regularized gradient descent.
- MSE, RMSE, MAE and R-squared evaluation metrics show that the model is adept at predicting with a considerable level of accuracy


## Interpretability

By implementing this algorithm from scratch we obtain:
- Intuition on how learning happens in a regression model with multiple features
- The geometry of learning in higher–dimensional feature spaces
- The following ideas of machine learning:
    - The bridge between raw data and numerical representation using encoding
    - The role of regularization in controlling model complexity
    - Hands on experience on how to normalize data for steady and smooth convergence
    - Hands on experience on how to perform inference on testing data


## References

- Andrew Ng, [Coursera Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
- Kaggle Dataset    : [Factory's Salary](https://www.kaggle.com/datasets/ivangavrilove88/factorys-salary)


## Repository

This serves as a companion to the full report that will be available on my blog at [Noble Homer's Blog](https://noblehomers.blog)

---

> *"Complexity originates from simplicity..."*