# Logistic Regression

## 1. User Guide:
### **1.1 Class**
#### LogisticRegression.LogisticRegression(learn_rate=0.001, iter=10000)
Our tool is a class described in LogisticRegression.py, which can achieve Logistic Regression and visualization of data point, results and effect on cost. The two parameters that users can tweak are learn_rate(learning rate in gradiant decent) and iter(iteration in training). Users can use this class by:
>from LogisticRegression import LogisticRegression
### **1.2 Method**
#### fit(X_train,Y_train)
This method is used for training model and plot the effect on cost function. The inputs are X_train and Y_train, X_train should be ndarray of shape (num_sample, dim), Y_train should be ndarray of shape (num_sample, ). Users can use this method by:
>LogisticRegression.fit(X_train,Y_train)
#### predict(X)
This method is used for predict class labels for samples in X.The input X should be ndarray of shape (num_sample, dim). Users can use this method by:
>LogisticRegression.predict(X)
#### score(X_test,Y_test)
This method is used for return the mean accuracy on the given test data and plot the data points with predicted labels and ground truth labels. The inputs are X_test and Y_test, X_test should be ndarray of shape (num_sample, dim), Y_test should be ndarray of shape (num_sample, ). Users can use this method by:
>LogisticRegression.score(X_train,Y_train)
## 2. Concept and Implementation details:

### **2.2 Model**

The logistic model is a statistical model that models the probability of an event taking place by having the log-odds for the event be a linear combination of one or more independent variables. Logistic regression is estimating the parameters of a logistic model (the coefficients in the linear combination).

The logistic function is of the form:

$p(x)=\frac{1}{1+e^{-(Wx+b)}}$ 

where $W$ and $b$ are the parameters to be estimated. Thus, in implementation, we firstly define self.W and self.b.

### **2.2 Fit**
Given the $p(x)$ is the predicted probability and $y$ is the ground-truth. Define $lr$ as the learning rate during optimization.

**2.2.1 Cost function**: we use "negative log-likelihood" as the cost function to measure the goodness of fit for logistic regression, which is also termed as log loss.

l = cost = $-y(\log y) - (1-y)\log(1-p(x))$

**2.2.2 Parameter update**:


The gradient w.r.t $W$ and $b$ is computed as:

$\frac{\delta l}{\delta b}  = y-p(x)$

$\frac{\delta l}{\delta W}  = (y-p(x))x$

Then we apply gradient descent to update parameters:

$W = W - lr * \frac{\delta l}{\delta W}$

$b = b - lr * \frac{\delta l}{\delta b}$

### **2.2.3 Prediction**
The updated parameters $W$ and $b$ are entered into logistic regression equation to estimate the probability.

$p(x)=\frac{1}{1+e^{-(Wx+b)}}$ 


## 3. Environment preparation:

We require an environment of python 3.8, with packages including numpy, matplotlib, sklearn.


## 4. Demo dataset:

The demo is described in main.py. We load the dataset from sklearn.datasets

> datasets.load_breast_cancer()

This is a breast cancer wisconsin dataset for binary classification. The dataset are with 569 samples, each of which has a feature length of 30. 

The objective of logistic function is to predict a probability between 0-1 for negative/positive classification.
