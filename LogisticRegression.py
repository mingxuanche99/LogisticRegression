import numpy as np
def sigmod(x):
    return 1/(1+np.exp(-x))
class LogisticRegression:
    def __init__(self,  learn_rate=0.01, iter=10000):
        self.learn_rate=learn_rate
        self.iter=iter
        self.W=np.zeros((self.train_x.shape[1],1))
        self.b=0
    def fit(self,X_train,Y_train):
    #forward proagate
        res=sigmod(np.dot(self.W.T,self.train_x)+self.b)

    def predict(self,X):
        pass

    def score(self,X_test,Y_test):
        pass


